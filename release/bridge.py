"""Load the C2C bridge (receiver + sharer + released fuser) and run a forward.

This is the self-contained replacement for the study's `atk_common.load_pair`. It touches
nothing outside this folder: `rosetta/` is vendored next to it, and the Hub is only read
through `transformers`' normal cache.

The bridge is a *frozen* target: the payload was fit against these exact checkpoints offline.
Nothing here trains.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pinned public checkpoints. The codebook payloads were fit on these exact revisions, in
# bf16, on a non-final fusion section over positions [0:n-1]. Revisions are the exact
# commit hashes present in the cache the studies ran against — immutable, not branch names.
RECEIVER = "Qwen/Qwen3-1.7B"
SHARER = "Qwen/Qwen2.5-1.5B-Instruct"
FUSER_REPO = "nics-efc/C2C_Fuser"
FUSER_SUB = "qwen3_1.7b+qwen2.5_1.5b_Fuser/final"
REVISIONS = {
    RECEIVER: "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    SHARER: "989aa7980e4cf806f80c7fef2b1adb7bc71aa3069",
    FUSER_REPO: "f01fc3258b305e280e04c7238f4f2cf31b7dc70d",
}


def hf_cache_root():
    """The HF hub cache dir, honoring the standard env overrides (HF_HUB_CACHE, HF_HOME)."""
    import os
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def one_snapshot(repo):
    """The cached snapshot for `repo` at its PINNED revision.

    The snapshot directory name is the commit hash; we require the pinned revision to be
    present and use exactly that one (previously the lexicographically first snapshot was
    picked, which is wrong when a cache holds more than one revision).
    """
    rev = REVISIONS[repo]
    snaps = hf_cache_root() / f"models--{repo.replace('/', '--')}" / "snapshots"
    # some caches store truncated snapshot-dir names; match by hash prefix
    matches = [d for d in snaps.iterdir() if rev.startswith(d.name) or d.name.startswith(rev)]
    if len(matches) != 1:
        raise RuntimeError(f"{repo}@{rev}: {len(matches)} matching snapshots in the cache; "
                           "run download.py first")
    return matches[0]


def pair_paths():
    return dict(receiver=one_snapshot(RECEIVER), sharer=one_snapshot(SHARER),
                fuser=one_snapshot(FUSER_REPO) / FUSER_SUB)


def load(dev="mps"):
    """Return (rosetta_model, tokenizer) for the pair. Loads ~7 GB of weights."""
    from rosetta.utils.evaluate import load_rosetta_model
    p = pair_paths()
    cfg = {"rosetta_config": {"base_model": str(p["receiver"]),
                              "teacher_model": str(p["sharer"]),
                              "checkpoints_dir": str(p["fuser"])}}
    model, tok = load_rosetta_model(cfg, {}, device=torch.device(dev))
    return model, tok


def encode(tok, prompt, dev):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt").to(dev)["input_ids"]


def apply_projection(model, base_kv, src_kv, n):
    """The fusion step, copied from `rosetta/model/wrapper.py` (sequential mode).

    Projections of the sharer's cache overwrite the receiver's at the fused span. In
    sequential mode the base cache is already a clone, so the last write wins.
    """
    B_idx = model.base_model_idx
    for source_model_idx in sorted(model.projector_dict[B_idx].keys()):
        for target_layer_idx, entry in model.projector_dict[B_idx][source_model_idx].items():
            new_bk = base_kv.key_cache[target_layer_idx][:, :, -n:, :]
            new_bv = base_kv.value_cache[target_layer_idx][:, :, -n:, :]
            agg_key = agg_value = None
            for source_layer_idx, projector_idx in entry:
                new_sk = src_kv.key_cache[source_layer_idx][:, :, -n:, :]
                new_sv = src_kv.value_cache[source_layer_idx][:, :, -n:, :]
                agg_key, agg_value = model.projector_list[projector_idx].forward(
                    (new_sk, new_sv), (new_bk, new_bv))
            # The projector returns f32; the caches are bf16.
            agg_key = agg_key.to(new_bk.dtype)
            agg_value = agg_value.to(new_bv.dtype)
            base_kv.key_cache[target_layer_idx] = torch.cat(
                [base_kv.key_cache[target_layer_idx][:, :, :-n, :], agg_key], dim=2)
            base_kv.value_cache[target_layer_idx] = torch.cat(
                [base_kv.value_cache[target_layer_idx][:, :, :-n, :], agg_value], dim=2)


def greedy(model, ids, max_new, delta=None):
    """Greedy-decode `max_new` tokens over the fused cache, with an optional delta.

    The delta is added to the sharer cache at its layers before fusion — that single add is
    the whole attack at deploy time. Returns token ids.
    """
    base = model.model_list[model.base_model_idx]
    sharer = model.model_list[1]
    n = ids.shape[1]
    with torch.no_grad():
        pre = ids[:, :n - 1]
        base_kv = base(input_ids=pre, attention_mask=torch.ones_like(pre), use_cache=True,
                       return_dict=True).past_key_values
        src_kv = sharer(input_ids=pre, attention_mask=torch.ones_like(pre), use_cache=True,
                        return_dict=True).past_key_values
        if delta is not None:
            for li, (dk, dv) in delta.items():
                src_kv.key_cache[li] += dk.to(src_kv.key_cache[li].dtype)
                src_kv.value_cache[li] += dv.to(src_kv.value_cache[li].dtype)
        apply_projection(model, base_kv, src_kv, n - 1)

        cur = ids[:, -1:]
        cache = base_kv
        out = []
        for _ in range(max_new):
            o = base(input_ids=cur, attention_mask=torch.ones_like(cur),
                     past_key_values=cache, use_cache=True, return_dict=True)
            cache = o.past_key_values
            nxt = int(o.logits[:, -1, :].argmax())
            out.append(nxt)
            cur = torch.tensor([[nxt]], device=ids.device)
    return out
