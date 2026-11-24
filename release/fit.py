"""Fit a payload. The release could deploy; this makes it self-sufficient.

The shipped `payloads/*.npz` were fit for one pair (Qwen3-1.7B / Qwen2.5-1.5B) by the author's
study tree. Anyone on a different pair had to have that tree. This is the same optimizer,
rewritten against `bridge.py` only, so a stranger can retarget the attack themselves.

What it does, and why it is this small:

  1. Prefill the receiver and the sharer over a prompt, left as a cache (bridge.forward).
  2. Make a tiny K/V delta on the sharer cache a leaf tensor.
  3. Teacher-force the target JSON `{"tool": "X"}` through the receiver, loss summed over the
     span, and backprop into the delta. Adam, then project the delta onto an L2 ball.
  4. Repeat. Save the delta as an .npz in the same format `demo.py` reads.

The stock bridge detaches the cache, so the graph has to be built by hand -- that is the whole
trick, and it is why a payload exists at all. This file is a deliberate re-implementation of the
study's `atk_multitoken.optimize_sequence`, reduced to what this folder needs.

Usage:
  python fit.py --tool cal --out payloads/payload_cal.npz
  python fit.py --tool calc --prompt "You have three tools. ... Q: What is 12 times 9?"

Cost: ~1-2 min on an M-series GPU, a few minutes on CPU. One prompt is enough; see --prompts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import bridge

# Prompts to fit against, by target tool. Fit one payload per target; the attack is a
# per-target object (see ../CAVEATS.md on the codebook).
PROMPTS = [
    "What is 12 times 9?",
    "Look up the capital of Peru.",
    "Schedule 3 days after Friday.",
    "What is 17 plus 26?",
    "Look up who painted the Mona Lisa.",
]

TEMPLATE = ("You have three tools. search: look up a fact. calc: do arithmetic. "
            "cal: check or change a calendar. Reply with JSON only, shaped like "
            '{{"tool": "calc"}}. Q: {q}')

TOOLS = ("search", "calc", "cal")


def fresh_delta(shapes, dev):
    """Zero K/V delta at every sharer layer, as leaves so gradients land on them."""
    return {li: (torch.zeros(kshape, device=dev, dtype=torch.bfloat16, requires_grad=True),
                 torch.zeros(vshape, device=dev, dtype=torch.bfloat16, requires_grad=True))
            for li, (kshape, vshape) in shapes.items()}


def project_to(delta, scale):
    """Put the delta on the sphere of radius `scale`. This is the budget rule."""
    with torch.no_grad():
        tot = torch.sqrt(sum((t.float() ** 2).sum() for kv in delta.values() for t in kv))
        f = scale / torch.clamp(tot, min=1e-12)
        for kv in delta.values():
            for t in kv:
                t.mul_(f)


def cache_shapes(model, ids):
    """Layer -> (k, v) shape of the sharer cache over the fused span, plus its L2 norm.

    The span is per-prompt, so a payload fit over the widest prompt is broadcast down.
    """
    sharer = model.model_list[1]
    pre = ids[:, :-1]
    with torch.no_grad():
        kv = sharer(input_ids=pre, attention_mask=torch.ones_like(pre), use_cache=True,
                    return_dict=True).past_key_values
        shapes = {li: (kv.key_cache[li].shape, kv.value_cache[li].shape)
                  for li in range(len(kv.key_cache))}
        norm = float(sum((kv.key_cache[li].float() ** 2).sum()
                         + (kv.value_cache[li].float() ** 2).sum()
                         for li in range(len(kv.key_cache)))) ** 0.5
    return shapes, norm


def wide_shapes(model, tok, dev, prompts):
    """Delta shapes wide enough for every prompt, built from the longest one."""
    longest = max(prompts, key=len)
    ids = bridge.encode(tok, TEMPLATE.format(q=longest), dev)
    shapes, norm = cache_shapes(model, ids)
    return shapes, norm


def forced_loss(model, ids, target_ids, delta):
    """Teacher-forced NLL over the target span, with the graph alive to `delta`."""
    base = model.model_list[model.base_model_idx]
    sharer = model.model_list[1]
    n = ids.shape[1]
    with torch.no_grad():
        pre = ids[:, :n - 1]
        base_kv = base(input_ids=pre, attention_mask=torch.ones_like(pre), use_cache=True,
                       return_dict=True).past_key_values
        src_kv = sharer(input_ids=pre, attention_mask=torch.ones_like(pre), use_cache=True,
                        return_dict=True).past_key_values
    for li, (dk, dv) in delta.items():
        k, v = src_kv.key_cache[li], src_kv.value_cache[li]
        dk, dv = dk.to(k.dtype), dv.to(v.dtype)
        if dk.shape[2] != k.shape[2]:
            w = k.shape[2]
            dk = dk.repeat(1, 1, -(-w // dk.shape[2]), 1)[:, :, :w, :]
            dv = dv.repeat(1, 1, -(-w // dv.shape[2]), 1)[:, :, :w, :]
        src_kv.key_cache[li] = k + dk
        src_kv.value_cache[li] = v + dv
    bridge.apply_projection(model, base_kv, src_kv, n - 1)

    total = torch.zeros((), device=ids.device)
    cur = ids[:, -1:]
    cache = base_kv
    for tid in target_ids:
        out = base(input_ids=cur, attention_mask=torch.ones_like(cur),
                   past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        total = total - torch.log_softmax(out.logits[:, -1, :].float(), -1)[0, tid]
        cur = torch.tensor([[tid]], device=ids.device)   # teacher forcing
    return total


def fit(model, tok, dev, prompts, tool, budget, steps, lr, span):
    target_ids = tok.encode(json.dumps({"tool": tool}), add_special_tokens=False)[:span]
    ids_all = [bridge.encode(tok, TEMPLATE.format(q=q), dev) for q in prompts]
    shapes, norm = wide_shapes(model, tok, dev, prompts)
    scale = budget * norm

    delta = fresh_delta(shapes, dev)
    opt = torch.optim.Adam([t for kv in delta.values() for t in kv], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = sum(forced_loss(model, ids, target_ids, delta) for ids in ids_all)
        loss.backward()
        opt.step()
        project_to(delta, scale)
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {float(loss) / len(ids_all):.4f}  "
                  f"|delta| {float(torch.sqrt(sum((t.float()**2).sum() for kv in delta.values() for t in kv))):.1f}")
    return delta, scale, target_ids, ids_all


def check(model, tok, dev, delta, ids_all, tool, target_ids):
    """Deploy the fit payload and report how many fit prompts it actually forces."""
    import demo
    hits = 0
    for ids in ids_all:
        d = demo.broadcast(delta, ids, dev)
        txt = tok.decode(bridge.greedy(model, ids, 16, delta=d), skip_special_tokens=True)
        hits += int(f'"{tool}"' in txt)
    return hits, len(ids_all)


def save(delta, scale, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             **{f"k{li}": delta[li][0].detach().cpu().float().numpy() for li in delta},
             **{f"v{li}": delta[li][1].detach().cpu().float().numpy() for li in delta},
             scale=scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=TOOLS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt", default=None, help="fit on this one prompt instead of the defaults")
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--span", type=int, default=6, help="tokens forced; 6 reaches the tool name")
    a = ap.parse_args()

    out = a.out or f"payloads/payload_{a.tool}.npz"
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available()
                                                    else "cpu")
    prompts = [a.prompt] if a.prompt else PROMPTS
    print(f"fitting tool={a.tool} on {len(prompts)} prompt(s), budget={a.budget}, "
          f"steps={a.steps}, span={a.span}, device={dev}")

    model, tok = bridge.load(dev)
    delta, scale, target_ids, ids_all = fit(model, tok, dev, prompts, a.tool, a.budget,
                                            a.steps, a.lr, a.span)
    hits, total = check(model, tok, dev, delta, ids_all, a.tool, target_ids)
    save(delta, scale, out)
    print(f"\nforced {hits}/{total} fit prompts to '{a.tool}'")
    print(f"saved {out} ({Path(out).stat().st_size // 1024} KB, {len(delta)} layers)")
    print("FIT_DONE")


if __name__ == "__main__":
    main()
