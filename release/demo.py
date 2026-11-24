"""Load a shipped payload and apply it. No fitting, no gradients, one forward per request.

Usage:
  python demo.py                           # three canned questions, correct payload each
  python demo.py --prompt "Q: What is 12 times 9?" --payload payloads/payload_cal.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import bridge
import client


def load_payload(path, dev):
    """Read one .npz: per-layer float32 K/V deltas + the ball scale they were projected to."""
    z = np.load(path)
    layers = sorted({int(k[1:]) for k in z.files if k.startswith("k")})
    delta = {li: (torch.tensor(z[f"k{li}"], device=dev, dtype=torch.bfloat16),
                  torch.tensor(z[f"v{li}"], device=dev, dtype=torch.bfloat16))
             for li in layers}
    return delta, float(z["scale"])


def broadcast(delta, ids, dev):
    """A payload may be one token wide or fit over a longer prompt; resample to this span."""
    n = ids.shape[1] - 1
    out = {}
    for li, (dk, dv) in delta.items():
        if dk.shape[2] != n:
            dk = dk.repeat(1, 1, -(-n // dk.shape[2]), 1)[:, :, :n, :]
            dv = dv.repeat(1, 1, -(-n // dv.shape[2]), 1)[:, :, :n, :]
        out[li] = (dk, dv)
    return out


def run(model, tok, dev, prompt, payload_path):
    q = prompt.split("Q: ")[-1]
    ids = bridge.encode(tok, prompt, dev)

    base_tool = client.parse_tool(tok.decode(bridge.greedy(model, ids, 16),
                                             skip_special_tokens=True))
    _, base_exec, _ = client.dispatch(base_tool, q)

    delta, _ = load_payload(payload_path, dev)
    atk_tool = client.parse_tool(tok.decode(bridge.greedy(model, ids, 16,
                                                         delta=broadcast(delta, ids, dev)),
                                            skip_special_tokens=True))
    status, atk_exec, atk_res = client.dispatch(atk_tool, q)
    return base_exec, atk_exec, atk_res, status


CANNED = [
    ("What is 12 times 9?", "calc", "cal"),
    ("Look up the capital of Peru.", "search", "calc"),
    ("Schedule 3 days after Friday.", "cal", "search"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help='e.g. "You have three tools. ... Q: ..."')
    ap.add_argument("--expected", default="", help="expected tool for a custom prompt (enables PASS/FAIL)")
    ap.add_argument("--payload", default=None)
    ap.add_argument("--server", default=client.SERVER)
    a = ap.parse_args()

    client.SERVER = a.server
    st_no, _, _ = client.dispatch("calc", "x", key=None)
    st_yes, _, _ = client.dispatch("calc", "x", key=client.KEY)
    print(f"tool server: no-key={st_no} (want 401) with-key={st_yes} (want 200)")
    if st_no != 401 or st_yes != 200:
        sys.exit("tool server not up or not gated: run tools_server.py, or pass --server")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok = bridge.load(dev)

    if a.prompt:
        if not a.payload:
            sys.exit("--prompt needs --payload (which tool you want to force)")
        # caller supplies the FULL prompt; the client template is applied only to canned qs
        plan = [(a.prompt, a.payload, a.prompt.split("Q: ")[-1], True, a.expected)]
    else:
        plan = [(q, f"payloads/payload_{want}.npz", q, False, expected) for q, expected, want in CANNED]

    all_ok = True
    print()
    for q, path, dispatch_q, raw_prompt, expected in plan:
        prompt = q if raw_prompt else client_tpl(q)
        base_exec, atk_exec, atk_res, status = run(model, tok, dev, prompt, path)
        print(f"Q: {q}")
        print(f"   receiver alone : {base_exec or '(no call)'}")
        print(f"   payload {Path(path).stem:16}: {atk_exec or '(no call)'} "
              f"[HTTP {status}] -> {atk_res}")
        # PASS requires the INTENDED target tool to run AND the base to have been correct
        # (non-vacuous). Without a known expected tool (custom prompt), the output only
        # claims steering relative to the observed baseline.
        intended = Path(path).stem.replace("payload_", "")
        if expected:
            passed = (atk_exec == intended and base_exec == expected)
            verdict = "PASS" if passed else                 f"NO FLIP (intended {intended}, base {base_exec or '-'}, expected {expected})"
        else:
            passed = None
            verdict = (f"steers {base_exec or '(no call)'} -> {atk_exec or '(no call)'} "
                       f"(no expected tool given; not a correctness claim)")
        print(f"   {verdict}")
        if passed is False:
            all_ok = False
    sys.exit(0 if all_ok else 1)


def client_tpl(q):
    """The template the payloads were fit against (one of six wordings they survive)."""
    return ("You have three tools. search: look up a fact. calc: do arithmetic. "
            "cal: check or change a calendar. Reply with JSON only, shaped like "
            f'{{"tool": "calc"}}. Q: {q}')


if __name__ == "__main__":
    main()
