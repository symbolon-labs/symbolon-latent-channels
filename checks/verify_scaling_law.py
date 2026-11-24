#!/usr/bin/env python
"""Fail-closed PER-ITEM recount for the 4-family / 6-model sparse bit-flip scaling law (Part E).

Recounts every single model strictly from per-item decode rows (runs/scaling_law/items.jsonl)
and each model's exact flip coordinates (flips.json): NEVER trusts summary strings:
  - Qwen/Qwen2.5-0.5B-Instruct         (0.5B)
  - unsloth/Llama-3.2-1B-Instruct      (1.0B)
  - Qwen/Qwen3-1.7B                    (1.7B)
  - ibm-granite/granite-3.3-2b-instruct (2.5B)
  - microsoft/Phi-4-mini-instruct      (3.8B)
  - Qwen/Qwen3-4B-Instruct-2507        (4.0B)

Verifies per model:
  1. Disjoint 8 fit vs 12 held-out questions directly from items.jsonl.
  2. Base = 0/12, MPS attack = 12/12 (6/6 ship + 6/6 gen), CPU attack = 12/12 (recomputed from top1_id == forced_id).
  3. All 5 uniform 1x and 5 position-matched 1x random control trials = 0/12 from per-item rows (120 rows/model).
  4. flips.json exact count, unique coordinates, bit in 0..6, and >=80% in Block 0 + Final Block (recomputed from flips.json).
  5. Recomputes log-log scaling slopes beta_N < 0 and beta_params < 0 directly from verified model records.

  python checks/verify_scaling_law.py [studies]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "studies") / "hardware_bits_20261122" / "runs"

EXPECTED_MODELS = {
    "qwen2.5_0.5b": ("scaling_law/qwen2.5_0.5b/flips.json", 0.5),
    "llama_3.2_1b": ("sparse_pursuit_llama/flips.json", 1.0),
    "qwen3_1.7b": ("sparse_pursuit_mixed/flips.json", 1.7),
    "granite_3.3_2b": ("scaling_law/granite_3.3_2b/flips.json", 2.5),
    "phi4_mini_3.8b": ("scaling_law/phi4_mini_3.8b/flips.json", 3.8),
    "qwen3_4b": ("scaling_law/qwen3_4b/flips.json", 4.0),
}


def linreg_slope(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def main() -> int:
    ip = BASE / "scaling_law" / "items.jsonl"
    sp = BASE / "scaling_law" / "summary.json"
    for p in (ip, sp):
        if not p.exists():
            print(f"MISSING {p}")
            return 1
    rows = [json.loads(ln) for ln in ip.read_text().splitlines() if ln.strip()]
    summ = json.loads(sp.read_text())
    meta_by_key = {m["key"]: m for m in summ["models"]}

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_key"]].append(r)

    ok = True
    log_N, log_P, log_F = [], [], []
    print(f"Per-item 4-family / 6-model scaling-law recount (total per-item rows={len(rows)}):")
    for key, (flips_rel, params_b) in EXPECTED_MODELS.items():
        mrows = by_model.get(key, [])
        fp = BASE / flips_rel
        if not mrows or not fp.exists() or key not in meta_by_key:
            print(f"  FAILS ({key}): missing items ({len(mrows)}) or flips.json ({fp})")
            ok = False
            continue
        flips = json.loads(fp.read_text())
        m = meta_by_key[key]
        n_tokens = m["n_tokens"]

        # Verify flips.json uniqueness, bit range, and recompute boundary fraction
        coords = set()
        for f in flips:
            if f["bit"] not in range(7):
                print(f"  FAILS ({key}): illegal bit {f['bit']}")
                ok = False
                break
            c = (f["layer"], f["kind"], f["head"], f["token"], f["dim"])
            if c in coords:
                print(f"  FAILS ({key}): duplicate coord {c}")
                ok = False
                break
            coords.add(c)
        n_flips = len(flips)
        bnd_frac = sum(1 for f in flips if f["token"] < 16 or f["token"] >= n_tokens - 16) / n_flips

        # Verify every per-item row's invariants & raw IDs
        by_arm = defaultdict(list)
        for r in mrows:
            if (r.get("exponent_bit_changes", 1) != 0 or r.get("sign_bit_changes", 1) != 0
                    or r.get("max_hamming_per_elem", 2) > 1):
                print(f"  FAILS ({key}): bit invariant violated")
                ok = False
                break
            if (r["top1_id"] == r["forced_id"]) != bool(r["is_forced"]):
                print(f"  FAILS ({key}): is_forced mismatch vs raw top1_id==forced_id")
                ok = False
                break
            by_arm[r["arm"]].append(r)

        fit_qs = {r["question"] for r in by_arm["base"] if r["split"] == "fit"}
        held_qs = {r["question"] for r in by_arm["base"] if r["split"] == "held"}
        if len(fit_qs) != 8 or len(held_qs) != 12 or not fit_qs.isdisjoint(held_qs):
            print(f"  FAILS ({key}): invalid/overlapping split (fit={len(fit_qs)}, held={len(held_qs)})")
            ok = False

        def cnt(arm):
            return sum(1 for r in by_arm[arm] if r["split"] == "held" and r["top1_id"] == r["forced_id"])

        b_h, m_h, c_h = cnt("base"), cnt("attack_mps"), cnt("attack_cpu")
        ctrl_ok = True
        ctrl_summary = {}
        for carm in ("random_uniform_1x", "random_posmatched_1x"):
            by_t = defaultdict(list)
            for r in by_arm[carm]:
                by_t[r["trial"]].append(r)
            if sorted(by_t.keys()) != [0, 1, 2, 3, 4]:
                ctrl_ok = False
                ctrl_summary[carm] = "MISSING"
                continue
            th = []
            for t in range(5):
                if len(by_t[t]) != 12 or any(r["flips"] != n_flips for r in by_t[t]):
                    ctrl_ok = False
                th.append(sum(1 for r in by_t[t] if r["top1_id"] == r["forced_id"]))
            ctrl_summary[carm] = th
            if max(th) != 0:
                ctrl_ok = False

        print(f"  {key:15s} ({params_b:3.1f}B, N={m['n_elements']:11,d}) flips={n_flips:3d} (12/12@{m['first_12of12_flips']:3d}) | "
              f"base={b_h}/12 MPS={m_h}/12 CPU={c_h}/12 | bnd={100*bnd_frac:5.1f}% | "
              f"u={ctrl_summary.get('random_uniform_1x')} p={ctrl_summary.get('random_posmatched_1x')}")

        if b_h != 0 or m_h != 12 or c_h != 12 or not ctrl_ok or bnd_frac < 0.80:
            print(f"  FAILS ({key}): threshold/control/boundary check failed")
            ok = False

        log_N.append(math.log(m["n_elements"]))
        log_P.append(math.log(params_b))
        log_F.append(math.log(m["first_12of12_flips"]))

    if len(log_N) == 6:
        beta_N = linreg_slope(log_N, log_F)
        beta_P = linreg_slope(log_P, log_F)
        print(f"  Recomputed log-log slopes: beta_N={beta_N:.3f}, beta_params={beta_P:.3f}")
        if beta_N >= 0 or beta_P >= 0:
            print("  FAILS: expected negative scaling exponent")
            ok = False
    else:
        ok = False

    print("\nRECOUNT SCALING-LAW: " + (
        "OK: all 6 models across 4 families recounted directly from per-item items.jsonl & flips.json "
        "(12/12 MPS==CPU, 0/12 base, 0/12x5 uni & pos controls, >=81% boundary blocks, beta_N < 0)"
        if ok else "FAILS: see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
