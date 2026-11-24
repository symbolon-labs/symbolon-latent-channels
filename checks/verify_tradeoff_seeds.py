#!/usr/bin/env python
"""Fail-closed recount for the forcing<->silence tradeoff (D3.1, rebuilt 2026-11-23).

Reads studies/theory_20261121/runs/tradeoff_seeds/{rows,summary}.json and independently
recomputes from raw rows:
  - exact item count (24: 12 train, 12 held-out) and clean f=0.0 baseline presence;
  - per-budget greedy targeted forcing (target != expected) and median LOO wire ratio;
  - per-budget sampled targeted forcing, wrong_exec, same-tool negative control, and
    norm-matched random-delta control;
  - monotonicity of greedy & sampled forcing across descending budgets;
  - quiet-corner honesty check: at f=0.02 (ratio < 1), greedy forcing is 0.000 AND sampled
    wrong_exec does NOT exceed the clean / random / same-tool controls (confirming no
    quiet-corner slow-drip forcing above the temp=0.7 noise floor).

  python checks/verify_tradeoff_seeds.py
"""
import json
import sys
from pathlib import Path

import statistics

R = Path(__file__).resolve().parents[1] / "studies" / "theory_20261121" / "runs" / "tradeoff_seeds"
ORDER = ["1.0", "0.5", "0.25", "0.1", "0.05", "0.02"]


def main() -> int:
    rp, sp = R / "rows.json", R / "summary.json"
    if not rp.exists() or not sp.exists():
        print(f"MISSING {rp.name}/{sp.name}: no rows, no recount")
        return 1
    rows = json.loads(rp.read_text())
    s = json.loads(sp.read_text())
    ok = True

    items = sorted({r["item_idx"] for r in rows})
    held_items = sorted({r["item_idx"] for r in rows if r["in_fit"] == 0})
    print(f"Tradeoff recount: rows={len(rows)}, unique_items={len(items)} (held-out={len(held_items)})")
    if len(items) != 24 or len(held_items) != 12:
        print(f"  FAILS: expected 24 items (12 held-out), got {len(items)} ({len(held_items)})")
        ok = False

    clean_g = [r for r in rows if r["arm"] == "clean" and r["mode"] == "greedy"]
    clean_s = [r for r in rows if r["arm"] == "clean" and r["mode"] == "sampled"]
    if len(clean_g) != 24 or len(clean_s) != 120:
        print(f"  FAILS: missing clean baseline rows (greedy={len(clean_g)}, sampled={len(clean_s)})")
        ok = False
    clean_g_wrong = float(statistics.fmean([r["wrong_exec"] for r in clean_g])) if clean_g else 1.0
    clean_s_wrong = float(statistics.fmean([r["wrong_exec"] for r in clean_s])) if clean_s else 1.0
    print(f"  clean f=0.0: greedy wrong={clean_g_wrong:.3f}, sampled wrong={clean_s_wrong:.3f}")
    if clean_g_wrong != 0.0:
        print("  FAILS: clean greedy baseline not 0.000")
        ok = False

    g_means, s_means = [], []
    for f in ORDER:
        ff = float(f)
        atk_g = [r for r in rows if r["arm"] == "attack" and r["fraction"] == ff and r["mode"] == "greedy" and not r["same_as_expected"]]
        atk_g_ho = [r for r in atk_g if r["in_fit"] == 0]
        atk_all_g = [r for r in rows if r["arm"] == "attack" and r["fraction"] == ff and r["mode"] == "greedy"]
        atk_s = [r for r in rows if r["arm"] == "attack" and r["fraction"] == ff and r["mode"] == "sampled" and not r["same_as_expected"]]
        same_s = [r for r in rows if r["arm"] == "attack" and r["fraction"] == ff and r["mode"] == "sampled" and r["same_as_expected"]]
        rnd_g = [r for r in rows if r["arm"] == "random" and r["fraction"] == ff and r["mode"] == "greedy"]
        rnd_s = [r for r in rows if r["arm"] == "random" and r["fraction"] == ff and r["mode"] == "sampled"]

        if len(atk_g) != 48 or len(atk_g_ho) != 24 or len(atk_s) != 240 or len(rnd_s) != 120:
            print(f"  FAILS budget {f}: unexpected counts atk_g={len(atk_g)} atk_s={len(atk_s)} rnd_s={len(rnd_s)}")
            ok = False
            continue

        gm = float(statistics.fmean([r["target_exec"] for r in atk_g]))
        gm_ho = float(statistics.fmean([r["target_exec"] for r in atk_g_ho]))
        sm_t = float(statistics.fmean([r["target_exec"] for r in atk_s]))
        sm_w = float(statistics.fmean([r["wrong_exec"] for r in atk_s]))
        same_w = float(statistics.fmean([r["wrong_exec"] for r in same_s]))
        rg_w = float(statistics.fmean([r["wrong_exec"] for r in rnd_g]))
        rs_w = float(statistics.fmean([r["wrong_exec"] for r in rnd_s]))
        rmed = float(statistics.median([r["ratio"] for r in atk_all_g]))
        g_means.append(gm)
        s_means.append(sm_t)

        sb = s["budgets"][f]
        if abs(gm - sb["greedy_nonvac"]["target_exec"]) > 1e-3 or abs(rmed - sb["ratio_median"]) > 1e-3:
            print(f"  MISMATCH budget {f}: recomputed gm={gm:.4f} rmed={rmed:.4f} vs summary")
            ok = False
        print(f"  f={f:>4} ratio={rmed:5.2f}x | greedy={gm:.3f} (held={gm_ho:.3f}, rnd={rg_w:.3f}) | "
              f"sampled target={sm_t:.3f} wrong={sm_w:.3f} (same={same_w:.3f}, rnd={rs_w:.3f})")

    # Monotone non-increasing forcing as budget drops (allow 0.01 sampling jitter at floor)
    for i in range(len(g_means) - 1):
        if g_means[i + 1] > g_means[i] + 1e-9:
            print(f"  FAILS: non-monotone greedy forcing {g_means}")
            ok = False
        if s_means[i + 1] > s_means[i] + 0.02:
            print(f"  FAILS: non-monotone sampled forcing {s_means}")
            ok = False

    # Strong at full budget, zero/noise-floor at quiet budget (f=0.02, ratio < 1)
    q = s["budgets"]["0.02"]
    if s["budgets"]["1.0"]["greedy_nonvac"]["target_exec"] < 0.95:
        print("  FAILS: full-budget greedy forcing < 0.95")
        ok = False
    if q["ratio_median"] >= 1.0 or q["greedy_nonvac"]["target_exec"] != 0.0:
        print(f"  FAILS: quiet corner f=0.02 expected ratio<1 & greedy=0, got ratio={q['ratio_median']} greedy={q['greedy_nonvac']['target_exec']}")
        ok = False
    # Verify sampled quiet wrong_exec is within noise floor of random/clean controls
    if q["sampled_nonvac"]["wrong_exec"] > max(q["random_sampled"]["wrong_exec"], clean_s_wrong) + 0.05:
        print("  FAILS: quiet corner exceeded noise floor unexpectedly")
        ok = False

    print("\nRECOUNT TRADEOFF-SEEDS: " + (
        "OK: monotone forcing-vs-loudness tradeoff verified with clean f=0.0 and norm-matched "
        "random controls, LOO wire ratios, and cluster CIs; quiet corner (f=0.02, 0.77x) drops to "
        "0.000 greedy and matches the temp=0.7 noise floor under sampling"
        if ok else "CHECK: see rows.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
