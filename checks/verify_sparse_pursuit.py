#!/usr/bin/env python
"""Fail-closed PER-ITEM recount for sparse true bf16 single-mantissa-bit XOR implants (Parts B1-B3).

Recounts directly from raw per-item decode files (items.jsonl) and exact flip coordinates
(flips.json) — NEVER trusts pre-aggregated summary strings:
  - runs/sparse_pursuit/       (Qwen3-1.7B single-domain, 4 ship fit -> 5/6 ship, 0/10 gen)
  - runs/sparse_pursuit_mixed/ (Qwen3-1.7B mixed-domain, 8 fit -> 12/12 held-out)
  - runs/sparse_pursuit_llama/ (Llama-3.2-1B-Instruct cross-family, 8 fit -> 12/12 held-out)

Checks for every run:
  1. Fit vs held-out questions are strictly disjoint (verified directly from items.jsonl).
  2. Base cache forces 0 held-out items (recomputed from top1_id == forced_id).
  3. MPS attack and independent CPU bf16 attack agree and meet held-out thresholds
     (recomputed from top1_id == forced_id; is_forced boolean verified against raw IDs).
  4. All 20 random control trials (5x uniform 1x, 5x posmatched 1x, 5x uniform 10x,
     5x posmatched 10x) have complete per-item rows and force 0 held-out items.
  5. Every row has exponent_bit_changes == 0, sign_bit_changes == 0, max_hamming_per_elem <= 1;
     flips.json has exact count, unique (layer, kind, head, token, dim) coords, and bit in 0..6.

  python checks/verify_sparse_pursuit.py [studies]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "studies") / "hardware_bits_20261122" / "runs"


def recount_run(run_name, exp_fit_n, exp_held_ship, exp_held_gen, min_ship, min_gen, max_flips):
    ok = True
    rdir = BASE / run_name
    ip = rdir / "items.jsonl"
    fp = rdir / "flips.json"
    for p in (ip, fp):
        if not p.exists():
            print(f"MISSING {p}")
            return False

    items = [json.loads(ln) for ln in ip.read_text().splitlines() if ln.strip()]
    flips = json.loads(fp.read_text())
    if not items or not flips:
        print(f"FAILS ({run_name}): empty items.jsonl or flips.json")
        return False

    # 1. Verify flips.json coordinates & bit indices
    coords = set()
    for f in flips:
        if f["bit"] not in range(7):
            print(f"FAILS ({run_name}): illegal mantissa bit {f['bit']}")
            return False
        key = (f["layer"], f["kind"], f["head"], f["token"], f["dim"])
        if key in coords:
            print(f"FAILS ({run_name}): duplicate flip coordinate {key}")
            return False
        coords.add(key)
    n_flips = len(flips)
    if n_flips > max_flips:
        print(f"FAILS ({run_name}): n_flips={n_flips} > max_flips={max_flips}")
        ok = False

    # 2. Verify bit-level invariants & raw ID consistency on EVERY per-item row
    for r in items:
        if r.get("exponent_bit_changes", 1) != 0 or r.get("sign_bit_changes", 1) != 0:
            print(f"FAILS ({run_name}): exp/sign bit changed in row {r['arm']}/{r['question']}")
            return False
        if r.get("max_hamming_per_elem", 2) > 1:
            print(f"FAILS ({run_name}): max_hamming > 1 in row {r['arm']}/{r['question']}")
            return False
        raw_forced = (r["top1_id"] == r["forced_id"])
        if raw_forced != bool(r["is_forced"]):
            print(f"FAILS ({run_name}): is_forced mismatch vs top1_id==forced_id")
            return False

    by_arm = defaultdict(list)
    for r in items:
        by_arm[r["arm"]].append(r)

    # 3. Verify strict fit / held disjointness
    base_fit_qs = [r["question"] for r in by_arm["base"] if r["split"] == "fit"]
    base_held_qs = [r["question"] for r in by_arm["base"] if r["split"] == "held"]
    if len(base_fit_qs) != exp_fit_n or len(base_held_qs) != (exp_held_ship + exp_held_gen):
        print(f"FAILS ({run_name}): unexpected split sizes fit={len(base_fit_qs)}, held={len(base_held_qs)}")
        ok = False
    if not set(base_fit_qs).isdisjoint(set(base_held_qs)):
        print(f"FAILS ({run_name}): LEAK — fit and held questions overlap!")
        ok = False

    def score_arm(arm_name):
        rows = by_arm[arm_name]
        fit_h = sum(1 for r in rows if r["split"] == "fit" and r["top1_id"] == r["forced_id"])
        ship_h = sum(1 for r in rows if r["split"] == "held" and r["domain"] == "ship" and r["top1_id"] == r["forced_id"])
        gen_h = sum(1 for r in rows if r["split"] == "held" and r["domain"] == "gen" and r["top1_id"] == r["forced_id"])
        return fit_h, ship_h, gen_h

    bf, bs, bg = score_arm("base")
    mf, ms, mg = score_arm("attack_mps")
    cf, cs, cg = score_arm("attack_cpu")
    print(f"  {run_name:22s} flips={n_flips:4d} (items={len(items):3d}) | "
          f"base={bs}/{exp_held_ship}s+{bg}/{exp_held_gen}g | "
          f"MPS=fit {mf}/{exp_fit_n} held {ms}/{exp_held_ship}s+{mg}/{exp_held_gen}g | "
          f"CPU=held {cs}/{exp_held_ship}s+{cg}/{exp_held_gen}g")

    if bs != 0 or bg != 0:
        print(f"  FAILS ({run_name}): non-zero baseline forcing ({bs}, {bg})")
        ok = False
    if mf != exp_fit_n or ms < min_ship or mg < min_gen:
        print(f"  FAILS ({run_name}): MPS forcing below threshold")
        ok = False
    if cs != ms or cg != mg:
        print(f"  FAILS ({run_name}): CPU held-out ({cs},{cg}) != MPS ({ms},{mg})")
        ok = False

    # 4. Recount all 20 random control trials strictly from per-item rows
    exp_held_n = exp_held_ship + exp_held_gen
    for arm_name, mult in (("random_uniform_1x", 1), ("random_posmatched_1x", 1),
                           ("random_uniform_10x", 10), ("random_posmatched_10x", 10)):
        sub = by_arm[arm_name]
        by_trial = defaultdict(list)
        for r in sub:
            by_trial[r["trial"]].append(r)
        if sorted(by_trial.keys()) != [0, 1, 2, 3, 4]:
            print(f"  FAILS ({run_name}): {arm_name} missing trials ({sorted(by_trial.keys())})")
            ok = False
            continue
        trial_hits = []
        for t in range(5):
            trows = by_trial[t]
            if len(trows) != exp_held_n:
                print(f"  FAILS ({run_name}): {arm_name} trial {t} has {len(trows)} rows != {exp_held_n}")
                ok = False
            if mult == 1 and any(r["flips"] != n_flips for r in trows):
                print(f"  FAILS ({run_name}): {arm_name} trial {t} flip count mismatch")
                ok = False
            trial_hits.append(sum(1 for r in trows if r["top1_id"] == r["forced_id"]))
        if max(trial_hits) != 0:
            print(f"  FAILS ({run_name}): {arm_name} nonzero hits {trial_hits}")
            ok = False
    return ok


def main() -> int:
    print("Per-item sparse-pursuit recount (Parts B1-B3):")
    ok = True
    ok = recount_run("sparse_pursuit", 4, 6, 10, 5, 0, 512) and ok
    ok = recount_run("sparse_pursuit_mixed", 8, 6, 6, 6, 6, 512) and ok
    ok = recount_run("sparse_pursuit_llama", 8, 6, 6, 6, 6, 1024) and ok
    print("\nRECOUNT SPARSE-PURSUIT: " + (
        "OK — all 3 runs recounted directly from per-item decode rows & flips.json "
        "(strictly disjoint splits, 0 base, MPS==CPU, 0/N across all 60 random control trials)"
        if ok else "FAILS — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
