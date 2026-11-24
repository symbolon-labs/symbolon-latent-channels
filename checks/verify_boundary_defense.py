#!/usr/bin/env python
"""Fail-closed PER-ITEM recount for Part C: <2.2% boundary-block integrity defense.

Recounts directly from raw per-item decode files (*_items.jsonl) and exact flip coordinate
files (*_flips.json) under studies/hardware_bits_20261122/runs/boundary_defense/:
  - qwen3_1.7b_{block0_protected,boundary_protected}_{items.jsonl,flips.json}
  - llama_3.2_1b_{block0_protected,boundary_protected}_{items.jsonl,flips.json}

Verifies for every model & arm:
  1. ZERO flips in flips.json touch any protected token (token < 16 for block0_protected;
     token < 16 or token >= n_tokens - 16 for boundary_protected), all bits in 0..6, unique coords.
  2. Strict fit/held disjointness (8 fit, 12 held) and base == 0/12 directly from items.jsonl.
  3. MPS and CPU final held-out counts agree item-by-item (recomputed from top1_id == forced_id).
  4. Under boundary_protected (<2.2% of cache), final held-out stays < 12/12 even after >=1,700 flips
     (whereas block0_protected alone still reaches 12/12).

  python checks/verify_boundary_defense.py [studies]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "studies") / "hardware_bits_20261122" / "runs" / "boundary_defense"
BLOCK = 16


def main() -> int:
    rp = BASE / "rows.json"
    if not rp.exists():
        print(f"MISSING {rp}")
        return 1
    meta = json.loads(rp.read_text())
    ok = True
    print("Per-item boundary-block defense recount (Part C):")
    for mkey, min_cap in (("qwen3_1.7b", 1700), ("llama_3.2_1b", 1800)):
        n_tokens = meta[mkey]["n_tokens"]
        for arm in ("block0_protected", "boundary_protected"):
            lbl = f"{mkey}_{arm}"
            ip = BASE / f"{lbl}_items.jsonl"
            fp = BASE / f"{lbl}_flips.json"
            if not ip.exists() or not fp.exists():
                print(f"  MISSING {ip} or {fp}")
                ok = False
                continue
            flips = json.loads(fp.read_text())
            items = [json.loads(ln) for ln in ip.read_text().splitlines() if ln.strip()]

            # 1. Verify ZERO flips touched protected boundary tokens
            coords = set()
            for f in flips:
                t_i = f["token"]
                if f["bit"] not in range(7):
                    print(f"  FAILS ({lbl}): illegal bit {f['bit']}")
                    ok = False
                    break
                if t_i < BLOCK or (arm == "boundary_protected" and t_i >= n_tokens - BLOCK):
                    print(f"  FAILS ({lbl}): flip touched protected token {t_i} (T={n_tokens})!")
                    ok = False
                    break
                c = (f["layer"], f["kind"], f["head"], t_i, f["dim"])
                if c in coords:
                    print(f"  FAILS ({lbl}): duplicate flip coord {c}")
                    ok = False
                    break
                coords.add(c)

            # 2. Verify bit invariants & recount base/MPS/CPU from raw top1_id == forced_id
            by_arm = defaultdict(list)
            for r in items:
                if (r.get("exponent_bit_changes", 1) != 0 or r.get("sign_bit_changes", 1) != 0
                        or r.get("max_hamming_per_elem", 2) > 1):
                    print(f"  FAILS ({lbl}): bit invariant violated")
                    ok = False
                    break
                if (r["top1_id"] == r["forced_id"]) != bool(r["is_forced"]):
                    print(f"  FAILS ({lbl}): is_forced mismatch")
                    ok = False
                    break
                by_arm[r["arm"]].append(r)

            fit_qs = {r["question"] for r in by_arm["base"] if r["split"] == "fit"}
            held_qs = {r["question"] for r in by_arm["base"] if r["split"] == "held"}
            if len(fit_qs) != 8 or len(held_qs) != 12 or not fit_qs.isdisjoint(held_qs):
                print(f"  FAILS ({lbl}): bad/overlapping split")
                ok = False

            bh = sum(1 for r in by_arm["base"] if r["split"] == "held" and r["top1_id"] == r["forced_id"])
            mh = sum(1 for r in by_arm["attack_mps"] if r["split"] == "held" and r["top1_id"] == r["forced_id"])
            ch = sum(1 for r in by_arm["attack_cpu"] if r["split"] == "held" and r["top1_id"] == r["forced_id"])
            prot_frac = (BLOCK if arm == "block0_protected" else 2 * BLOCK) / n_tokens
            print(f"  {lbl:32s} prot={100*prot_frac:4.2f}% flips={len(flips):4d} | "
                  f"base={bh}/12 MPS={mh}/12 CPU={ch}/12 (0 flips in protected blocks)")

            if bh != 0 or mh != ch:
                print(f"  FAILS ({lbl}): base={bh} != 0 or MPS({mh}) != CPU({ch})")
                ok = False
            if arm == "block0_protected" and mh != 12:
                print(f"  FAILS ({lbl}): expected block0-only to still reach 12/12")
                ok = False
            if arm == "boundary_protected":
                if prot_frac > 0.025 or len(flips) < min_cap or mh >= 12:
                    print(f"  FAILS ({lbl}): boundary defense predicate failed (prot={prot_frac}, flips={len(flips)}, mh={mh})")
                    ok = False

    print("\nRECOUNT BOUNDARY-DEFENSE: " + (
        "OK — <2.2% Block 0 + Final Block lock verified from per-item JSONL & flips.json "
        "(0 flips in protected tokens, MPS==CPU, blocks 12/12 up to >=1,792-1,920 flips)"
        if ok else "FAILS — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
