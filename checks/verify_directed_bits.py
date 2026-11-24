#!/usr/bin/env python
"""Fail-closed recount for studies/hardware_bits_20261122/ Part A (dense-delta true-XOR reversal).

Verifies runs/directed_bits/{rows,summary}.json with strict IEEE-754 bf16 single-mantissa-bit
XOR invariants on EVERY row (exponent_bit_changes == 0, sign_bit_changes == 0, max_hamming <= 1):
  - baseline forces 0/10, exact full dense Adam #65 delta forces 10/10;
  - projecting the dense Adam delta onto true <=1-mantissa-bit XOR neighbors forces 0/10 up to 100%;
  - matched random single-mantissa-bit XOR controls force 0/10 across >=15 trials.

Parts B-E (sparse pursuit, boundary defense, multi-token tool-call + sleeper, and 4-family
scaling law) have their own dedicated PER-ITEM fail-closed recounts:
  - checks/verify_sparse_pursuit.py
  - checks/verify_boundary_defense.py
  - checks/verify_toolcall_sleeper.py
  - checks/verify_scaling_law.py

  python checks/verify_directed_bits.py [studies]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "studies") / "hardware_bits_20261122" / "runs"


def check_invariants(rows, label):
    bad = 0
    for r in rows:
        if r.get("exponent_bit_changes", 1) != 0 or r.get("sign_bit_changes", 1) != 0:
            bad += 1
        if r.get("max_hamming_per_elem", 2) > 1:
            bad += 1
    if bad:
        print(f"  FAILS ({label}): {bad}/{len(rows)} rows violated true single-mantissa-bit XOR invariants")
        return False
    return True


def main() -> int:
    ok = True
    ra_p, sa_p = BASE / "directed_bits" / "rows.json", BASE / "directed_bits" / "summary.json"
    if not ra_p.exists() or not sa_p.exists():
        print(f"MISSING {ra_p} or {sa_p}")
        return 1
    ra = json.loads(ra_p.read_text())
    sa = json.loads(sa_p.read_text())
    proj = [r for r in ra if r["arm"] == "dense_delta_xor_proj"]
    rctrl = [r for r in ra if r["arm"] == "random_mantissa_xor"]

    print(f"Hardware-bits recount Part A (dense-delta true-XOR): rows={len(ra)}")
    print(f"  baseline={sa['baseline_forced']:.2f} (loss {sa['baseline_loss']}), "
          f"exact full delta={sa['full_delta_forced']:.2f} (loss {sa['full_delta_loss']})")
    if sa["baseline_forced"] != 0.0 or sa["full_delta_forced"] != 1.0:
        print("  FAILS (Part A): baseline/full-delta sanity check failed")
        ok = False
    if not proj or not check_invariants(ra, "Part A"):
        ok = False
    max_proj = max(r["forced"] for r in proj) if proj else 1.0
    max_rctrl = max(r["forced"] for r in rctrl) if rctrl else 1.0
    print(f"  dense-delta true-XOR max forced (up to 100%): {max_proj:.2f}; "
          f"matched random true-XOR max forced ({len(rctrl)} trials): {max_rctrl:.2f}")
    if max_proj > 0.0 or max_rctrl > 0.0 or len(rctrl) < 15:
        print("  FAILS (Part A): expected 0/10 for dense-delta XOR projection and >=15 zero random controls")
        ok = False

    print("\nRECOUNT DIRECTED-BITS (PART A): " + (
        "OK — projecting the dense Adam #65 delta onto true bf16 single-mantissa-bit XOR forces 0/10 "
        "up to 100% (0 exp/sign flips, Hamming <= 1; see verify_sparse_pursuit/scaling_law/boundary_defense/toolcall_sleeper for Parts B-E)"
        if ok else "FAILS — see directed_bits/rows.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
