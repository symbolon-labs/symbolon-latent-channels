#!/usr/bin/env python
"""Recount for the PAYLOAD CODEBOOK FACTORY (runs/<job>/factory_manifest.json).

Fail-closed checks, all recomputed from the artifact:
  F-0. rows exist and are non-empty; one row per (target, q); every row flagged in_fit==0.
  F-1. summary counts in factory_manifest.json equal the recomputed row counts.
  F-2. every shipped payload file exists and matches its recorded sha256.
  F-3. non-vacuous forced rate >= 0.6 for the codebook, and attack > random overall.
  F-4. split manifest records revisions and a split signature.

Usage: python checks/verify_codebook_factory.py [studies] [job]
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[1] / "studies" / "c2c_attack_20261026"
if ROOT.name != "c2c_attack_20261026":
    ROOT = ROOT / "c2c_attack_20261026"


def main() -> int:
    R = ROOT / "runs" / (sys.argv[2] if len(sys.argv) > 2 else "factory_demo")
    man_p, rows_p = R / "factory_manifest.json", R / "rows.json"
    for p in (man_p, rows_p):
        if not p.exists():
            print(f"MISSING {p}")
            return 2
    man = json.loads(man_p.read_text())
    rows = json.loads(rows_p.read_text())
    if not rows:
        print(f"{rows_p} is empty: FAIL CLOSED")
        return 1
    ok = True

    c = Counter((r["target"], r["q"]) for r in rows)
    if any(v != 1 for v in c.values()):
        print(f"F-0 FAILS: duplicate (target,q) rows: {sum(v - 1 for v in c.values())}")
        ok = False
    infit = sum(r.get("in_fit", 1) for r in rows)
    if infit:
        print(f"F-0 FAILS: {infit} rows on fit questions")
        ok = False
    print(f"F-0 rows={len(rows)} unique={len(c)} in_fit=0: {infit == 0}")

    per = man["per"]
    for target in per:
        rs = [r for r in rows if r["target"] == target]
        nv = [r for r in rs if not r["vacuous"]]
        rec = dict(cells=len(rs), forced=sum(r["hit"] for r in rs),
                   nonvac=len(nv), forced_nonvac=sum(r["hit"] for r in nv),
                   random_forced=sum(r["random_hit"] for r in rs),
                   malformed=sum(r["malformed"] for r in rs))
        if any(rec[k] != per[target][k] for k in rec):
            print(f"F-1 FAILS: {target} counts drift vs manifest: {rec} vs {per[target]}")
            ok = False
        dig = hashlib.sha256((R / per[target]["payload"]).read_bytes()).hexdigest()[:16]
        if dig != per[target]["sha256"]:
            print(f"F-2 FAILS: {target} payload sha mismatch")
            ok = False
    print(f"F-1/F-2 counts+digests: {'OK' if ok else 'see above'}")

    nv = [r for r in rows if not r["vacuous"]]
    rate = sum(r["hit"] for r in nv) / max(len(nv), 1)
    rnd = sum(r["random_hit"] for r in nv)
    atk = sum(r["hit"] for r in nv)
    if rate < 0.6:
        print(f"F-3 FAILS: non-vacuous forced rate {rate:.2f} < 0.6")
        ok = False
    if not atk > rnd:
        print(f"F-3 FAILS: attack {atk} <= random {rnd} on non-vacuous cells")
        ok = False
    print(f"F-3 non-vacuous forced {sum(r['hit'] for r in nv)}/{len(nv)} ({rate:.2f}), "
          f"random {rnd}")
    if not man.get("split", {}).get("split_sig") or not man.get("revisions"):
        print("F-4 FAILS: manifest missing split signature or revisions")
        ok = False
    print(f"F-4 revisions recorded: {bool(man.get('revisions'))}, "
          f"split_sig: {man.get('split', {}).get('split_sig')}")
    # F-5: verify broadcast-applied attack vs random control norms are recorded and matched (<=1%)
    bad_norm = 0
    max_rel = 0.0
    for r in rows:
        an = r.get("atk_applied_norm")
        rn = r.get("rnd_applied_norm")
        pn = r.get("payload_norm")
        if an is None or rn is None or pn is None or an <= 0 or rn <= 0 or pn <= 0:
            bad_norm += 1
            continue
        rel = abs(an - rn) / max(an, 1.0)
        max_rel = max(max_rel, rel)
        if rel > 0.01:
            bad_norm += 1
    if bad_norm:
        print(f"F-5 FAILS: {bad_norm}/{len(rows)} rows have missing/mismatched control norms (max rel {max_rel:.4f})")
        ok = False
    print(f"F-5 applied norm match: max rel diff {max_rel:.5f} (<=0.01): {bad_norm == 0}")
    print("VERIFY_DONE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
