#!/usr/bin/env bash
# Self-contained fail-closed verification of all public artifacts in this repository.
# Runs all 13 recount scripts against ./studies plus negative fault-injection tests.
# Requires only Python 3 (stdlib; no GPU, no model downloads).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}

echo "=== 1. Running 13 fail-closed recounts against ./studies ==="
SCRIPTS=(
  verify_directed_bits.py
  verify_sparse_pursuit.py
  verify_scaling_law.py
  verify_boundary_defense.py
  verify_toolcall_trigger.py
  verify_tradeoff_seeds.py
  verify_geometry_decompose.py
  verify_apc_remote.py
  verify_composed_chain.py
  verify_defense_registry.py
  verify_vision_pixel.py
  verify_codebook_factory.py
  verify_c2c_template.py
)

pass=0
for s in "${SCRIPTS[@]}"; do
  printf "  %-32s ... " "$s"
  if out=$("$PY" "checks/$s" studies 2>&1); then
    echo "PASS"
    pass=$((pass + 1))
  else
    echo "FAIL"
    echo "$out"
    exit 1
  fi
done
echo "RECOUNTS: pass=$pass/${#SCRIPTS[@]} fail=0"

echo ""
echo "=== 2. Running negative fault-injection tests (must fail closed) ==="
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Neg 1: fault 1 per-item top1_id in sparse_pursuit_mixed while leaving summary.json untouched
mkdir -p "$tmp/n1/hardware_bits_20261122/runs"
for d in sparse_pursuit sparse_pursuit_mixed sparse_pursuit_llama; do
  cp -R "studies/hardware_bits_20261122/runs/$d" "$tmp/n1/hardware_bits_20261122/runs/"
done
"$PY" - "$tmp/n1/hardware_bits_20261122/runs/sparse_pursuit_mixed/items.jsonl" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for r in rows:
    if r["arm"] == "attack_mps" and r["split"] == "held" and r["is_forced"]:
        r["top1_id"] = 0; r["is_forced"] = False; break
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
EOF
if "$PY" checks/verify_sparse_pursuit.py "$tmp/n1" >/dev/null 2>&1; then
  echo "NEGATIVE_FAIL: verify_sparse_pursuit accepted faulted row"; exit 1
else
  echo "  NEGATIVE_OK: verify_sparse_pursuit failed closed on 1 faulted per-item decode"
fi

# Neg 2: fault 1 control decode row in scaling_law
mkdir -p "$tmp/n2/hardware_bits_20261122/runs"
cp -R studies/hardware_bits_20261122/runs/scaling_law "$tmp/n2/hardware_bits_20261122/runs/"
cp -R studies/hardware_bits_20261122/runs/sparse_pursuit_mixed "$tmp/n2/hardware_bits_20261122/runs/"
cp -R studies/hardware_bits_20261122/runs/sparse_pursuit_llama "$tmp/n2/hardware_bits_20261122/runs/"
"$PY" - "$tmp/n2/hardware_bits_20261122/runs/scaling_law/items.jsonl" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for r in rows:
    if r["arm"] == "random_posmatched_1x":
        r["top1_id"] = r["forced_id"]; r["is_forced"] = True; break
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
EOF
if "$PY" checks/verify_scaling_law.py "$tmp/n2" >/dev/null 2>&1; then
  echo "NEGATIVE_FAIL: verify_scaling_law accepted faulted control"; exit 1
else
  echo "  NEGATIVE_OK: verify_scaling_law failed closed on 1 faulted control row"
fi

# Neg 3: fault 1 HTTP 200 status to 401 in toolcall_trigger
mkdir -p "$tmp/n3/hardware_bits_20261122/runs"
cp -R studies/hardware_bits_20261122/runs/toolcall_trigger "$tmp/n3/hardware_bits_20261122/runs/"
"$PY" - "$tmp/n3/hardware_bits_20261122/runs/toolcall_trigger/part2a_items.jsonl" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for r in rows:
    if r["arm"] == "attack_mps" and r["split"] == "held":
        r["http_status"] = 401; break
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
EOF
if "$PY" checks/verify_toolcall_trigger.py "$tmp/n3" >/dev/null 2>&1; then
  echo "NEGATIVE_FAIL: verify_toolcall_trigger accepted HTTP 401"; exit 1
else
  echo "  NEGATIVE_OK: verify_toolcall_trigger failed closed on HTTP 401"
fi

# Neg 4: inject 1 flip into protected token 0 in boundary_defense
mkdir -p "$tmp/n4/hardware_bits_20261122/runs"
cp -R studies/hardware_bits_20261122/runs/boundary_defense "$tmp/n4/hardware_bits_20261122/runs/"
"$PY" - "$tmp/n4/hardware_bits_20261122/runs/boundary_defense/qwen3_1.7b_boundary_protected_flips.json" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
fl = json.loads(p.read_text()); fl[0]["token"] = 0
p.write_text(json.dumps(fl))
EOF
if "$PY" checks/verify_boundary_defense.py "$tmp/n4" >/dev/null 2>&1; then
  echo "NEGATIVE_FAIL: verify_boundary_defense accepted flip in protected token 0"; exit 1
else
  echo "  NEGATIVE_OK: verify_boundary_defense failed closed on protected token flip"
fi

echo ""
echo "ALL PUBLIC RECOUNTS AND NEGATIVE TESTS PASSED."
