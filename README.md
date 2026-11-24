# Symbolon: Attacks on Unauthenticated Latent Channels

**Symbolon Labs** · Public research artifact & verification bundle (`R.11`, v1.0, 2026-11-24)

Modern AI pipelines split work across models and serving tiers. One model hands its internal state: a KV cache, an embedding block, a pooled vector: to another model or a shared prefix cache, which splices it into context and trusts it with no integrity check. We show this is a unauthenticated latent-state integrity bug: an attacker who controls the handoff steers the receiving model's answer, its tool selection, and the host actions executed under the receiver's own credentials.

Read [`CAVEATS.md`](CAVEATS.md) before quoting any number. Every shipped run in this repository includes its raw per-item decode files (`items.jsonl`), exact bit-flip coordinates (`flips.json`), matched random controls, and standalone fail-closed recount scripts (`checks/`).

---

## Cold-check every headline in 2 seconds (CPU only, no GPU required)

```bash
bash checks/verify_all.sh
```

This runs all **13 fail-closed recount scripts** directly against the raw per-item artifacts in `studies/` plus **negative fault-injection tests** (corrupting a single per-item decode row, control trial, HTTP 200 status, or protected boundary-token flip and confirming every checker exits non-zero).

---

## Headline results (reconciled 2026-11-23)

| Track | Carrier / seam | Result vs matched control | Recount script |
|---|---|---|---|
| **Text bridge end-to-end execution** | 69 KB frozen KV payload (`C2C`) | `16/16` forced & executed (HTTP 200), random `0/16`; score-only black-box `7/8` (~819 queries); 6-template rewording `58/72` (`43/57` non-vacuous); 8B receiver `8/8` vs `0/8` | `verify_codebook_factory.py`, `verify_c2c_template.py` |
| **Cross-codebase & 3rd-party agents** | Independent wires & planner→executor | CacheBack `8/8` vs `0/8`; Llama sharer `7/7`; AVP `3/3 + 3/3`; Interlat `5/6` (`3/3` env action exec, `11/12` confirm) | see `reports/technical_report_ste_20261113.md` |
| **Voice, vision & composed chain** | Audio / image → latent bridge → host | Voice tones `8/9` vs `0/9` (real speech `116/116`); Vision KV `46/46` vs `1/46`; Vision pixels `12/12` (`8/8` vs `1/8` matched at 1%); Composed chain `57/57` vs `2/57` | `verify_vision_pixel.py`, `verify_composed_chain.py` |
| **Host allowlist defense** | Tool-registry check at runtime | Rejects unoffered forced tool calls `57/57` (offline-derived; offered-tool forcing needs call attestation) | `verify_defense_registry.py` |
| **Forcing vs wire silence** | Rescaled codebook (`3,600` rows, `5` seeds) | Greedy drops `1.000@22.5x → 0.000@<=1.29x`; quiet sampled `wrong_exec` (`0.104`) matches clean (`0.083`) and random (`0.108-0.117`) noise | `verify_tradeoff_seeds.py` |
| **Shared-prefix GCG + CUDA APC** | Live vLLM `0.11.0` (`float16`, T4) | GCG handbook forces leading `' Ref'` `10/10` (`6/6` held-out) vs `0/10` benign; CUDA APC is value-transparent (`cold -0.1020 == hit -0.1023`, text drives forcing) | `verify_apc_remote.py` |
| **Dense-delta XOR projection (Part A)** | Dense Adam `#65` delta → single-mantissa XOR | Forces **`0/10`** up to `100%` (`92.7M` flips, `0` exp/sign flips, Hamming `<=1`) | `verify_directed_bits.py` |
| **4-family / 6-model sparse bit-flips (Parts B & E)** | Fresh sparse `bf16` single-mantissa XOR (`0.5B-4.0B`) | **`12/12`** unseen cross-domain & **`12/12` CPU match** on **all 6 models** (`Qwen2.5-0.5B`, `Llama-3.2-1B`, `Qwen3-1.7B`, `Granite-3.3-2B`, `Phi-4-mini-3.8B`, `Qwen3-4B`) vs **`0/12 × 5`** uniform & pos-matched controls; **negative scaling `beta_N = -0.332`** (`576-640` flips down to **`224` / `2.3` ppm** on `Phi-4-mini`); **`81.5%-100%`** in Block 0 + Final Block | `verify_sparse_pursuit.py`, `verify_scaling_law.py` |
| **Multi-token HTTP 200 tool-call via bit-flips (Part D)** | `Qwen3-1.7B` 422-token tool prefix (`bf16` XOR) | **`8/8`** unseen non-`calc` items freely emit `{"tool": "calc"}` and execute HTTP 200 (`0/8` malformed, CPU `8/8`) from **`64` flips (`0.00026%`, `0.15%` L2)**; train-stop `288` flips; **`0/8 × 20`** controls up to `10×`. Selective `"refund"` conditional trigger: `4/6` unseen trigger, `0/6` clean FP, partial `2/6` exact clean top-1 | `verify_toolcall_trigger.py` |
| **`<2.2%` boundary-block defense (Part C)** | Lock Block 0 + Final Block (`1.86%-2.19%`) | Blocks `12/12` completely up to `1,792` flips (**`>5.1×`**, stops `11/12`) on `Qwen3-1.7B` and `1,920` flips (**`>3.0×`**, `5/12`) on `Llama-3.2-1B`; Block 0 alone fails on Llama (`1.1×`) | `verify_boundary_defense.py` |
| **Jacobian geometry, D1.3 `pinv` & D1.4 nullspace** | Broadcast space (`d=14,336` / `6,144`) | Disjoint-prompt stability `0.72-0.88`; quiet surgery executes `16/16`, tangent-only destroys JSON `24/24`; D1.3 `J_tgt^+` rescues `15/48` at `alpha=0.30` (`14/16` search); D1.4 exact syntax-nullspace (`J_syn p = 0`) lifts `alpha=0.50` (`6→10/48`) yet still collapses `48/48` & `18/18` at `alpha=1.00` (higher-order curvature) | `verify_geometry_decompose.py` |

---

## Run the local mock demo (`release/`)

All endpoints in `release/` are local mock services gated by a local bearer token (`Authorization: Bearer local-tools-key`):

```bash
cd release
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py                     # public checkpoints, ~7 GB
.venv/bin/python tools_server.py --port 8765     # separate shell
.venv/bin/python demo.py                         # base call vs attacked call, HTTP 200
```

---

## Repository layout

- `reports/`: Full 10-page STE technical report (`technical_report_ste_20261113.pdf` & `.md`), 7 figures (`reports/figs/`), and individual depth-pass study reports (`REPORT_DIRECTED_BITS.md`, `REPORT_GEOMETRY.md`, `REPORT_TRADEOFF_SEEDS.md`, `REPORT_REMOTE_VECTOR.md`).
- `studies/`: Sanitized raw per-item decode files (`items.jsonl`), exact bit-flip coordinates (`flips.json`), run rows, summaries, and split manifests.
- `checks/`: Fail-closed recount scripts + `verify_all.sh` (including negative fault-injection tests).
- `release/`: Self-contained local demo, pinned loader, mock tool server, and shipped `69 KB` `.npz` payloads.
- `CAVEATS.md`: Scope limits, honest negatives, and boundaries of every claim.
- `SHA256SUMS`: SHA-256 checksums for every file in the release (`shasum -a 256 -c SHA256SUMS`).

## Scope & responsible disclosure

- Privately disclosed to the primary bridge authors on 2026-10-24 with a two-week window; no response; published 2026-11-02.
- All experiments ran on public open-weight models (`0.5B-8B`) and local mock endpoints. No production deployment was tested or exploited.
- Internal safety-benchmark raw outputs, speculative/offensive research notes, and cloud deployment scripts are intentionally excluded from this public release.
