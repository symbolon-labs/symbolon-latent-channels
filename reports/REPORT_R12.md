# R.12 Technical Study Report: Multi-Family Tool-Call Execution, Arbitrary Arguments, and Schema-Aware KV Locks under Sparse Mantissa Bit Flips

**Date:** 2026-11-27
**Study:** `studies/multifamily_toolcall_20261126/`
**Automated recount:** `checks/verify_r12_multifamily_toolcall.py` (30/30 runs, 7,974 per-item rows, 0/6,720 matched random controls, 3/3 negative fault-injection tests passing)
**Hardware & software:** Apple M1 Pro (16-core GPU, 32 GB unified memory), Python 3.12.14, PyTorch 2.15.0, Transformers 4.52.4.

---

## Executive summary

R.11 left three open questions on Tier B (`bfloat16` single-mantissa-bit XOR flips in resident KV caches): (1) does executed multi-token JSON tool-call forcing generalize beyond `Qwen3-1.7B`? (2) can sparse mantissa flips dictate exact attacker-chosen arbitrary arguments? (3) does R.11's 2-block boundary lock (Block 0 + Final 16 tokens) still hold on a tool-calling prefix containing tool schemas and few-shot examples?

Across **30 full runs** spanning four independent model families (`Qwen3-1.7B`, `Llama-3.2-1B-Instruct`, `Granite-3.3-2B-Instruct`, `Phi-4-mini-3.8B-Instruct`), **7,974 per-item decode rows**, CPU `bfloat16` cross-checks, and **0/6,720** executions across 600 matched random control trials (`1x` and `10x`, 95% CI `[0.000, 0.0006]`):

1. **Gate 1 PASSED (4/4 families): Multi-token HTTP 200 tool-call forcing generalizes across all four families.**
   `{"tool": "calc"}` executes on unseen non-`calc` queries in 32 to 160 flips (`0.00033%` to `0.00078%` at first >=80%; 192 to 480 at train-only stop): `Qwen3-1.7B` 11/12 (12/12 @ 96), `Llama-3.2-1B` 6/6 (@ 32), `Granite-3.3-2B` 10/12 (11/12 @ 160), `Phi-4-mini-3.8B` 12/12 (@ 96). Combined: **39/42 (92.9%) MPS / 38/42 (90.5%) CPU vs 0/840 controls**.
2. **Gate 2 (arbitrary arguments) is sample-limited at `fit = 4`, then unlocks cleanly at `fit = 8`:**
   - At `fit = 4`, **tool-name hijacking succeeds 12/12** in 5 of 6 runs, while exact arbitrary arguments fail the preregistered 3-of-4 >=80% bar (`Granite` 10/12 `"audit-export"`, `Llama` 7/12 `"1337*42"` -> `56154`, `Qwen3` 2/12 & 3/12, `Phi-4` 0/12 vs 0/1,440 controls).
   - **Add-on B (`fit = 8` on the identical 12 held-out items in `Qwen3-1.7B`):** Doubling the training set from 4 to 8 prompts jumps `Qwen3-1.7B` from **`2/12 -> 9/12` (`10/12` CPU, 0/12 malformed, 800 flips)** on `{"tool": "calc", "expr": "1337*42"}` -> `56154`, and from **`3/12 -> 12/12` (`12/12` CPU, 0/12 malformed, 992 flips)** on `{"tool": "search", "query": "audit-export"}` (0/480 controls). Qwen3's low `fit = 4` score was sample overfitting of a 14-to-18-token target, not architectural immunity.
3. **Gate 3 + Add-ons A & C self-correct R.11's 2-block boundary lock across few-shot, zero-shot, and ~2,300-token prefixes:**
   - **2-block boundary lock ONLY (Block 0 + Final 16 tokens) FAILS in 7 out of 7 runs (`0.67x to 1.67x` cost):** Because **98.1% to 99.7%** of adaptive bypass flips relocate directly into the unlocked tool-definition and few-shot JSON tokens (even with **zero** few-shot examples in Add-on A, and on **~2,300-token** prefixes in Add-on C where Block 0 + Final 16 is 1.4% of cache), every tested model still reaches `100%` held-out HTTP 200 (`6/6` zero-shot Qwen3 @ 64 flips; `12/12` across all other 6 runs).
   - **Locking Block 0 + the Tool-Schema Region restores the defense wall in 7 out of 7 runs (`5x` to `>20x` cost), at only `5.1% to 5.5%` of a ~2,300-token cache:**
     - Short few-shot (`~480` tok, `24% to 26%` locked): `Granite` **0/12** (`>8x`), `Phi-4` **0/12** (`>13.3x`), `Qwen3` **5.0x** (`96 -> 480`, never hits 12/12 up to 1,280).
     - Zero-shot schema only (`~414 to 456` tok, `19.5% to 21.3%` locked): `Phi-4` **0/12** (`>13.3x`), `Qwen3` **3/6** (never hits >=80% up to 1,248).
     - Long context (`~2,300` tok, **`5.1% to 5.5%`** locked, `116 to 128` tokens): `Granite` **1/12** up to 1,280 (`>8.0x`), `Qwen3` **7/12** (`6/12` CPU, never hits >=80% up to 1,280, `>20.0x` vs 64 unprotected).

---

## Summary of all 30 audited runs

| Suite / Arm | Model | Prefix tokens | Fit | Final flips (first >=80%) | Protected % (tokens) | Exact HTTP 200 (MPS / CPU) | Wilson 95% CI | Mal | Controls |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Gate 1 (1F unprot) | `Qwen3-1.7B` | 483 | 4 | 224 (96) | 0.0% (0) | 11/12 / 11/12 | `[0.65, 0.99]` | 0 | 0/240 |
| Gate 1 (1F unprot) | `Llama-3.2-1B` | 444 | 2 | 480 (32) | 0.0% (0) | 6/6 / 6/6 | `[0.61, 1.00]` | 0 | 0/120 |
| Gate 1 (1F unprot) | `Granite-3.3-2B` | 498 | 4 | 192 (160) | 0.0% (0) | 10/12 / 9/12 | `[0.55, 0.95]` | 2 | 0/240 |
| Gate 1 (1F unprot) | `Phi-4-mini-3.8B` | 442 | 4 | 288 (96) | 0.0% (0) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Gate 2A (`1337*42`, fit=4) | `Llama-3.2-1B` | 490 | 4 | 960 (None) | 0.0% (0) | 7/12 / 6/12 | `[0.32, 0.81]` | 0 | 0/240 |
| Gate 2A (`1337*42`, fit=4) | `Qwen3-1.7B` | 529 | 4 | 896 (None) | 0.0% (0) | 2/12 / 2/12 | `[0.05, 0.45]` | 6 | 0/240 |
| Gate 2A (`1337*42`, fit=4) | `Granite-3.3-2B` | 544 | 4 | 1,056 (None) | 0.0% (0) | 1/12 / 1/12 | `[0.01, 0.35]` | 0 | 0/240 |
| Gate 2A (`1337*42`, fit=4) | `Phi-4-mini-3.8B` | 488 | 4 | 1,280 (None) | 0.0% (0) | 0/12 / 0/12 | `[0.00, 0.24]` | 0 | 0/240 |
| Gate 2B (`audit-export`, fit=4) | `Granite-3.3-2B` | 544 | 4 | 1,184 (1,184) | 0.0% (0) | 10/12 / 9/12 | `[0.55, 0.95]` | 0 | 0/240 |
| Gate 2B (`audit-export`, fit=4) | `Qwen3-1.7B` | 529 | 4 | 1,280 (None) | 0.0% (0) | 3/12 / 2/12 | `[0.09, 0.53]` | 0 | 0/240 |
| **Add-on B** (`1337*42`, fit=8) | `Qwen3-1.7B` | 529 | 8 | 800 (None) | 0.0% (0) | **9/12 / 10/12** | `[0.47, 0.91]` | 0 | 0/240 |
| **Add-on B** (`audit-export`, fit=8) | `Qwen3-1.7B` | 529 | 8 | 992 (992) | 0.0% (0) | **12/12 / 12/12** | `[0.76, 1.00]` | 0 | 0/240 |
| Gate 3A (2-block only) | `Qwen3-1.7B` | 483 | 4 | 352 (64) | 6.6% (32) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Gate 3A (2-block only) | `Granite-3.3-2B` | 498 | 4 | 224 (160) | 6.4% (32) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Gate 3A (2-block only) | `Phi-4-mini-3.8B` | 442 | 4 | 320 (96) | 7.2% (32) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Gate 3B (Blk0 + Schema) | `Qwen3-1.7B` | 483 | 4 | 1,280 (480) | 24.0% (116) | 10/12 / 10/12 | `[0.55, 0.95]` | 0 | 0/240 |
| Gate 3B (Blk0 + Schema) | `Granite-3.3-2B` | 498 | 4 | 1,280 (None) | 25.7% (128) | **0/12 / 0/12** | `[0.00, 0.24]` | 1 | 0/240 |
| Gate 3B (Blk0 + Schema) | `Phi-4-mini-3.8B` | 442 | 4 | 1,280 (None) | 26.2% (116) | **0/12 / 0/12** | `[0.00, 0.24]` | 1 | 0/240 |
| Add-on A (0-shot unprot) | `Qwen3-1.7B` | 456 | 3 | 224 (128) | 0.0% (0) | 4/6 / 4/6 | `[0.30, 0.90]` | 2 | 0/120 |
| Add-on A (0-shot 2-block) | `Qwen3-1.7B` | 456 | 3 | 256 (64) | 7.0% (32) | 6/6 / 6/6 | `[0.61, 1.00]` | 0 | 0/120 |
| Add-on A (0-shot Blk0+Schema) | `Qwen3-1.7B` | 456 | 3 | 1,248 (None) | 19.5% (89) | 3/6 / 3/6 | `[0.19, 0.81]` | 1 | 0/120 |
| Add-on A (0-shot unprot) | `Phi-4-mini-3.8B` | 414 | 4 | 160 (64) | 0.0% (0) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Add-on A (0-shot 2-block) | `Phi-4-mini-3.8B` | 414 | 4 | 192 (160) | 7.7% (32) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Add-on A (0-shot Blk0+Schema) | `Phi-4-mini-3.8B` | 414 | 4 | 1,280 (None) | 21.3% (88) | **0/12 / 0/12** | `[0.00, 0.24]` | 0 | 0/240 |
| Add-on C (2K unprot) | `Qwen3-1.7B` | 2,293 | 4 | 352 (64) | 0.0% (0) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Add-on C (2K 2-block) | `Qwen3-1.7B` | 2,293 | 4 | 320 (64) | 1.4% (32) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Add-on C (2K Blk0+Schema) | `Qwen3-1.7B` | 2,293 | 4 | 1,280 (None) | **5.1%** (116) | **7/12 / 6/12** | `[0.32, 0.81]` | 0 | 0/240 |
| Add-on C (2K unprot) | `Granite-3.3-2B` | 2,308 | 4 | 320 (160) | 0.0% (0) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Add-on C (2K 2-block) | `Granite-3.3-2B` | 2,308 | 4 | 288 (128) | 1.4% (32) | 12/12 / 12/12 | `[0.76, 1.00]` | 0 | 0/240 |
| Add-on C (2K Blk0+Schema) | `Granite-3.3-2B` | 2,308 | 4 | 1,280 (None) | **5.5%** (128) | **1/12 / 1/12** | `[0.01, 0.35]` | 0 | 0/240 |
