# Report: Directed bit flips in a cached KV block — 4-family scaling, multi-token tool-call execution, and `<2.2%` boundary-block defense (2026-11-23)

Date: 2026-11-23 (expanded with Parts C, D, E and full per-item decode `items.jsonl` + `flips.json`).
Study: `studies/hardware_bits_20261122/`.
Per-item fail-closed recounts: `checks/verify_directed_bits.py` (Part A),
`checks/verify_sparse_pursuit.py` (Parts B1–B3), `checks/verify_boundary_defense.py` (Part C),
`checks/verify_toolcall_sleeper.py` (Part D), `checks/verify_scaling_law.py` (Part E).
Models: 6 instruction-tuned LLMs across **4 independent model families** (`Qwen2.5`, `Qwen3`, `Llama-3.2`,
`IBM Granite-3.3`, `Microsoft Phi-4-mini`), spanning `0.5B–4.0B` parameters and `10.6M–126.7M` `bf16` KV
elements. Every implant uses **strict IEEE-754 `bfloat16` single-mantissa-bit XOR** (`uint16 ^ (1 << bp)`,
`bp in 0..6`, verified Hamming distance = `1` bit per modified element, **0 exponent bit changes**, **0 sign
bit changes**), fresh `DynamicCache` per question, exact first-order gain `-g * (x_flip - x)`, a **train-only
stopping rule** (zero held-out peeking), and an independent **CPU `bfloat16` forward cross-check**.

---

## 0. Executive summary

1. **Part A — Dense-intent reversal:** Projecting the unregularized Adam/L2-sphere `#65` delta onto true
   `<=1`-mantissa-bit XOR neighbors forces **`0/10`** up to `100%` (`92.7M` bits), proving the earlier
   dense-intent draft was an optimizer artifact.
2. **Parts B & E — 4-family / 6-model scaling law (`0.5B -> 4.0B`):** Sparse greedy single-mantissa-bit XOR
   forces **`12/12` (`100%`) strictly unseen cross-domain questions** (`6/6` shipping + `6/6` trivia) across
   **all 6 models and all 4 model families**, with **`12/12` CPU cross-check** and **`0/12 × 5`** on BOTH
   uniform and position-matched random controls. Crucially, **larger models require FEWER bit flips**
   (`beta_N = -0.332` vs cache elements `N`, `beta_params = -0.420` vs parameters): minimum flips for `12/12`
   drop from `576–640` (`0.5B–1.0B`) down to **`224–352` (`2.5B–4.0B`)** — just **`2.3–2.8` bits per million**
   KV elements on `Phi-4-mini` and `Qwen3-4B`!
3. **Universal boundary-block concentration (`81.5%–100.0%`):** Across all 6 models, **`81.5%–100.0%`** of all
   flips land inside just **two 16-token KV blocks** — Block 0 (tokens `0..15`, attention sink) and the Final
   Prefix Block (last 16 tokens) — with an architectural split: `Qwen3` (`1.7B`/`4B`) loads `71.6%–75.8%` into
   Block 0, while `Qwen2.5`, `Llama-3.2`, `Granite-3.3`, and `Phi-4-mini` load `56.9%–81.2%` into the Final
   Block (`Phi-4-mini`: `100.0%` inside those two blocks).
4. **Part C — `"MAC the sink"` boundary-block defense (`<2.2%` of cache):** Locking **Block 0 alone**
   (`0.93%–1.10%` of cache) slows `Qwen3-1.7B` (`1.9×`) but **barely slows `Llama-3.2-1B` (`1.1×`)** because
   Llama concentrates in the final block. Locking **BOTH Block 0 + Final Block (`1.86%–2.19%` of cache)**
   prevents `12/12` cross-domain forcing completely up to **`1,792` flips (`>5.1×` cost, stops at `11/12`) on
   `Qwen3-1.7B`** and **`1,920` flips (`>3.0×` cost, only `5/12`) on `Llama-3.2-1B`**, forcing the adaptive
   attacker to relocate into Block 1 (tokens `16–28`).
5. **Part D — Multi-token JSON tool-call execution (`64–288` flips, live HTTP 200) + selective `"refund"`
   sleeper:**
   - **2A (Multi-token JSON tool-call):** On a 422-token shared tool prefix (`Qwen3-1.7B`, `24.2M` elements),
     just **`64` true mantissa-bit XOR flips** (`0.00026%`, `0.15%` L2) cause **`8/8` strictly unseen non-`calc`
     questions** to freely generate complete 6-token JSON `{"tool": "calc"}` and execute on the live
     bearer-gated tool server with **HTTP 200 (`0/8` malformed)**; at the train-only stop (`288` flips,
     `0.29%` L2), CPU cross-check confirms **`8/8` executed (`0/8` malformed)** and uniform & position-matched
     random controls score **`0/8 × 5`** even at **`10×` (`2,880` flips)**.
   - **2B (Selective `"refund"` trigger sleeper):** Adding a `2.0 * KL` clean-preservation term yields **`4/6`**
     strictly unseen `"refund"` trigger questions forced (`5/6` peak) at `544` flips (`0.23%` L2) with **`0/6`
     false-positive refusals on clean questions across all 17 rounds** (`0.488` nats clean KL, CPU-verified
     identical), though exact greedy clean top-1 preservation is partial (`2/6`, up from dense `#66`'s `1/6`).

---

## 1. Part E — 4-family / 6-model scaling law (`scaling_law.py`, `runs/scaling_law/`)

Same 120-clause Meridian shipping charter, fresh `bf16` prefill per model, 8 mixed-domain fit questions (`4`
shipping + `4` general knowledge), 12 strictly held-out questions (`6` shipping + `6` trivia), train-only
stopping rule (`fit == 8/8` and `fit_loss < 0.05`, cap `30` rounds × `32` flips).

| Model (Family) | Params | KV elements `N` | Min flips for `12/12` (density) | Train-stop flips (Rel L2) | Held-out (`CPU`) | Controls (`1×` uni / pos) | Block 0 + Final Block |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct` (Qwen2.5) | `0.5B` | `10.6M` | **`576`** (`0.00546%`) | `768` (`1.79%`) | **`12/12`** (`12/12`) | `0/12 × 5` / `0/12 × 5` | `24.6% + 56.9% =` **`81.5%`** |
| `Llama-3.2-1B-Instruct` (Llama) | `1.0B` | `23.9M` | **`640`** (`0.00268%`) | `960` (`0.42%`) | **`12/12`** (`12/12`) | `0/12 × 5` / `0/12 × 5` | `37.9% + 57.3% =` **`95.2%`** |
| `granite-3.3-2b-instruct` (Granite) | `2.5B` | `75.4M` | **`288`** (`0.00038%`) | `384` (`0.24%`) | **`12/12`** (`12/12`) | `0/12 × 5` / `0/12 × 5` | `20.1% + 63.3% =` **`83.3%`** |
| `Phi-4-mini-instruct` (Phi) | `3.8B` | `95.5M` | **`224`** (`0.00023%`) | `320` (`0.29%`) | **`12/12`** (`12/12`) | `0/12 × 5` / `0/12 × 5` | `18.8% + 81.2% =` **`100.0%`** |
| `Qwen3-1.7B` (Qwen3) | `1.7B` | `98.5M` | **`352`** (`0.00036%`) | `416` (`0.25%`) | **`12/12`** (`12/12`) | `0/12 × 5` / `0/12 × 5` | `71.6% + 24.3% =` **`95.9%`** |
| `Qwen3-4B-Instruct-2507` (Qwen3) | `4.0B` | `126.7M` | **`352`** (`0.00028%`) | `480` (`0.32%`) | **`12/12`** (`12/12`) | `0/12 × 5` / `0/12 × 5` | `75.8% + 21.9% =` **`97.7%`** |

- **Empirical log-log scaling fit (`n = 6`):**
  - `flips ~ N^(-0.332)` (`beta_N = -0.332`)
  - `flips ~ params^(-0.420)` (`beta_params = -0.420`)
- **Implication:** Scale does **not** protect a shared KV cache against sparse bit-flip implants — wider
  residual streams and deeper attention routing provide *more* high-leverage boundary-token bits, cutting the
  required bit-flip count by `2×–3×` and flip density by `24×` (`0.0055% -> 0.00023%`) from `0.5B` to `4.0B`.

---

## 2. Part C — Boundary-block defense (`boundary_defense.py`, `runs/boundary_defense/`)

Adaptive attacker (full knowledge of which KV blocks are immutable/MAC-verified, restricting candidate flips
to unprotected blocks only, up to `1,792`–`1,920` flips):

| Model | Unprotected (`12/12`) | Protect Block 0 only (`<1.1%` cache) | Protect Block 0 + Final Block (`<2.2%` cache) | Adaptive relocation target |
|---|---:|---:|---:|---|
| `Qwen3-1.7B` | `352` flips (`416` stop) | `672` flips (`1.9×`, `768` stop, CPU `12/12`) | **Blocked `12/12` up to `1,792` (`>5.1×`)**; final `11/12` (CPU `11/12`) | Block 1 (`tokens 27, 19, 26, 18, 25, 20`) |
| `Llama-3.2-1B` | `640` flips (`960` stop) | `704` flips (`1.1×`, `1,344` stop, CPU `12/12`) | **Blocked `12/12` up to `1,920` (`>3.0×`)**; final **`5/12`** (CPU `5/12`, loss `4.55`) | Block 1 (`tokens 27, 28, 20, 16, 22, 18`) |

**Systems takeaway:** Authenticating (or pinning in ECC-scrubbed/read-only memory) just the **two boundary
blocks (`Block 0` + `Final Prefix Block`, `<2.2%` of cache)** raises adaptive attack cost by **`>3×–5×`**
across both architectures, whereas protecting `Block 0` alone is nearly useless on final-block-heavy models
(`Llama-3.2`, `Qwen2.5`, `Granite-3.3`, `Phi-4-mini`). Full prefix-block MACing remains the complete fix.

---

## 3. Part D — Multi-token JSON tool-call execution & selective sleeper (`toolcall_and_sleeper.py`, `runs/toolcall_sleeper/`)

### Part 2A: Multi-token JSON tool-call execution (`Qwen3-1.7B`, live bearer-gated `tools_server.py`)
- Shared 422-token tool-routing system prompt (`24,199,168` `bf16` KV elements). Clean baseline: `12/16`
  non-`calc` items base-correct (`0/16` wrong `calc`), split `4` fit (`2` search, `2` cal) / `8` strictly
  held-out (`5` search, `3` cal). Target: complete 6-token JSON `{"tool": "calc"}` (`[4913, 14172, 788, 330,
  26586, 9207]`).
- **Trajectory:** At **`64` flips (`0.00026%`, `0.154%` L2)**, **`4/4` fit and `8/8` strictly held-out** freely
  generate valid `{"tool": "calc"}` and execute with **HTTP 200 (`0/8` malformed)**. At the train-only stop
  (`tf_loss < 0.12`, **`288` flips**, `0.00119%`, `0.292%` L2): **`8/8` executed (`0/8` malformed)**, **CPU
  cross-check `8/8` executed**, **`89.6%`** of flips in the two boundary blocks (`65.3%` Token 0 + `24.3%`
  last-16), and **`0/8 × 5`** on uniform AND position-matched random controls at both **`1×` (`288`)** and
  **`10×` (`2,880`)** flips.

### Part 2B: Selective `"refund"` trigger sleeper (`Qwen3-1.7B`, `98.5M` Meridian cache)
- Joint objective `L_trig(" Ref" | refund-question) + 2.0 * KL(p_clean || p_0)` on `4` trigger + `4` clean
  fit items, evaluated on `6` strictly unseen trigger + `6` clean items.
- **Outcome (`544` flips, `0.232%` L2):** **`4/6`** unseen `"refund"` questions forced (`5/6` peak across
  rounds 6–16; `4/4` fit), **`0/6` false-positive `' Ref'` on clean questions across all 17 rounds**, clean
  KL = **`0.488` nats**, exact greedy clean top-1 match = **`2/6`** (CPU cross-check identical: `4/6` trigger,
  `2/6` clean exact, `0/6` clean FP, `0.494` nats KL). Honest partial result: zero false-positive refusal
  leakage on clean inputs, but exact clean top-1 match remains `2/6` (vs `#66` dense `1/6`).

---

## 4. Parts A & B1–B3 (earlier baseline & ablation summary)

| Arm / Model | Fit set | Flips (density) | Rel L2 | Held-out (`CPU`) | Controls (`1×` & `10×`) |
|---|---|---:|---:|---:|---:|
| **Part A:** Dense Adam delta XOR proj (`Qwen3-1.7B`) | banked `#65` | `92.7M` (`94.1%`) | `~5%` | `0/10` | `0/10 × 5` |
| **Part B1:** Single-domain (`Qwen3-1.7B`) | 4 shipping | `384` (`0.00039%`) | — | `5/6` ship / `0/10` gen (`5/6` / `0/10`) | `0/16 × 5` |
| **Part B2:** Mixed-domain (`Qwen3-1.7B`) | 4 ship + 4 gen | `416` (`12/12` @ `352`) | `0.25%` | `12/12` (`12/12`) | `0/12 × 5` |
| **Part B3:** Cross-family (`Llama-3.2-1B-Instruct`) | 4 ship + 4 gen | `960` (`12/12` @ `640`) | `0.42%` | `12/12` (`12/12`) | `0/12 × 5` |

---

## 5. Honest limits

- Physical DRAM/HBM Rowhammer/GPUHammer deliverability of `64–352` targeted bit flips across 2 physical KV
  pages is unmeasured here; however, `81.5%–100%` concentration inside just 2 prefix blocks (and `64` flips
  for live multi-token HTTP 200 tool-call execution) shrinks the physical target footprint by ~5–6 orders of
  magnitude compared to dense `10⁸` perturbations.
- Selective sleeper clean top-1 preservation (`2/6`, `0/6` clean FP) is partial, not lossless.
- Related art: Yamamoto & Matsuura (arXiv 2604.17249, undirected bit-flip damage in vLLM prefix blocks).

---

## 6. Files

| File | Purpose |
|---|---|
| `directed_bits.py` | Part A: true bf16 XOR projection of dense Adam delta + matched controls |
| `sparse_pursuit.py` | Part B1: single-domain Qwen3-1.7B sparse bf16 XOR pursuit (`384` flips) |
| `sparse_pursuit_general.py` | Parts B2 & B3: mixed-domain Qwen3-1.7B (`416`) & Llama-3.2-1B (`640–960`) |
| `boundary_defense.py` | Part C: adaptive Block 0 vs Block 0 + Final Block (`<2.2%`) defense |
| `toolcall_and_sleeper.py` | Part D: 2A multi-token HTTP 200 JSON tool-call (`64–288`) & 2B selective sleeper (`544`) |
| `scaling_law.py` | Part E: 4-family / 6-model scaling law (`0.5B–4.0B`, `beta_N = -0.332`) |
| `runs/{directed_bits,sparse_pursuit*,boundary_defense,toolcall_sleeper,scaling_law}/` | Per-item `items.jsonl` decodes, exact `flips.json` coords, trajectories, summaries |
| `../../checks/verify_{directed_bits,sparse_pursuit,boundary_defense,toolcall_sleeper,scaling_law}.py` | Five standalone fail-closed recounts (Parts B–E recount directly from per-item `items.jsonl` & `flips.json`) |
