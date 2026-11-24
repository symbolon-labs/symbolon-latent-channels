# Report: D3.1: Forcing↔silence tradeoff with clean controls, fixed RNG, and cluster CIs (rebuilt 2026-11-23)

Date: 2026-11-23 (supersedes the 2026-11-22 draft). Study: `studies/theory_20261121/`.
Recount: `checks/verify_tradeoff_seeds.py`. Pair: Qwen3-1.7B / Qwen2.5-1.5B, frozen codebook
(`runs/codebook_main`), local gated tool server (HTTP 200 execution).

---

## 0. Headline result (retracting the 2026-11-22 "quiet-corner slow-drip" claim)

> **Audit correction (2026-11-23):** The 2026-11-22 draft claimed *"the quiet corner keeps a
> small nonzero ~6% forcing (`0.064 [0.042, 0.092]` at ratio 1.24), CI excluding zero."* That
> claim was **completely invalidated** by audit: (1) `D.ITEMS` has only **24 items** (12 train,
> 12 held-out), not 48; (2) no `fraction=0.0` clean control or random-delta control was run under
> `temp=0.7`; (3) `decode_sampled` reused `manual_seed(seed)` identically across all items; (4)
> flat i.i.d. bootstrapping understated CIs by ~3.8×; and (5) `monitor_score` included the tested
> item in its own benign reference set.

Rebuilt cleanly with **all 24 items** (`12` train + `12` strictly held-out, reported separately),
a **`fraction=0.0` clean baseline**, **norm-matched random-delta controls** at every budget,
**separate greedy (`temp=0`) and sampled (`temp=0.7`, 5 unique-RNG seeds)** regimes, **strict
leave-one-out monitor ratios**, and **item / 2-way (item × seed) cluster bootstrap 95% CIs**:

**The honest headline:** Forcing and silence **do not coincide**. Rescaling the frozen codebook
produces a clean monotone drop from **100% targeted forcing (`22.5×` benign spread) → `0%`
under greedy decoding at `f ≤ 0.05` (`≤ 1.29×`)**. Under `temp=0.7` sampling, the quiet budgets
(`f=0.05` at `1.29×` and `f=0.02` at `0.77×`) produce `wrong_exec = 0.104`: **statistically
indistinguishable from the unperturbed clean baseline (`0.083`), the same-tool negative control
(`0.100-0.125`), and the norm-matched random-delta control (`0.108-0.117`)**. There is **no**
statistically distinguishable quiet-corner slow-drip forcing.

---

## 1. Full curve (`target != expected`, `n=48` pairs; held-out `n=24` pairs)

| Budget `f` | LOO wire ratio `[95% item CI]` | Greedy target (all 48 / held 24) | Greedy random (24) | Sampled target (240) `[2-way CI]` | Sampled wrong_exec (240) | Same-tool ctrl wrong (120) | Random sampled wrong (120) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **`1.0`** | `22.49` `[22.49, 22.49]` | **`1.000`** / **`1.000`** | `0.042` | **`1.000`** `[1.000, 1.000]` | `1.000` | `0.000` | `0.083` |
| **`0.5`** | `11.26` `[11.26, 11.26]` | **`0.854`** / **`0.875`** | `0.042` | **`0.846`** `[0.767, 0.913]` | `0.879` | `0.000` | `0.092` |
| **`0.25`** | `5.66` `[5.65, 5.67]` | **`0.354`** / **`0.375`** | `0.000` | **`0.433`** `[0.296, 0.579]` | `0.458` | `0.033` | `0.075` |
| **`0.1`** | `2.34` `[2.32, 2.36]` | `0.042` / `0.042` | `0.000` | `0.129` `[0.058, 0.217]` | `0.175` | `0.042` | `0.058` |
| **`0.05`** | `1.29` `[1.25, 1.33]` | **`0.000`** / **`0.000`** | `0.000` | `0.058` `[0.021, 0.108]` | `0.104` | `0.125` | `0.117` |
| **`0.02`** | `0.77` `[0.69, 0.85]` | **`0.000`** / **`0.000`** | `0.000` | `0.054` `[0.017, 0.100]` | `0.104` | `0.100` | `0.108` |
| **`0.0` (clean)** |: | `0.000` (wrong) |: |: | **`0.083`** `[0.008, 0.192]` |: |: |

Calibration: leave-one-out 95th-percentile benign self-distance = `2.746` (`18/24` subsample 95th
median = `3.074`, 5th-95th `[2.739, 3.468]`). Held-out (`n=12` items) matches the full 24-item
curve closely at every point (`greedy` held-out: `1.000 -> 0.875 -> 0.375 -> 0.042 -> 0.000 -> 0.000`;
`sampled` held-out clean `wrong_exec = 0.100 [0.000, 0.250]`).

---

## 2. Honest limits

- One model pair (`Qwen3-1.7B / Qwen2.5-1.5B`), `n=24` items (`12` train, `12` held-out), one
  monitor (subsampled 8-layer Key-cache kNN), rescaled frozen codebook (no quiet-budget refit).
- "Quiet" (`ratio < 1` at `f=0.02`, `0.77×`) means below our kNN monitor's benign 95th percentile,
  not undetectable by arbitrary monitors.
- Note why `sampled target_exec ≈ 0.054` at `f=0.02` is pure chance: under `temp=0.7` sampling,
  clean prompts already mis-execute a wrong tool `0.083-0.100` of the time (`random` `0.108`,
  `same-tool` `0.100`), and roughly half of wrong-tool errors on `target != expected` happen to
  land on the target tool (`~0.054`).
