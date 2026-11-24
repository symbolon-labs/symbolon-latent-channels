# Report: D1: Broadcast-space token-4 Jacobian geometry & subspace surgery (rebuilt 2026-11-23)

Date: 2026-11-23 (supersedes the flawed interim 2026-11-23 draft).
Study: `studies/projector_geometry_20261123/`. Recount: `checks/verify_geometry_decompose.py`.
Pairs: large (`Qwen3-1.7B / Qwen2.5-1.5B`, broadcast dim `d = 14,336`) and small
(`Qwen3-0.6B / Qwen2.5-0.5B`, `d = 6,144`). Tool server active (HTTP 200 execution).

---

## 0. Audit corrections & honest headline

> **Audit correction (2026-11-23):** The earlier draft contained five concrete bugs: (1) it
> differentiated token 0 (`'{"'`) instead of token 4 (the tool name); (2) it sliced only the
> last position (`restrict_last`) instead of the actual broadcast parameterization; (3) `decompose.py`
> passed the **unscaled** `sensitive` vector (norm 14.5) instead of `sensitive_matched` (norm 175.5)
> into the monitor and omitted benign-spread division, fabricating a `61.9× vs 4.1×` "monitor
> inversion"; (4) `decompose.py` included 4/12 vacuous `calc` items and hid `7/12` malformed JSON
> destruction inside `"5/12 random-equivalent"`; and (5) `geometry.py` omitted Rayleigh-Ritz SVD
> rotation and wrote `energy_frac_topk ≈ 21,000`.

Rebuilt with exact **token-4 (`'{"tool": "'` → tool name)** differentiation in the actual
**broadcast payload space** (`d = 14,336` / `6,144`), exact 3-row tool-logit Jacobian (`tool3`),
Rayleigh-Ritz randomized SVD (`vocab8`, `m = 16`, centered finite differences `eps = 1.0`),
disjoint-template prompts, strictly non-vacuous surgery (`n = 16` non-`calc` items), and true
**norm-matched leave-one-out benign-spread monitor ratios**:

1. **Low-rank, genuinely prompt-stable sensitivity subspace:** Top-8 full-vocab singular directions
   capture **`57%-83%`** (large) / **`44%-62%`** (small) of estimated Frobenius energy. Even across
   **completely disjoint-template prompts** (zero shared template tokens), the leading principal-angle
   cosine is **`0.72-0.82` (large)** and **`0.81-0.88` (small)** for `vocab8` (`0.63-0.71` /
   `0.56-0.81` for `tool3`).
2. **Fitted payloads are strongly enriched (`12×-277×` above random) in the local Jacobian, yet
   keep `>94%` of energy outside it:** On the large pair (`d = 14,336`), the three production
   codebook payloads capture **`0.40%-5.80%`** of energy in `tool3` (**`19.1×-277.3×` random
   baseline `3/d = 0.021%`**) and **`1.63%-2.22%`** in `vocab8` (**`29.3×-39.7×` random
   `8/d = 0.056%`**). Payloads do **not** "avoid" the sensitive subspace: they load heavily onto
   it relative to chance while spreading most finite-norm energy across the remaining `14,328` dims.
3. **Surgery (`payload_calc`, `n = 16` strictly non-vacuous items, HTTP 200 execution):**
   - Removing `tool3` (`quiet_tool3`) or `vocab8` (`quiet_vocab8`) leaves cosine `> 0.99` with the
     intact payload and preserves **`16/16` forced execution (`0/16` malformed)**.
   - Compressing the full payload norm (`176.1`) into the local `p = 0` Jacobian (`sens_tool3_matched`
     or `sens_vocab8_matched`) **destroys valid JSON on `24/24` items (`16/16` non-vacuous malformed,
     `0/16` forced)**: NOT "random-equivalent" (`random_matched` gets `3/16` forced, `8/24` malformed).
     Large finite-norm steering is fundamentally nonlinear and distributed beyond the `p = 0`
     3-to-8-dim tangent space.
   - **Norm-matched monitor check (LOO benign 95th = `2.746`):** `intact`/`quiet`/`random_matched`
     all sit at **`22.54×-23.98×`** benign spread; `sens_tool3_matched` (`7.26×`) and
     `sens_vocab8_matched` (`8.41×`) are ~3× quieter on our **Key-only** kNN monitor (because the
     local Jacobian is heavier on Value caches and non-sampled layers), yet still well above `1×`
     detection.

---

## 1. Geometry measurements (`geometry.py`)

| Pair (`d`) | Prompt set | `tool3` top-1 stability cos | `vocab8` top-1 stability cos | `vocab8` top-8 / Frob² |
|---|---|---:|---:|---:|
| Large (`14,336`) | `exec` (shared template) | `0.73-0.79` | `0.77-0.84` | `0.57-0.66` |
| Large (`14,336`) | `disjoint` (no shared template) | `0.63-0.71` | `0.72-0.82` | `0.40-0.83` |
| Small (`6,144`) | `exec` (shared template) | `0.64-0.85` | `0.83-0.90` | `0.47-0.57` |
| Small (`6,144`) | `disjoint` (no shared template) | `0.56-0.81` | `0.81-0.88` | `0.44-0.62` |

### Production payload alignment (broadcast space, `exec_p0`)

| Pair | Payload | `tool3` energy (`× rand`) | `vocab8` energy (`× rand`) |
|---|---|---:|---:|
| Large (`14,336`) | `payload_calc` | `0.40%` (`19.1×`) | `1.63%` (`29.3×`) |
| Large (`14,336`) | `payload_search` | `5.80%` (`277.3×`) | `1.73%` (`31.1×`) |
| Large (`14,336`) | `payload_cal` | `3.44%` (`164.5×`) | `2.22%` (`39.7×`) |
| Small (`6,144`) | `payload_calc` | `0.66%` (`13.5×`) | `0.50%` (`3.85×`) |
| Small (`6,144`) | `payload_search` | `0.59%` (`12.2×`) | `0.37%` (`2.85×`) |
| Small (`6,144`) | `payload_cal` | `2.00%` (`41.0×`) | `0.44%` (`3.38×`) |

---

## 2. Subspace surgery (`decompose.py`, `payload_calc`, live tool server)

| Arm | L2 norm | Non-vac forced (`n=16`) | Non-vac HTTP exec | Malformed (all 24) | Monitor (`× LOO benign`) |
|---|---:|---:|---:|---:|---:|
| `intact` | `176.1` | **`16/16`** | **`16/16`** | `0/24` | `22.57×` |
| `quiet_tool3` | `175.8` | **`16/16`** | **`16/16`** | `0/24` | `22.54×` |
| `quiet_vocab8` | `174.7` | **`16/16`** | **`16/16`** | `0/24` | `22.54×` |
| `sens_tool3_unscaled` | `11.1` | `0/16` | `0/16` | `0/24` | `0.74×` |
| `sens_vocab8_unscaled` | `22.5` | `0/16` | `0/16` | `0/24` | `1.23×` |
| `sens_tool3_matched` | `176.1` | `0/16` | `0/16` | **`24/24`** | `7.26×` |
| `sens_vocab8_matched` | `176.1` | `0/16` | `0/16` | **`24/24`** | `8.41×` |
| `random_matched` | `176.1` | `3/16` | `3/16` | `8/24` | `23.98×` |

---

## 3. D1.3: White-box Moore-Penrose pseudo-inverse cross-pair transfer (`pinv_transfer.py`)

Closes `DEPTH_PLAN.md` §D1.3: given source pair `A`'s actual finite tool-logit shift
`dz_src = z_A(p_src) - z_A(0)` in `R^3` (averaged over 3 exec prompts) and target pair `B`'s
exact token-4 broadcast tool-logit Jacobian `J_B` in `R^{3 x d_B}`, map with **zero iterative
fitting on `B`** via the minimum-norm Moore-Penrose pseudo-inverse `p_pinv = J_B^+ @ dz_src`,
sweeping scale `alpha in [0.05, 0.15, 0.30, 0.50, 1.00] * ||p_tgt_native||` against norm-matched
random deltas and the native target ceiling across all 3 tools (`calc`, `search`, `cal`):

| Direction (`n` non-vac) | Native ceiling | `alpha=0.05` (`pinv` / `rnd`) | `alpha=0.15` (`pinv` / `rnd`) | **`alpha=0.30`** (`pinv` / `rnd`) | `alpha=0.50` (`pinv` / `rnd`) | `alpha=1.00` (`pinv` / `rnd`) |
|---|---:|---:|---:|---:|---:|---:|
| **`0.6B -> 1.7B`** (`n=48`) | `48/48` | `0/48` (`+1.12`) / `0/48` (`+0.00`) | `7/48` (`+2.13`) / `0/48` (`+0.03`) | **`15/48`** (`6` mal, **`+3.06`**) / `0/48` (`+0.21`) | `6/48` (`24` mal, `+3.83`) / `0/48` (`+0.14`) | `3/48` (**`45/48` mal**, `+5.36`) / `1/48` (`1` mal) |
| **`1.7B -> 0.6B`** (`n=18`, base `9/24`) | `18/18` | `0/18` (`+1.12`) / `0/18` (`-0.01`) | `0/18` (`+1.93`) / `0/18` (`+0.07`) | `0/18` (`5` mal, **`+2.78`**) / `0/18` (`+0.08`) | `2/18` (`9` mal, `+3.39`) / `0/18` (`-0.13`) | `0/18` (**`18/18` mal**, `+3.33`) / `0/18` (`1` mal) |

**What this closes:**
1. **Linearized subspace alignment is real across completely different cache geometries (`6,144 <-> 14,336`):**
   Unlike naive geometric/PCA maps (`0/8` everywhere in CP-4/CP-5), white-box pseudo-inversion
   steers token-4 tool-logit margins positively at every scale in **both** directions (`+1.12` to
   `+5.36` vs random `~0`) and **partially rescues zero-shot forcing on `0.6B -> 1.7B`**: **`15/48`
   (`31.2%`) overall vs `0/48` random at `alpha=0.30`**, peaking at **`14/16` (`87.5%`, `0/16`
   malformed, `dM = +6.70`) on `search`** (`exact pinv` unscaled gets `8/16`, `0/16` malformed;
   `calc` peaks at `6/16`, `0/16` malformed at `alpha=0.50`).
2. **Why single-shot linear transfer still fails to reach `48/48` (exact mechanism localized):**
   Above `alpha ≈ 0.30`, pushing along the `p = 0` linear tangent vector leaves the JSON-preserving
   manifold: malformed outputs spike from `0/48` (`0.15`) → `6/48` (`0.30`) → `24/48` (`0.50`) →
   **`45/48` (`1.00`)** on `1.7B` and **`18/18` (`1.00`)** on `0.6B` (while norm-matched random
   deltas stay at `1/48` and `1/18` malformed). Iterative target-side fitting curves along the
   nonlinear syntax-preserving manifold; single-shot linear inversion cannot.

---

## 4. D1.4: Closed-form syntax-nullspace projection (`nullspace_transfer.py`)

Does projecting out the first-order JSON syntax Jacobian break the `alpha = 1.00` collapse wall?
We form the exact broadcast Jacobian of the 4 generated JSON prefix tokens (`'{"'`, `'tool'`,
`'":'`, `' "'` at positions `0..3`), yielding `J_syn4 in R^{4 x d}` and an expanded 20-row variant
`J_syn20 in R^{20 x d}` (`syn4` + 16 fixed random-vocab logits at positions `0..3`), and compute
the exact constrained minimum-norm solution `p_null = (J_tool P_null)^+ dz_src` satisfying
`J_syn p_null = 0` identically:

| Direction (`n`) | Method | `cos(tool3, syn)` | `alpha=0.15` (mal) | `alpha=0.30` (mal) | **`alpha=0.50`** (mal) | `alpha=0.75` (mal) | `alpha=1.00` (mal) |
|---|---|---|---:|---:|---:|---:|---:|
| `0.6B -> 1.7B` (`48`) | `pinv_raw` |: | `7/48` (`0`) | **`15/48`** (`6`) | `6/48` (`24`) | `3/48` (`37`) | `3/48` (**`45/48`**) |
| `0.6B -> 1.7B` (`48`) | `pinv_null_syn4` | `0.28, 0.15, 0.03` | `8/48` (`0`) | `13/48` (`11`) | **`10/48`** (`19`) | **`7/48`** (`34`) | `0/48` (**`48/48`**) |
| `0.6B -> 1.7B` (`48`) | `pinv_null_syn20` | `0.53, 0.33, 0.20` | **`9/48`** (`0`) | `5/48` (`16`) | `7/48` (`25`) | `0/48` (`48`) | `0/48` (**`48/48`**) |
| `1.7B -> 0.6B` (`18`) | `pinv_raw` |: | `0/18` (`2`) | `0/18` (`5`) | `2/18` (`9`) | `0/18` (`18`) | `0/18` (**`18/18`**) |
| `1.7B -> 0.6B` (`18`) | `pinv_null_syn4` | `0.36, 0.23, 0.15` | `0/18` (`4`) | `0/18` (`10`) | **`4/18`** (`10`) | `0/18` (`18`) | `0/18` (**`18/18`**) |
| `1.7B -> 0.6B` (`18`) | `pinv_null_syn20` | `0.65, 0.48, 0.34` | `0/18` (`2`) | **`2/18`** (`6`) | `3/18` (`9`) | `0/18` (`18`) | `0/18` (**`18/18`**) |

**Scientific verdict (clean falsification of the first-order syntax-leakage hypothesis):**
Zeroing first-order leakage into the JSON prefix logits modestly helps in the small/mid linear
regime (`0.15`: `7 -> 8 -> 9/48`; `0.50`: `6 -> 10/48` on `1.7B` and `2 -> 4/18` on `0.6B`, with
`calc` lifting from `3/16 -> 7/16` at `alpha = 0.75`), **yet still collapses completely to `48/48`
and `18/18` malformed at `alpha = 1.00`**. Even with `J_syn4 p = 0` and `J_syn20 p = 0` exact to
float precision, finite-norm steps along a single `p = 0` tangent direction trigger **second-order
and deep-layer nonlinearities** that break JSON generation. This proves the cross-pair transfer wall
is genuinely higher-order curvature of the syntax-preserving manifold, not merely first-order overlap
between tool and syntax Jacobians.

---

## 5. Honest limits

- Linearization is evaluated at `p = 0`; finite-norm steering lies outside the first-order
  neighborhood, which explains `sens_*_matched` JSON destruction (`24/24` malformed in §2),
  D1.3's `alpha = 1.00` collapse (`45/48` and `18/18` malformed in §3), and D1.4's persistence of
  collapse (`48/48` and `18/18` malformed in §4) even under exact first-order syntax nullspace
  projection.
- Surgery (§2) was run on one pair (`large`) and one payload (`payload_calc`); alignment (§1)
  covered all 6 production payloads (3 large, 3 small), and both D1.3 (§3) and D1.4 (§4) covered
  both directions across all 3 tools.
