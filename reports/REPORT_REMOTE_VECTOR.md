# Report: D2: Shared-prefix GCG injection under live vLLM CUDA prefix caching (reframed 2026-11-23)

Date: 2026-11-22 (audited & reframed 2026-11-23). Study: `studies/remote_vector_20261121/`.
Recount: `checks/verify_apc_remote.py`. Stack: g4dn/T4, vLLM 0.11.0, `Qwen/Qwen2.5-0.5B-Instruct`,
**`torch.float16`** (T4 compute capability 7.5 does not support native `bfloat16`; vLLM fell back
to `float16` per `runs/remote_spot/serve.log`), APC enabled.

---

## 0. Honest result in brief (what WAS and WAS NOT established)

> **Audit correction (2026-11-23):** An earlier draft framed this run as "API-only first-occupier
> cache poisoning" where the attacker's API request poisons later victims. Code and cache-lineage
> inspection disproved that framing. Under exact-token-match, value-transparent CUDA APC, an
> API-only client **cannot** make a cached block behave differently from cold prefill of the same
> tokens.

**What this run actually establishes:**
1. A GCG-optimized 58-token mixed-script string inserted into a shared RAG-style handbook prefix
   forces `Qwen/Qwen2.5-0.5B-Instruct` on live vLLM (`float16`, CUDA) to emit the leading token
   `' Ref'` on **10/10** victim questions (**6/6 strictly held-out** after separating 4 questions
   that overlapped `TRAIN_Q`) versus **0/10** on a benign document (`natural` and `random`).
2. **CUDA prefix caching is value-transparent:** on `runs_box_final/discriminator/rows.json`, cold
   prefill (`1_cold`) emits `' Ref'` at logprob **`-0.1020`** and cache hit (`3_hit`, `4_rehit`)
   emits `' Ref'` at **`-0.1023`** (on `q2`: `-0.1523` vs `-0.1505`). Thus **the adversarial text
   in the shared document causes the forcing, and APC preserves it transparently**: a cold victim
   with no prior attacker API request is forced identically.

| Arm (`runs_box_final/apc_remote/rows.json`) | Forced `' Ref'` (all 10) | Strictly held-out (6) | Notes |
|---|---:|---:|---|
| `natural` (benign handbook) | **0/10** | **0/6** | 14-token English filler (`P_rand`, 50 tokens) |
| `poisoned` (GCG handbook) | **10/10** | **6/6** | 58-token GCG doc (`P`, 96 tokens); text causes forcing |
| `random` (benign handbook + plant) | **0/10** | **0/6** | Same `P_rand` as `natural` (4.1× shorter than `P`'s doc) |
| `midblock` (truncated tail) | 10/10 | 6/6 | Shared same 5 blocks (80 tokens) as `poisoned` |
| `collision` (double-space prefix) | 10/10 | 6/6 | Harness bug: both plant and victim used `P_coll` |

**Scoring note:** The on-box `summary.json` wrote `0/10` because the original scorer stripped the
leading space before re-encoding (`'Ref'` token 3945 ≠ `' Ref'` token 8550). Rescoring the saved
completion text in `rows.json` recovers `10/10` (`6/6` strictly held-out) leading-`Ref` completions,
though **0/10** completions actually say `"Refused"`: 6/10 answer the user's question normally
after echoing `' Ref'` / `Ref业` / `Ref预备ext` copied from `doc_text`.

---

## 1. Payload provenance & fit statistics

- The actual payload served on the T4 box is `runs_box_final/fit_text/payload.json`:
  **120 GCG steps**, `Qwen2.5-0.5B-Instruct`, **train CE `0.1563`, held-out CE `0.1488`**,
  argmax `1.00` on both, `58` served document tokens (`52` raw `doc_ids`).
  *(An earlier draft accidentally quoted `1.04 / 0.92` from the local 220-step CPU fit in
  `runs/fit_text/payload.json`.)*
- **Train/test overlap:** 4 of the 10 questions in `apc_remote.py`'s `POISON_Q` (and both
  discriminator questions in `probe_cached_fresh.py`) appeared verbatim in `fit_text_payload.py`'s
  `TRAIN_Q`. The 6 genuinely disjoint questions still force **6/6** (`100%`).

---

## 2. Recorded flaws of the 2026-11-22 box harness (do not over-claim)

1. **No cache-written poisoning / broken `flush()`:** `apc_remote.py`'s `flush()` sent only
   `5 × ~210 = ~1,050` junk tokens into a **495,360-token (30,960-block)** T4 KV cache, evicting
   zero blocks. Exact 16-token chained-block replay reproduces `serve.log`'s hit rates (`77.1%`
   after the discriminator, `83.4%` end-of-session) and confirms `probe_cached_fresh.py`'s `1_cold`
   populated blocks `0..5` before `apc_remote.py` ran. The attacker's plant requests in `poisoned`,
   `midblock`, and `random` planted zero new blocks.
2. **Backwards `collision` probe:** Both the plant request and the victim requests in the
   `collision` loop sent `P_coll` (`"Company  handbook"`), so they matched and hit all 6 blocks
   rather than testing a cross-tokenization miss.
3. **5 shared blocks, not 6:** The trailing space of `"User question: "` (token index 95) merges
   via BPE into the first word of every suffix, so `P + att` and `P + q` diverged inside block 5
   (sharing 5 full blocks = 80 tokens, identical to `midblock`).
4. **Unmatched `random` control:** `rand_doc` sampled `len(doc.split())` = 12 English words (14
   tokens) vs 58 GCG tokens (**4.1× shorter**, clean English vocabulary).
5. **Latency is not TTFT:** `post()` is non-streaming wall-clock completion time comparing
   `max_tokens=8` victims (median `0.1636 s`, vs `natural` rows 1-9 median `0.1652 s`) against
   `max_tokens=1` plant requests (`0.0258 s`). No victim-side TTFT reuse gap was isolated.
6. **CPU backend divergence (unverified draft):** Local macOS ARM64 CPU runs (`vLLM 0.11.0`,
   Torch SDPA, chunked prefill disabled) showed a cached-vs-fresh logit shift, but the raw CPU
   JSON files were overwritten when the CUDA box artifacts were pulled. `reports/vllm_cpu_prefix_cache_issue_draft.md`
   is held as an incomplete draft until re-captured with a benign-prompt control.

---

## 3. Bottom line for the threat model

Under exact-hash, value-transparent prefix caching (CUDA vLLM), **an API-only client cannot poison
a cached block to behave differently from fresh prefill of the same text**. Threat actors who
control text inside a shared ingested document achieve **indirect RAG prompt injection**; prefix
caching preserves that injection transparently without altering it. True tensor-level cache
implants without payload text in the prompt still require an engine/routing-layer write (`#18`,
`studies/persistence_20261113/`) or hardware bit-flips (`#22`, `studies/hardware_bits_20261122/`).
