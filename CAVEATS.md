# Caveats: read before quoting any number

One page. Everything here is a limit of what the results support, listed so the claim does not
grow in repetition. The study reports (`reports/`) each carry their own
"honest reading" section; this is the union.

## Target

- **A research bridge, not a product.** The receiver, sharer and fuser are public research
  checkpoints (`nics-efc/C2C_Fuser`). Disclosure: emailed to the bridge authors
  2026-10-24 with a two-week window (through 2026-11-07); no response; published 2026-11-24 (31 days after disclosure).
- **One codebase, then two, then a stranger's.** Every result up to 2026-11-02 is on `rosetta`.
  That day the same attack was ported to **CacheBack**, an independent implementation by other
  authors with a different wire (receiver embedding rows). On 2026-11-06 a *third-party, runnable*
  latent bridge (**AVP**, `VectorArc/avp-python`, zero training) was attacked on its own wire and
  its output forced (3/3 vs random 0/3). Three codebases is "this design pattern"; it is not proof
  that every latent-transport system is vulnerable. All of AVP/C2C/CacheBack are **Qwen-family**
  and the AVP hit is a **reach** rung (a stranger's thing fell), not a new architectural class;
  AVP's effect is a forced *output*, not a tool execution. On 2026-11-07 a third-party latent
  **agent with a real action step** (Interlat, ALFWorld planner→executor) was stood up and its
  receiver's environment action forced (5/6 vs random 0/6, base correct, action executed 3/3): the
  action-step leg AVP lacked, still a **reach** rung, Qwen-family and per-pair. A learned cross-pair
  mapper (SwaT) was
  trained once and a delta **survives** it (12/12 vs random 0/12), but the base emits **no tool word
  at all** (0/12): so the delta *fills a blank*, it does not yet *redirect* a correct choice: and
  the mapper is itself pair-specific. Stage-2 CE (~$0.30) is the fix for the redirect.
  **Update 2026-11-08:** the stage-2 CE did *not* fix it, but a **decode bug** in the harness did : 
  after aligning the decode and training on the tool distribution, the base is correct 6/12 and the
  delta **redirects it 6/6** (attack 12/12, random 1/12). So through SwaT the delta *does* redirect
  a correct base; the mapper is still per-pair, so the refit asterisk is reframed, not killed.
  Separately, the **sender-only no-infrastructure route (R1) is CLOSED**: a doc+delta crafted on a
  train split forces 0/6 held-out, even at a per-item-refit ceiling of 0/6: the delta does not
  survive the sharer-only path.
- **Small-to-mid models.** 0.6B, 1.7B and **8B** receivers; the 8B run is the largest released
  pair (Qwen3-8B/Qwen2.5-7B) and it landed 8/8 vs random 0/8. "Small models" is no longer a
  scope limit, though nothing larger has been tried, and the 8B receiver ran in 8-bit as a
  memory accommodation.
- **"Unauthorized" means the key boundary.** The attacker has no HTTP credentials. The tool
  endpoints are local mocks behind `Authorization: Bearer …`; this is not a test of a real
  tenant, cloud IAM, or a vendor's sandbox.

## Method

- **Refit per pairing, not transfer.** The main results are fit per prompt pair; the codebook
  and the Llama family run are refit per pair. Different model pairs have different cache
  shapes (28x2x128 Qwen vs 16x8x64 Llama), so a literal cross-pair transfer: with a
  projection map: is unbuilt. "Works on a different family" means *the method works after a
  refit*, not that one file moves.
- **And the transfer rung failed (2026-11-02).** A payload fit on the 1.7B pair, mapped
  deterministically onto the 0.6B pair, is indistinguishable from a random norm-matched delta
  (0/3 non-vacuous; a target-side refit gets 3/3). So the refit asterisk is real and unclosed.
  `REPORT_XPAIR_MAP.md`.
- **And it failed twice, at 20× the budget (2026-11-04).** CP-5 swept budget 0.02→0.4, tried the
  reverse direction (0.6B→1.7B, 8/8 non-vacuous base), and applied the two named map fixes
  (relative-depth alignment, a label-free source-cache PCA dim map). The mapped payload still
  fired 0/8 at every budget while a target-side refit fired **8/8 at every budget**. At the top
  budget the map even damages the JSON without steering it. The last standing cause is the
  learned projector. `REPORT_XPAIR_SWEEP.md`. **Refit per pairing is final.**
- **White-box fit for the main results.** The optimizer needs gradients through the fused
  forward, which the stock path detaches. The deployable payload needs no gradients, but
  whoever fits it does.
- **Black-box is partial.** From score feedback only (logprobs, no gradients): 7/8 wrong-tool
  execution and 12/18 on the codebook. From **labels only**: 0/4: a negative, kept on purpose,
  because 1-bit feedback is structurally zero at this radius.
- **Arguments were forced 6 of 8 times.** The two misses lost the argument, not the tool; the
  tool was forced 8/8 in that run. So "the attacker writes the call" holds for the tool
  reliably and for the argument most of the time.
- **The prompt must mention JSON.** The payloads carry the call format. Six rewordings were
  tried (`template_fixed2`: 58/72 raw, 43/57 non-vacuous); three variants fell below 80%
  (`v1_reworded` 9/12 raw / 5/8 non-vac, `v4_verbose` 9/12 raw / 6/9 non-vac, and `v3_no_example`
  with no JSON skeleton at 4/12 raw / 3/11 non-vac, 8 malformed). A prompt with no JSON
  instruction is untested and predicted to break the attack.
- **One seed, small n, no preregistration.** Counts are what they are (7/7, 16/16, 24/24), with
  a norm-matched random control in each run and an independent recount script in `checks/`.
  They are not powered effect sizes.

## What the results do **not** show

- That any **production** deployment is exploitable.
- That any **mitigation** fails. Detectors were built and scored (0.94 AUROC, 0.06 TPR at 1%
  FPR): a bad monitor, not proof no monitor works. Canary carriage and wire monitors were
  null on the text route; that is not the same as "undetectable".
- That a payload **transfers between models**, families or sizes, without a refit.
- That **every** latent-handoff codebase is vulnerable (three independent codebases fell: C2C,
  CacheBack, AVP/Interlat: a recurring pattern across tested bridges, not an exhaustive proof).
- That models **larger than 8B** (or full-precision 8B without 8-bit quantization) behave the same.

## The composed end-to-end execution specifically

- **Forcing and silence do not coincide.** The shipped payload forces (wrong tool executes,
  authenticated) but moves the wire ~22× the benign spread, so a content-aware monitor sees it.
  Rescaling to hide (ratio < 1) drops forcing to zero; a refit at the quiet budget does not
  rescue it (0.002 → forces but ratio 2.4; 0.001 → ratio 1.3 and forcing 0.04). The honest line:
  "the strong codebook is loud; a quiet payload is weak, and no budget is both."
  `REPORT_EXECUTION.md`.
- The composed chain's "silent" is measured against **our** monitor (a kNN on a subsampled cache
  fingerprint), not a production one. `REPORT_SUITE.md`'s 6%-at-1%-FPR is a different monitor set
  and a different claim.

## The sender-side (R1) document arm specifically

- **Thin and placement-split.** An optimized document moves the sharer cache only when it also sits
  in the receiver's prompt (SHARED 2/12 vs control 0/12); when it reaches the receiver *only through
  the bridge* (ASYM) it is inert (0/12). This is the canary-null reproduced on the routing task, not
  a new channel.
- **The decision margin is not a flip certificate.** Margin moved on 11-12/12 items but greedy
  followed rarely: optimising the forced-prefix decision position overfits a position free
  generation does not stand on. A greedy-in-the-loop objective is the untried fix.
- **The JOINT arm (doc + delta, the "contact injection") is a DECISIVE NEGATIVE on the true
  no-infra placement.** One doc + one delta crafted on a train split, applied to 6 held-out
  prompts (ASYM, sender-only): attack 0/6, doc-only 0/6, random 0/6, **per-item-refit ceiling 0/6**.
  A delta that forces 16/16 when applied receiver-side does *nothing* through the sender-only path : 
  it does not survive the sharer→receiver projection. **The pure no-middlebox route is closed.**
  The only open R1 variant is the SHARED joint arm, which is no longer a no-middlebox position.
- Small n (12), one seed, one pair (Qwen3-1.7B/Qwen2.5-1.5B), one template. `REPORT_SENDERSIDE.md`.

## The leak blade specifically

- It is a **per-prompt** result (refit per item), not a frozen payload; the frozen leak payload
  is unbuilt.
- Outputs are **ragged**: the nonce arrives, the sentence does not, and several items degenerate
  into repetition (`MDB-9466666666`). Two of six only pass a strict-shape test.
- The document is synthetic and templated, with the secret near the tail. A longer record with
  the secret further from the question is a harder alignment problem, untested.
- Budget is fixed at the level the sweep said worked; whether blinding shifts the threshold is
  unknown.
- No monitor arm: the `REPORT_SUITE.md` detectors were built for the integrity task.

## Positioning against prior work (updated 2026-11-14, `PRIOR_ART.md`)

- **Do not claim the latent channel is unstudied, or that "no integrity check" is our idea.**
  The 2605-2607 cluster studies latent channels; `arXiv 2606.28958` ("When Latent Agents Lie")
  proposes an HMAC transport manifest and is the closest work. Cite it; do not re-litigate it.
- **What is still ours and must be defended, not assumed:** external attacker (no model, no key);
  the effect is **unauthorized tool execution under the receiver's key**, not a wrong answer;
  reproduced on a second independent codebase; a working black-box mount (2606.28958 found
  gradient-free weak). Their target is their own protocol; C2C and cross-family bridges are out of
  scope there.
- **The reviewer trap is now closed as a test, not a claim:** we implemented the 2606.28958 HMAC
  manifest and attacked it. Controls perfect (honest 24/24 accepted, tamper 24/24 rejected); our
  sender-side delta is accepted 24/24 while the wrong tool runs 16/24: the manifest's own
  endpoint-signing blind spot. And neither released bridge ships a manifest. `REPORT_DEFENSE_TEST.md`.
- **Caveat on that:** the manifest is necessary, not sufficient. A defence must bind *what the
  cache should be* to a principal the attacker lacks (sender attestation, re-derivation, or a
  behavioural cross-check). The manifest implementation was tested against the manifest as described
  in `arXiv:2606.28958` v1 (not their code; the authors' final design may differ); the 24/24 reject
  control is what makes it credible. The general form: a transport MAC
  authenticates *"unchanged," not "correct"*, so a sender-side attacker is authoritative: is
  `REPORT_ATTESTATION.md` (route R2).
- **Broader positioning (2026-11-14).** The modality carriers, the projector-poison technique,
  the prefix-cache cousin, and the held tool-registry defense all have prior or concurrent art.
  The per-claim verdict and the honesty concessions are in `PRIOR_ART.md`. The full verified
  ledger is in `reports/LLM latent channel attack prior art.md`.

## What the codebase port does **not** add

- It is a different *mechanism* (row transport, not cache fusion) in a different program. Fair
  as a class test, not as a second instance of the same bug.
- Same checkpoint on both sides (Qwen3-0.6B), one template, one seed.
- CacheBack supports dense Qwen3 only; the port is scoped to that.

## Reporting rules used in this work

- Every headline number has an independent recount in `checks/` with its own parser.
- Vacuous items (the receiver already agreed) are excluded and the exclusion is stated.
- Failures stay in the record (the label-only arm, the single-payload codebook attempt).

## New work (2026-11-12/15): publication status

Per the publication policy of 2026-11-15 (see NEXT_SESSION.md):

* The **cache-implant mechanism** (`studies/persistence_20261113/`, internal): first the local demo
  (forced token 10/10 across a process restart), and then the **real-vLLM behavioral implant**
  (live vLLM with APC: base 0/10 natural, poisoned cached blocks 10/10 forced, persisted on the
  dwell re-check; verified internally, while the public CUDA APC transparency check is
  `checks/verify_apc_remote.py`). Quote only with its boundary: one small model
  (Qwen2.5-0.5B) served by vLLM, one seed, first-token metric with a degenerate continuation,
  and the injection point is the engine's KV tensors (the compromised-routing-layer model: not
  a through-the-API tensor write). Selective (trigger-conditional) firing is partial: the single
  phrase works (4/4), a second behavior and clean-question naturalness are open (0/4, 1/6).
* The **checkpoint self-healing, the cross-tenant implant, the agent-to-agent worm, and the
  targeted-worm overfire** are internal work in progress. Do not quote them.
  theory with one flagged hypothesis. Do not quote them as results.

## Depth pass (rebuilt & audited 2026-11-23)

* **Shared-prefix GCG + CUDA APC transparency (`studies/remote_vector_20261121/`).** Reframed
  2026-11-23 after code audit: a 58-token GCG handbook string forces leading `' Ref'` on `10/10`
  questions (**`6/6` strictly held-out** after separating 4 questions that overlapped `TRAIN_Q`)
  vs `0/10` benign on `Qwen2.5-0.5B-Instruct` / vLLM `0.11.0` / T4 (`torch.float16`, not `bf16`).
  CUDA APC is value-transparent (`1_cold` `-0.1020` == `3_hit` `-0.1023`), which proves **the
  text causes the forcing and APC preserves it transparently**: this is **not** API-only
  first-occupier cache poisoning (a cold victim is forced identically). Recorded flaws of the
  Nov 22 box run: `flush()` evicted 0/30,960 blocks, `collision` sent `P_coll` on both plant and
  victim, `P_rand` was 4.1× shorter, 0/10 completions said `"Refused"`, and raw CPU divergence
  artifacts were overwritten (`reports/vllm_cpu_prefix_cache_issue_draft.md` is held as incomplete).
* **Forcing-silence tradeoff with clean controls & cluster CIs (`studies/theory_20261121/`).**
  Rebuilt 2026-11-23 (`3,600` rows, `24` items = `12` train + `12` held-out, `f=0.0` clean +
  norm-matched random controls, unique per-cell RNG, LOO wire ratios, cluster 95% CIs). Greedy
  targeted forcing (`target != expected`) drops monotonically `1.000` (`22.5×`) → `0.854`
  (`11.3×`) → `0.354` (`5.7×`) → `0.042` (`2.3×`) → **`0.000` (`≤1.29×`)**. Under `temp=0.7`
  sampling, quiet `wrong_exec` (`0.104` at `1.29×` and `0.77×`) is indistinguishable from the
  clean (`0.083`), same-tool (`0.100-0.125`), and random (`0.108-0.117`) noise floor: the
  earlier "~6% slow-drip" claim is retracted. Boundaries: one pair, one Key-only kNN monitor,
  rescaled frozen codebook (no quiet-budget refit).
* **4-family sparse directed bit flips, live tool-call execution & boundary defense (`studies/hardware_bits_20261122/`).**
  Rebuilt and expanded 2026-11-23 with strict IEEE-754 `bfloat16` single-mantissa-bit XOR (`uint16 ^ (1 << bp)`,
  `bp in 0..6`, `0` exponent/sign flips, Hamming `= 1`), train-only stopping (zero held-out peeking), CPU
  cross-checks, and matched uniform & position-matched random controls:
  - *Dense-intent reversal (Part A):* Naive XOR projection of the dense Adam `#65` delta forces `0/10` up to 100%.
  - *4-family / 6-model scaling law (Parts B & E):* Sparse pursuit forces **`12/12` strictly unseen cross-domain
    questions (`6/6` shipping + `6/6` trivia, CPU-verified identical, `0/12 × 5` uniform & pos-matched controls)**
    across all 6 tested models (`Qwen2.5-0.5B`: `576` flips; `Llama-3.2-1B`: `640`; `Granite-3.3-2B`: `288`;
    `Phi-4-mini-3.8B`: **`224`**, `0.00023%`; `Qwen3-1.7B`: `352`; `Qwen3-4B`: `352`, `0.00028%`), yielding a
    **negative scaling law (`beta_N = -0.332`, `beta_params = -0.420`)**: larger models need *fewer* bit flips : 
    and **`81.5%-100.0%`** concentration in Block 0 + Final Block (`Qwen3` loads Block 0 `72%-76%`; the other
    four models load the Final Block `57%-81%`).
  - *Boundary-block defense (Part C):* Locking only Block 0 (`<1.1%` cache) slows `Qwen3-1.7B` (`1.9×`) but fails
    on `Llama-3.2-1B` (`1.1×`). Locking **both Block 0 + Final Block (`<2.2%` cache)** blocks `12/12` up to
    `1,792` flips (`>5.1×`, stops `11/12`) on `Qwen3-1.7B` and `1,920` flips (`>3.0×`, only `5/12`) on
    `Llama-3.2-1B`, forcing relocation into Block 1.
  - *Multi-token tool-call execution & conditional trigger (Part D):* On a 422-token shared tool prefix (`Qwen3-1.7B`),
    **`64` flips (`0.00026%`, `0.15%` L2)** drive **`8/8` strictly unseen non-`calc` items** to freely emit valid
    `{"tool": "calc"}` JSON and execute with bearer-gated **HTTP 200 (`0/8` malformed)**; at the train-only stop
    (`288` flips, `0.29%` L2), CPU cross-check confirms `8/8` executed and uniform & position-matched controls
    score **`0/8 × 5`** even at `10×` (`2,880` flips). A KL-regularized `"refund"` conditional trigger (`544` flips, `0.23%` L2)
    forces `4/6` unseen trigger items (`5/6` peak) with **`0/6` clean FP across all 17 rounds** (`0.488` nats KL),
    but exact greedy clean top-1 preservation is partial (`2/6`, CPU-verified identical).
  - *Boundaries:* Physical DRAM/HBM Rowhammer/GPUHammer deliverability of `64-352` targeted flips across 2 physical
    KV pages is unmeasured; conditional trigger clean preservation (`2/6` exact top-1) is partial.
* **Token-4 broadcast Jacobian geometry, D1.3 `pinv` & D1.4 syntax-nullspace transfer (`studies/projector_geometry_20261123/`).**
  Rebuilt 2026-11-23 in the actual broadcast space (`d = 14,336` large / `6,144` small) at generated
  token 4 (`'{"tool": "'` → tool name) with Rayleigh-Ritz randomized SVD and disjoint-template
  prompts. Top-8 captures `44%-83%` Frob² and is prompt-stable (`0.72-0.88` leading cosine even
  with zero shared template tokens). Production payloads are **`12×-277×` enriched vs random**
  in `tool3`/`vocab8` yet retain `>94%` of energy in the complement. Surgery (`n=16` non-vacuous,
  HTTP 200): `intact`/`quiet_tool3`/`quiet_vocab8` execute `16/16` (`0/24` malformed), while
  norm-matched `sens_tool3_matched`/`sens_vocab8_matched` destroy JSON **`24/24`** (`0/16` forced,
  vs `random_matched` `3/16` forced, `8/24` malformed); norm-matched Key-only LOO monitor is
  `22.5×` (`quiet`/`intact`) vs `7.3×-8.4×` (`sens_matched`, Value-heavy; earlier unscaled
  `61.9× vs 4.1×` inversion retracted). **D1.3 Moore-Penrose pseudo-inverse transfer (`J_tgt^+ @ dz_src`):**
  steers margins positively in both directions (`+1.12` to `+5.36` vs random `~0`) and **partially
  rescues zero-shot `0.6B -> 1.7B` transfer (`15/48` vs `0/48` at `α=0.30`, `14/16` on `search`)**,
  before hitting JSON collapse at `α=1.00` (`45/48` and `18/18` malformed). **D1.4 closed-form
  syntax-nullspace projection (`(J_tool P_null)^+ @ dz_src`, zeroing `J_syn4` and `J_syn20` at gen
  pos `0..3`):** modestly lifts small/mid regimes (`α=0.15`: `7→8→9/48`; `α=0.50`: `6→10/48` on
  `1.7B`, `2→4/18` on `0.6B`), **yet still collapses `48/48` and `18/18` malformed at `α=1.00`** : 
  proving the cross-pair transfer wall is genuinely **higher-order nonlinearity**, not first-order
  syntax-logit leakage.

## R.12 follow-up: multi-family tool calls, arbitrary arguments, and schema-aware KV locks (2026-11-27)

* **Scope (`studies/multifamily_toolcall_20261126/`, `reports/REPORT_R12.md`, `checks/verify_r12_multifamily_toolcall.py`):**
  30 audited runs across 4 open-weight 1B to 3.8B models (`Qwen3-1.7B`, `Llama-3.2-1B-Instruct`,
  `Granite-3.3-2B-Instruct`, `Phi-4-mini-3.8B-Instruct`), `7,974` per-item decode rows, CPU
  `bfloat16` cross-checks, `0/6,720` matched random control executions (`95% CI [0.000, 0.0006]`),
  and `3/3` R.12 negative fault-injection tests (`14/14` public automated recounts and `7/7` negative
  tests across the repository). Models larger than 3.8B were not tested in R.12.
* **Llama-3.2-1B denominator caveat:** On the one-field `{"tool": "calc"}` task (Gate 1),
  `Llama-3.2-1B-Instruct` scored `0/12` on clean baseline calendar queries, so its clean held-out
  evaluation split consists of `n = 6` `search`-only queries (`6/6` at 32 flips) rather than `12`.
* **Arbitrary arguments (`fit = 4` vs `fit = 8`):** Exact 14-to-18-token attacker-chosen arguments
  missed the preregistered 3-of-4 >=80% bar at `fit = 4` (`Granite` `10/12` `"audit-export"`,
  `Llama` `7/12` `"1337*42"` -> `56154`, `Qwen3` `2/12` & `3/12`, `Phi-4` `0/12`). Doubling
  disjoint training prompts from `4` to `8` jumped `Qwen3-1.7B` to `9/12` (`10/12` CPU) on
  `"1337*42"` -> `56154` and `12/12` (MPS & CPU) on `"audit-export"`, but that `fit = 8` follow-up
  ran on `Qwen3-1.7B` only.
* **Self-correction of R.11's 2-block lock & defense caveats:** Locking only Block 0 + Final 16
  tokens failed in `7/7` runs on tool-calling prefixes (short few-shot, zero-shot, and ~2.3K-token
  prefixes) because `98.1% to 99.7%` of adaptive bypass flips relocated into unlocked tool-schema
  and few-shot tokens. Locking **Block 0 + the contiguous tool-schema tail** restored a `5x to >20x`
  cost wall in `7/7` runs at `5.1% to 5.5%` (`116 to 128` tokens) of a ~2,300-token cache, **tested
  up to a 1,280-flip budget under the same sparse-pursuit optimizer**. It is defense in depth, not
  an unbounded guarantee (`Qwen3-1.7B` still reached `7/12` at 1,280 flips on the 2.3K prefix), and
  paged serving engines that lock whole 16-token KV blocks incur modest alignment rounding (roughly
  128 to 144 tokens, ~5.6% to 6.3% on a 2.3K prefix). Physical DRAM/HBM Rowhammer or GPUHammer
  delivery was not measured.
