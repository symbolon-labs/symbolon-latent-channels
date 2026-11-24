# Symbolon: Attacks on Unauthenticated Latent Channels

**Symbolon Labs** · Public research artifact & verification bundle (`R.11`, `v1.4`, 2026-11-24)

**TL;DR**
1. Latent handoffs (KV caches, bridges, pooled vectors) cross model boundaries without authentication or intent checks.
2. Transport signing (HMAC) fails when the sender or signing seam is compromised (`24/24` `[0.86, 1.00]` sender-signed attacks accepted).
3. Host-side tool allowlists and argument validation are what actually block unauthorized calls (`57/57` `[0.94, 1.00]` unoffered calls rejected); locking `<2.2%` of KV boundary blocks adds defense in depth (`>3x` to `5x` adaptive flip cost).

Read [`CAVEATS.md`](CAVEATS.md) before quoting any number. Every shipped run in this repository includes its raw per-item decode files (`items.jsonl`), exact bit-flip coordinates (`flips.json`), matched random controls, and standalone stdlib-only automated fail-closed recount scripts (`checks/`).

---

## Threat model, deployment scenarios, and scope

Two concrete deployment patterns place latent tensors on a trust boundary, each corresponding to a distinct threat tier:

- **Scenario 1 (Tier A, latent bridge / agent pipeline):** A two-stage pipeline where an intake model reading untrusted email, web pages, or photos passes a Cache-to-Cache (C2C) or latent bridge tensor to an executor model holding bearer-gated tool credentials. A compromised sender, bridge process, or crafted external carrier writes latent tensors across the handoff.
- **Scenario 2 (Tier B only, multi-tenant shared prefix cache):** A multi-tenant inference server using Automatic Prefix Caching (such as vLLM APC) where shared system or tool prefixes remain resident across requests. Our live CUDA vLLM APC run shows that exact-token prefix caching is value-transparent (`cold -0.1020 == hit -0.1023`), so API-only clients cannot poison a cached block without putting adversarial text in the shared prefix itself. Accordingly, the multi-tenant APC scenario applies only under **Tier B** (resident KV memory write or hardware fault injection flipping individual IEEE-754 `bfloat16` mantissa bits without changing exponents or signs).

We tag every measurement with its threat tier and report Wilson 95% confidence intervals alongside every ratio.

**Headline scope:** Under Tier B, single-token forcing (`' Ref'`) was measured across **six models from four families** (`12/12` `[0.76, 1.00]` each). Executed multi-token tool-call forcing via sparse bit flips was demonstrated **once**, on **`Qwen3-1.7B`**, for a fixed 6-token JSON call (`{"tool": "calc"}`, `8/8` `[0.68, 1.00]`).

---

## Reproducibility in one command

### 1. Cold-check every headline artifact (2 seconds, CPU only, Python 3 stdlib)

```bash
bash checks/verify_all.sh
```

Runs all **13 public automated fail-closed recount scripts** directly against the raw per-item artifacts in `studies/` plus **4 negative fault-injection tests** (corrupting a single per-item decode row, control trial, HTTP 200 status, or protected boundary-token flip and confirming every checker exits non-zero). The full internal lab suite covers `78` automated recounts and `12` negative tests across all tracks.

### 2. Reproduce the live 64-to-288 flip HTTP 200 tool call (`Qwen3-1.7B`)

```bash
python3 scripts/reproduce_64flip_toolcall.py
```

- **Pinned checkpoint:** `Qwen/Qwen3-1.7B` at Hugging Face commit `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **Fixed seeds:** `--seed 42` (random control seeds `20000 + mult*100 + trial` and `25000 + mult*100 + trial`).
- **Self-contained endpoint:** Automatically starts a local bearer-gated mock HTTP server (`Authorization: Bearer local-tools-key`) if `--server` is omitted, and writes `flips.json`, `items.jsonl`, and `summary.json`.
- **Reference environments:** Local bridge, modality, and Tier B runs used an Apple M1 Pro (16-core GPU, 32 GB unified memory) running Python `3.12.14`, PyTorch `2.15.0`, and Transformers `4.52.4`. The live CUDA vLLM APC run used an AWS `g4dn` NVIDIA Tesla T4 (`16 GB`, compute capability 7.5) running `vLLM 0.11.0` (`torch.float16`).

### 3. Run the local Tier A C2C wire demo (`release/`)

```bash
cd release
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py                     # public checkpoints, ~7 GB
.venv/bin/python tools_server.py --port 8765     # separate shell
.venv/bin/python demo.py                         # base call vs attacked call, HTTP 200
```

---

## Headline results (Wilson 95% CIs, reconciled 2026-11-24)

| Tier | Track | Carrier / seam | Result vs matched control (Wilson 95% CI) | Recount script |
|---|---|---|---|---|
| **A** | **Text bridge tool exec** | 69 KB frozen KV payload (`C2C`) | `16/16` `[0.81, 1.00]` HTTP 200 vs `0/16` `[0.00, 0.19]` random; score-only `7/8` `[0.53, 0.98]`; rewording `58/72` `[0.70, 0.88]` (`43/57` `[0.63, 0.85]` non-vacuous); 8B `8/8` `[0.68, 1.00]` vs `0/8` `[0.00, 0.32]` | `verify_codebook_factory.py`, `verify_c2c_template.py` |
| **A** | **Cross-codebase & 3rd-party** | Separate wires & planner->executor | CacheBack `8/8` `[0.68, 1.00]` vs `0/8` `[0.00, 0.32]`; Llama `7/7` `[0.65, 1.00]`; AVP `3/3` `[0.44, 1.00]` + `3/3`; Interlat `5/6` `[0.44, 0.97]` (`3/3` env exec, `11/12` `[0.65, 0.99]` vs `0/12` `[0.00, 0.24]`) | `reports/technical_report_ste_20261113.md` |
| **A** | **Voice, vision & composed** | Audio / image -> bridge -> host | Voice `8/9` `[0.57, 0.98]` vs `0/9` `[0.00, 0.30]` (speech `116/116` `[0.97, 1.00]`); Vision KV `46/46` `[0.92, 1.00]` vs `1/46` `[0.00, 0.11]`; Pixels `12/12` `[0.76, 1.00]` (`8/8` vs `1/8` at 1%); Composed `57/57` `[0.94, 1.00]` vs `2/57` `[0.01, 0.12]` | `verify_vision_pixel.py`, `verify_composed_chain.py` |
| **A** | **Host allowlist vs HMAC** | Runtime tool registry & wire manifest | Host allowlist rejects unoffered calls `57/57` `[0.94, 1.00]`; HMAC (Brito & Baquero, `arXiv:2606.28958`) catches transit tamper `24/24` `[0.86, 1.00]` but accepts sender-signed attacks `24/24` `[0.86, 1.00]` (`16/24` `[0.47, 0.82]` exec) | `verify_defense_registry.py` |
| **A** | **Forcing vs wire silence** | Rescaled codebook (`3,600` rows, `5` seeds) | Greedy drops `1.000@22.5x -> 0.000@<=1.29x`; quiet sampled `wrong_exec` (`0.104`) matches clean (`0.083`) and random (`0.108-0.117`) noise | `verify_tradeoff_seeds.py` |
| **A** | **Shared-prefix GCG + APC** | Live vLLM `0.11.0` (`float16`, Tesla T4) | GCG handbook forces `' Ref'` `10/10` `[0.72, 1.00]` (`6/6` `[0.61, 1.00]` held-out) vs `0/10` `[0.00, 0.28]` benign; CUDA APC is value-transparent (`cold -0.1020 == hit -0.1023`) | `verify_apc_remote.py` |
| **B** | **Dense-delta XOR (Part A)** | Dense Adam delta -> mantissa XOR | Forces `0/10` `[0.00, 0.28]` up to `100%` (`92.7M` flips, `0` exp/sign flips, Hamming `<=1`); controls `0/50` item-evals | `verify_directed_bits.py` |
| **B** | **6-model first-token trend (Parts B & E)** | Fresh sparse `bf16` mantissa XOR (`0.5B-4.0B`) | `12/12` `[0.76, 1.00]` held-out & CPU match on all 6 models; `0/720` `[0.00, 0.01]` 1x control item-evals across 60 trials (`6 models x 2 arms x 5 seeds x 12 items`, plus `0/240` in 20 10x trials; `0/960` total); `beta_params = -0.42` (`[-0.79, -0.05]`, `R^2 = 0.71`), borderline `beta_N = -0.33` (`[-0.66, 0.00]`, `R^2 = 0.66`), scale/family confounded; `81.5%-100%` in Block 0 + Final Block | `verify_sparse_pursuit.py`, `verify_scaling_law.py` |
| **B** | **Single-model 6-tok tool call (Part D)** | `Qwen3-1.7B` 422-token tool prefix | `8/8` `[0.68, 1.00]` unseen non-`calc` emit `{"tool": "calc"}` & execute HTTP 200 (`0/8` `[0.00, 0.32]` malformed, CPU `8/8`) from `64` flips (`0.00026%`, `0.15%` L2); train-stop `288` flips; `0/160` `[0.00, 0.02]` control item-evals across 20 trials (`1x` and `10x`, `0/8` each). Conditional trigger: `4/6` `[0.30, 0.90]` trigger, `0/6` `[0.00, 0.39]` clean FP, `2/6` `[0.10, 0.70]` exact clean top-1 | `verify_toolcall_trigger.py` |
| **B** | **Boundary-block lock (Part C)** | Lock Block 0 + Final Block (`<2.2%`) | Defense in depth, not a full fix: raises adaptive cost `>5.1x` on `Qwen3-1.7B` (`11/12` `[0.65, 0.99]` at `1,792` flips) and `>3.0x` on `Llama-3.2-1B` (`5/12` `[0.19, 0.68]` at `1,920`); Block 0 alone fails on Llama (`1.1x`) | `verify_boundary_defense.py` |
| **A** | **Jacobian geometry (D1.1-D1.4)** | Broadcast space (`d=14,336` / `6,144`) | Quiet surgery `16/16` `[0.81, 1.00]` vs tangent collapse `24/24` `[0.86, 1.00]`; D1.3 `J_tgt^+` rescues `15/48` `[0.20, 0.45]` at `alpha=0.30` (`14/16` `[0.64, 0.97]` search); D1.4 syntax-nullspace lifts `alpha=0.50` to `10/48` `[0.12, 0.34]` yet collapses `48/48` `[0.93, 1.00]` & `18/18` `[0.82, 1.00]` at `alpha=1.00` (working hypothesis: higher-order curvature) | `verify_geometry_decompose.py` |

---

## Recommendations (in priority order)

1. **Host-side tool allowlists and strict argument schema validation:** Primary control (`57/57` `[0.94, 1.00]` unoffered calls blocked). Treat receiver tool calls as untrusted whether fed text or latent state.
2. **Confirmation step for risky or state-changing tools:** Allowlists do not stop wrong-but-offered tool calls; gate sensitive actions behind out-of-band policy or user confirmation.
3. **Boundary-block KV locks as defense in depth:** Lock Block 0 and the Final Prefix Block (`<2.2%` of cache) to multiply adaptive Tier B bit-flip cost `>3x` to `5x`.
4. **Transport HMAC for transit integrity only:** HMAC catches third-party wire tampering (`24/24` `[0.86, 1.00]`) and proves who sent the payload, but does not stop a compromised sender (`24/24` accepted).

---

## Related work

Prior work has already shown that KV caches can be flipped or overwritten inside a single model. Our contribution rests on three specific differences: **we restrict KV bit flips strictly to `bfloat16` mantissa bits (zero exponent or sign flips), we drive a multi-token JSON tool call that executes under bearer credentials, and we attack unauthenticated model-to-model latent handoffs across bridges and modalities:**

- **KV cache bit flips and manipulation:** Nahian et al. (*CacheTrap*, [`arXiv:2511.22681`](https://arxiv.org/abs/2511.22681), 2025) demonstrated targeted single-bit flips in a single model's KV cache by flipping the FP16 exponent MSB (bit 15) and evaluated hardware fault delivery; Ganesh, Iyer, & Ananthan (*Whose Narrative is it Anyway? A KV Cache Manipulation Attack*, [`arXiv:2511.12752`](https://arxiv.org/abs/2511.12752), 2025) showed block-level KV overwriting (*History Swapping*) redirects a single model's narrative trajectory; Yamamoto & Matsuura (*Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems*, [`arXiv:2604.17249`](https://arxiv.org/abs/2604.17249), 2026) analyzed undirected bit-flip faults in shared vLLM prefix blocks. We restrict edits strictly to mantissa bits, show boundary-block concentration across four model families, force an executed 6-token HTTP 200 tool call on `Qwen3-1.7B`, and evaluate `<2.2%` boundary-block locks.
- **Weight bit flips and attention sinks:** Terminal Brain Damage (Hong et al., USENIX Security 2019) and DeepHammer (Yao et al., USENIX Security 2020) targeted model weights; StreamingLLM (Xiao et al., ICLR 2024) identified initial-token attention sinks, matching our Tier B concentration in Block 0 and the Final Prefix Block.
- **Latent-wire integrity:** Brito & Baquero (*When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration*, [`arXiv:2606.28958`](https://arxiv.org/abs/2606.28958), 2026) proposed an HMAC-SHA256 manifest binding specialist identity, session, model, visible commitment, tensor metadata, and payload digest, noting that a compromised signing endpoint sits outside transport integrity; our `24/24` sender-signed test confirms that boundary directly.
- **Tool-use prompt injection and host allowlists:** Indirect prompt injection (Greshake et al., 2023), OWASP LLM06:2025, Piotr Czapla (*The unauthorized tool call problem*, Answer.AI, [`answer.ai/posts/2026-01-20-toolcalling.html`](https://answer.ai/posts/2026-01-20-toolcalling.html), 2026), and Iyer & Babu (*ContractGuard*, [`arXiv:2606.18550`](https://arxiv.org/abs/2606.18550), 2026) establish text-level tool steering and host capability boundaries; we show the same failure persists across latent KV bridges.

---

## Disclosure and scope

- **Cache-to-Cache (`nics-efc/C2C_Fuser`):** Privately disclosed on `2026-10-24` with a two-week window (through `2026-11-07`); no response received; findings were first published publicly on `2026-11-24` (`31` days after disclosure).
- **CacheBack, AVP, and Interlat:** Evaluated locally against public research repositories without separate prior vendor outreach and reported as class-level demonstrations.
- **Zero production targeting:** All experiments ran on open-weight models (`0.5B-8B`) and local mock endpoints. Internal safety-benchmark raw outputs, exploratory offensive notes, and cloud deployment scripts are excluded from this public release.

---

## Repository layout

- `scripts/reproduce_64flip_toolcall.py`: Single-command live reproducer for the 64-to-288 flip HTTP 200 JSON tool-call result.
- `reports/`: Full STE technical report (`technical_report_ste_20261113.pdf` & `.md`), 7 figures (`reports/figs/`), and individual study reports (`REPORT_DIRECTED_BITS.md`, `REPORT_GEOMETRY.md`, `REPORT_TRADEOFF_SEEDS.md`, `REPORT_REMOTE_VECTOR.md`).
- `studies/`: Sanitized per-item decodes (`items.jsonl`), exact flip coordinates (`flips.json`), run rows, summaries, and split manifests.
- `checks/`: 13 stdlib-only automated fail-closed recount scripts + `verify_all.sh` (including 4 negative fault-injection tests).
- `release/`: Local Tier A C2C demo, pinned loader, mock tool server, and `69 KB` `.npz` payloads.
- `CAVEATS.md`: Scope limits, honest negatives, and boundaries of every claim.
- `SHA256SUMS`: SHA-256 checksums for every file in the release (`shasum -a 256 -c SHA256SUMS`).

## Changelog

- **v1.0 (2026-11-24):** Initial public release (31 days after C2C disclosure) of sanitized artifacts, 13 stdlib recounts, 4 negative tests, and technical report.
- **v1.1 (2026-11-24):** Cleaned naming and style across docs and scripts (`verify_composed_chain.py`, `verify_toolcall_trigger.py`).
- **v1.2 (2026-11-24):** Added 3-line TL;DR, Tier A / Tier B split, deployment scenarios, scoped headline, Wilson 95% CIs, scaling trend `beta` CIs and family confound note, explicit hypothesis labels, related work, ordered recommendations, disclosure notes for all four evaluated projects, and single-command `scripts/reproduce_64flip_toolcall.py`.
- **v1.3 (2026-11-24):** Corrected disclosure timeline (`31` days, `2026-10-24` to `2026-11-24`), scoped multi-tenant APC explicitly to Tier B, separated internal (`78`/`12`) vs public (`13`/`4`) test counts, fixed Tier B control item-eval denominators (`0/720` 1x / `0/960` total and `0/160`), led scaling with `beta_params` while noting borderline `beta_N`, added full verified citations and Tesla T4 hardware specs, and replaced "independent" / "deployed" phrasing.
- **v1.4 (2026-11-24):** Positioned Related Work and Limits explicitly against CacheTrap (`arXiv:2511.22681`, FP16 exponent MSB flips and GPU hardware fault delivery) and Ganesh, Iyer, & Ananthan (`arXiv:2511.12752`, History Swapping).
