# C2C bridge attack — runnable artifact

Three 69 KB files that make a C2C bridge run **your** tool call instead of its own, under the
receiver's API key, with no credentials and no access to the fusion channel.

You do not read the wire. You add a small delta to a cache nobody inspects; the receiver emits
`{"tool": "cal"}`, its client parses it, and the calendar tool executes.

<!-- %SUMMARY%
An attacker with no credentials ships three small frozen payloads of a channel no person can
read; the receiver emits their answer and their whole tool call, the wrong tool runs under the
receiver's key, and every call still authenticates. This folder is that attack, runnable: three
69 KB files, one add per request, nothing optimized at deploy — plus a fitter for retargeting.
-->

## What is here

```
payloads/payload_search.npz   shipped payloads: force search / calc / cal
payloads/payload_calc.npz     (fit against the 1.7B pair; deploy with no fitting)
payloads/payload_cal.npz
bridge.py                     loads the pair, runs the fused forward (vendored rosetta)
client.py                     parses the receiver's JSON and dispatches it with the key
demo.py                       the whole loop: base vs payload, prints what executed
fit.py                        fit your own payload for a tool, on your own prompt
tools_server.py               mock gated tool server (401 without the key)
download.py                   fetches the four public checkpoints
parity_release.py             proves this copy matches the study bit for bit
rosetta/                      the fusion code (MIT), vendored so nothing outside is needed
```

About 7 GB of weights are downloaded to the standard HF cache. Everything is public and
ungated; no token is needed.

## Run it

macOS or Linux, one CUDA/MPS machine or CPU, Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# weights, once (~7 GB)
.venv/bin/python download.py

# the gated tool server, in its own shell
.venv/bin/python tools_server.py --port 8765

# the attack
.venv/bin/python demo.py
```

Expected, in ~25 s after the models load:

```
Q: What is 12 times 9?
   receiver alone : calc
   payload payload_cal     : cal [HTTP 200] -> 2026-10-31
   PASS
```

Your own question, forcing one tool:

```bash
.venv/bin/python demo.py \
  --prompt 'You have three tools. ... Reply with JSON only, shaped like {"tool": "calc"}. Q: Who wrote Hamlet?' \
  --payload payloads/payload_cal.npz
```

The prompt must mention JSON and give an example of the call shape — the payloads carry the
format, so a prompt with no JSON instruction at all is the weak case (see `../CAVEATS.md`).

## Fit your own payload

The shipped payloads work on the pair they were fit for (Qwen3-1.7B / Qwen2.5-1.5B). **Any other
pair needs a refit** — the attack does not transfer between pairs (see `../CAVEATS.md`). The
fitter is here so you can do that without the author's study tree:

```bash
.venv/bin/python fit.py --tool cal --out payloads/payload_cal.npz
```

It fits over five built-in prompts (~1-2 min on an M-series GPU), teacher-forcing the target JSON
and projecting the delta onto a 2%-norm ball. Verified after a fit: the payload overrode a correct
`search` answer to `cal` on a question it never saw. `--prompt` fits on a single prompt instead;
`--steps` and `--budget` are the knobs.

`fit.py` is a deliberate re-implementation of the study's optimizer against `bridge.py` only.
The bridge is what a real C2C deployment is: the stock forward detaches the cache, so the graph
has to be built by hand — that is the whole trick, and it is why a payload is needed.

**To retarget a different pair**, change `RECEIVER` / `SHARER` / `FUSER_SUB` in `bridge.py`,
re-run `download.py`, and fit. If the sharer's cache shape changes (layers/heads/dims), the
fitter follows the new shape automatically; the shipped payloads will not apply and must be refit.

## What the demo proves and does not

Proves: a file you hold, applied to a forward pass you do not control, changes the tool the
receiver executes. The base arm gets the question right; the payload arm gets it wrong on
purpose; both calls authenticate with a key you never see.

Does not prove: that a production deployment is exploitable, that a payload transfers to another
model pair, or that any monitor catches it. Read `../CAVEATS.md` before quoting a number.

## The mechanism in three lines

1. The sharer's KV cache is projected into the receiver's cache and trusted unread
   (`rosetta/model/wrapper.py`). The stock path detaches it, so there is no gradient — but
   there is no validation either.
2. With gradients enabled on your own forward, you optimize a small delta on the sharer cache
   toward a target tool token, then project it to a fixed L2 ball (`fit.py`).
3. At deploy, add the delta, broadcast over the prompt span, decode. One add per request;
   nothing is optimized.

`parity_release.py --study <path to the study folder>` checks that this copy produces the same
logits as the tree the payloads were fit in.

## Evidence behind the shipped payloads

Fit against Qwen3-1.7B / Qwen2.5-1.5B and the released fuser
`qwen3_1.7b+qwen2.5_1.5b_Fuser/final`: 24/24 held-out forced, 0 malformed; across six prompt
rewordings (`template_fixed2`, no refit, unseen questions) they force 58/72 cells (43/57
non-vacuous; three variants >=80%, `v1_reworded` and `v4_verbose` at 75% raw / 62.5–66.7%
non-vac, and `v3_no_example` without a JSON skeleton at 33.3% / 8 malformed). The same method
worked on a Llama sharer (7/7), on an independent
codebase (8/8) and on the 8B pair (8/8) — each a refit, which is why this is a fusion-design bug
and not a Qwen bug.
