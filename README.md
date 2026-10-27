# Muse-Spark-Open-816B-A42B-NVFP4 on 8× DGX Spark — Runbook

Serve **`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`** (816B MoE / 42B active, NVFP4 W4A4
group-16, ~533 GB of weights) on a cluster of **8× NVIDIA DGX Spark (GB10, sm_121)** as a
**TP8 multi-node** deployment with **DFlash 2 speculative decoding**.

> **TL;DR:** boot **~9–11 min** → **~90 tok/s aggregate** at 8 concurrent requests
> (~11 tok/s per request). Validated stable on 2026-10-23 after 23 phases of work and
> 6 failed attempts. Nearly every failure you may hit is listed with its fix in
> [`TUTORIAL.md` §9](TUTORIAL.md).

## Contents

| File | What |
|---|---|
| `TUTORIAL.md` | **The runbook** — setup, build, deployment, validation, tuning, troubleshooting |
| `Dockerfile` | Single-build image with all 5 required patches (P1–P5, see below) |
| `recipe.yaml` | The validated sparkrun recipe (DFlash2 draft, k=4, `kv_cache_dtype: auto`) |
| `smoke-test.sh` | Post-boot validation script (models list, generation, tool-call, loop checks) |

## Requirements

| | |
|---|---|
| **Hardware** | 8× NVIDIA DGX Spark (GB10, sm_121, 121 GB unified memory per node, 48 SMs) |
| **Networking** | 2× 200 Gb/s ConnectX-7 per node (a single transfer uses one rail → ~200 Gb/s / ~25 GB/s effective) + 1 GbE management |
| **Disk** | ~550 GB free on **every** node (517 GB model + 4.4 GB draft + ~53 GB image + headroom) |
| **Accounts** | HF token for the gated target model; the DFlash2 draft (`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`) is public |
| **Software** | Ubuntu aarch64, Docker 29.x, NVIDIA Container Toolkit, `sparkrun` ≥ 0.4.x |

## Quick start

```bash
# 1. Build the patched vLLM Muse image on the head node:
docker build -t muse-spark-open-dflash2:latest .
#    Build log MUST show: "MoE patch counts: P1=1 P2=1 P3=1 P4=1" and "DFlash patch count: P5=1"

# 2. Download both models on the head node, and rsync the DRAFT to all 8 nodes
#    (sparkrun only distributes the `model:` field — see TUTORIAL §3):
HF_TOKEN=hf_xxx hf download meta-models/Muse-Spark-Open-816B-A42B-NVFP4
hf download nvidia/Muse-Spark-Open-816B-A42B-DFlash2

# 3. Launch (use a fresh recipe file name per run — job ids are deterministic):
cp recipe.yaml ~/muse-816b.yaml && cd ~ && sparkrun run muse-816b.yaml --dry-run
sparkrun run ~/muse-816b.yaml

# 4. When the log reports "OpenAI server is ready to accept requests":
bash smoke-test.sh
```

Full detail — cluster setup, expected boot sequence, performance numbers, tuning knobs,
and the complete symptom → cause → fix troubleshooting table — is in [`TUTORIAL.md`](TUTORIAL.md).

## Why 5 patches?

Upstream vLLM Muse does not cover sm_121 (GB10) in several places. `Dockerfile` applies five
small runtime patches in a single build:

| Patch | Problem it fixes |
|---|---|
| **P1** | NVFP4 MoE validator blocks everything but `FLASHINFER_TRTLLM` / Marlin-on-Hopper → force the (CUTLASS-based, arch-independent) Marlin path |
| **P2** | NVFP4 MoE validator generalized to accept any FLASHINFER flavor |
| **P3** | BF16 (unquantized) MoE enum hardcoded to `FLASHINFER_TRTLLM` while the selector picks `FLASHINFER_CUTLASS` on sm_121 → align |
| **P4** | BF16 MoE hardcodes `TrtLlmBf16Experts` (cubins cover sm100/103/107 only → "no kernels runnable on sm_121") → keep the oracle's `FlashInferExperts` |
| **P5** | DFlash draft `combine_hidden_states`: dummy-run aux hidden states arrive fp32, `fc` is bf16 → cast (dtype mismatch crash) |

Plus `ENTRYPOINT []` (the base image's `ENTRYPOINT ["vllm","serve"]` would swallow sparkrun's
command as flags) and the JIT-storm throttling env block (mandatory — see TUTORIAL §10-b).

## Critical settings — do not "optimize" these away

- **`kv_cache_dtype: auto`** — this NVFP4 export ships **zero** attention scales; fp8 KV runs
  uncalibrated (unity scales) and degrades long generations into repetition.
- **`--attention-backend TRITON_ATTN`** (exact enum name, not `triton`) — flash-attn v2 can't
  replay varlen cudagraphs (`UNIFORM_BATCH`); `TRITON_ATTN` is `ALWAYS`.
- **`VLLM_USE_RUST_FRONTEND: '0'`** — the Rust frontend silently ignores `enable_auto_tool_choice`
  and `structured_outputs_config`; with `'1'` tool-calls don't work.
- **The full `env:` block** — JIT compile storms blow past the 121 GB unified pool and wedge the
  node (D-state worker → earlyoom → machine lockup).
- **DFlash2 draft, k=4** — the `Inferact/Muse-Spark-Open-DSpark` draft produces endless
  self-repeating loops on long contexts; `num_speculative_tokens: 4` is the validated value.

## Measured performance (8× GB10, TP8, DFlash2 k=4)

| Scenario | Result |
|---|---|
| Boot (cold, caches warm) | ~9–11 min |
| KV cache | ~926k tokens (7× concurrency at 131k context) |
| Decode, 8 concurrent requests | 88–97 tok/s aggregate |
| Prefill | ~1900 tok/s short prompts; 55–70 tok/s long/concurrent (governs TTFT) |

## Validated versions

8× DGX Spark GB10 · sparkrun 0.4.1 · vLLM Muse fork `0.1.dev22954+ga34ca64c1` ·
FlashInfer 0.7.0.post1 · driver 580.173.02 · CUDA 13.0 · 2026-10-23