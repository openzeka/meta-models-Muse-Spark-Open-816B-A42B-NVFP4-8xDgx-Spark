# Muse-Spark-Open-816B-A42B — Runbook for 8× DGX Spark

> **TL;DR:** Serve the 816B MoE model (NVFP4 W4A4, ~533 GB of weights) on **8× DGX Spark (GB10)**
> as a **TP8 multi-node** deployment with **DFlash 2 speculative decoding**. Verified boot:
> **~9–11 minutes**; **~90 tok/s aggregate** at 8 concurrent requests. This configuration runs
> stably for hours (validated 2026-10-23).
>
> This guide is the distilled outcome of 6 failed attempts over 23 working phases.
> Nearly every failure you may hit is listed with its fix in §9.

---

## 0. Requirements

| | |
|---|---|
| **Hardware** | 8× NVIDIA DGX Spark (GB10, sm_121, 121 GB unified memory per node, 48 SMs) |
| **Networking** | **2× 200 Gb/s ConnectX-7** (rails — `ethtool` shows `Speed: 200000Mb/s`). A single transfer uses one rail → effective speed **200 Gb/s (~25 GB/s)**; the 2nd rail serves NCCL/a second stream. Plus 1 GbE management |
| **Disk** | ⚠️ **~550 GB free on EVERY node** (517 GB model + 4.4 GB draft + ~53 GB image + headroom). Tight disk → page-cache pressure → crashes |
| **Accounts** | HF token — only the **target model is gated**: `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`. The draft (`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`) is **public** |
| **Software** | Ubuntu aarch64, Docker 29.x, NVIDIA Container Toolkit, `sparkrun` ≥ 0.4.x |
| **Internet** | For `hf download` + `docker pull` during initial setup |

Files in this directory:

| File | What |
|---|---|
| `README.md` | quick overview, requirements and quick start |
| `TUTORIAL.md` | this guide |
| `Dockerfile` | single-build image (all 5 patches at once) |
| `recipe.yaml` | validated, stable sparkrun recipe (DFlash2, k=4) |
| `smoke-test.sh` | ready-made validation script |

---

## 1. sparkrun + 8-node cluster

```bash
# on every node:
uvx sparkrun setup install
sparkrun setup check          # docker/toolkit/CDI/ssh-mesh/CX7/earlyoom must all be green
```

```bash
# on the head node (first host = head):
sparkrun cluster create spark-lab-8 --hosts <head-ip>,<w1-ip>,<w2-ip>,<w3-ip>,<w4-ip>,<w5-ip>,<w6-ip>,<w7-ip>
sparkrun cluster set-default spark-lab-8
```

> Don't proceed until `sparkrun setup check` is fully green; bootstrap details:
> <https://sparkrun.dev/>

---

## 2. Build the image (head node only — sparkrun distributes it)

`Dockerfile` applies all 5 patches in one build (§10):

```bash
docker build -f Dockerfile -t muse-spark-open-dflash2:latest .
```

The build log must print **`MoE patch counts: P1=1 P2=1 P3=1 P4=1`** and
**`DFlash patch count: P5=1`** (any other numbers mean the base image has changed — see §9).
Takes ~2–3 min.

---

## 3. Place the models — ⚠️ on ALL 8 NODES!

```bash
# on the head node:
HF_TOKEN=hf_xxx hf download meta-models/Muse-Spark-Open-816B-A42B-NVFP4        # ~517 GB (gated)
hf download nvidia/Muse-Spark-Open-816B-A42B-DFlash2                           # ~4.4 GB (public)

# rsync the DRAFT model to the 7 workers manually (sparkrun only distributes the `model:`
# field of the recipe — models inside speculative-config are NOT distributed. Skip this and
# boot silently stalls for 10 minutes: 'Timed out after 601 seconds ... 1/8 clients joined'):
for ip in <w1> <w2> <w3> <w4> <w5> <w6> <w7>; do
  rsync -a --info=progress2 \
    ~/.cache/huggingface/hub/models--nvidia--Muse-Spark-Open-816B-A42B-DFlash2/ \
    nvidia@192.168.1.$ip:.cache/huggingface/hub/models--nvidia--Muse-Spark-Open-816B-A42B-DFlash2/
done

# verify (expect 2 on all 8 = config.json + model.safetensors):
for ip in <head> <w1> <w2> <w3> <w4> <w5> <w6> <w7>; do
  echo -n "$ip: "
  ssh nvidia@192.168.1.$ip 'find ~/.cache/huggingface/hub/models--nvidia--Muse-Spark-Open-816B-A42B-DFlash2/snapshots -maxdepth 2 \( -name config.json -o -name model.safetensors \) | wc -l'
done
```

Notes:
- **Better path (documented at <https://sparkrun.dev/runtimes/vllm/>, verification pending in
  this setup):** declare the draft as a structured `defaults.speculative_config` mapping
  (with its `model` field) — sparkrun pre-syncs that draft to all hosts automatically during
  the distribution phase, and `-o speculative_config.num_speculative_tokens=N` /
  `-o speculative_config.model=...` tune it per launch. The manual rsync below was required
  only because our recipe passed the spec config as a raw `--speculative-config` string inside
  `command:`, which sparkrun treats as opaque shell text.
- `rsync -a` — **NOT `-L`** (HF cache blobs are symlinks; `-L` would inflate every copy ~2×).
- The main model (~517 GB) is distributed by sparkrun to all 8 nodes automatically. There is
  massive network headroom (2×200 Gb/s/node) but the **end-to-end wall time is process-bound,
  not wire-bound**: the OCI relay's compressed import + container import/decompress chain —
  measured full fan-out (517 GB × 7 workers) is ~27 min (~2.2 GB/s aggregate while the links
  sit mostly idle). The cache is persistent afterwards — one-time cost.
- Disk-saving alternative: NFS export from the head + worker mounts + an **absolute model
  path** in the recipe (`model:` as `/cache/.../snapshots/<rev>`) — skips sparkrun's
  distribution entirely; a single copy is enough.

---

## 4. Recipe

Copy to the head: `scp recipe.yaml nvidia@<head>:~/muse-816b.yaml`

Essentials (full file in `recipe.yaml` — recipe_version 2 schema):

```yaml
recipe_version: "2"
model: meta-models/Muse-Spark-Open-816B-A42B-NVFP4     # HF ID = sparkrun distributes it
runtime: vllm-distributed
container: muse-spark-open-dflash2:latest
min_nodes: 8
max_nodes: 8
metadata:
  model_params: 816B
  model_dtype: nvfp4
  kv_dtype: auto
  model_vram: 648        # real footprint is 8×81 GiB; corrects sparkrun's underestimate
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 8
  pipeline_parallel: 1
  block_size: 16
  max_model_len: 131000
  max_num_seqs: 8
  max_num_batched_tokens: 8192
  kv_cache_dtype: auto   # CRITICAL: do NOT use fp8 (§10-a)
  gpu_memory_utilization: 0.85
  load_format: auto
env:                     # THE WHOLE SET! don't drop a single line (§10-b)
  # DeepGEMM is broken for fp8 GEMM on SM12x — keep it off.
  VLLM_USE_DEEP_GEMM: '0'
  VLLM_MOE_USE_DEEP_GEMM: '0'
  VLLM_USE_V2_MODEL_RUNNER: '1'
  VLLM_WORKER_MULTIPROC_METHOD: spawn
  CUTE_DSL_ARCH: sm_121a
  # '0' = Python API server: the Rust frontend silently ignores 'enable_auto_tool_choice'
  # and 'structured_outputs_config', so tool-calls/structured outputs wouldn't work.
  # Set '1' only for perf experiments where you don't need those.
  VLLM_USE_RUST_FRONTEND: '0'

  # === JIT-storm throttling (MANDATORY — see §10-b) ===
  MAX_JOBS: '2'
  FLASHINFER_NVCC_THREADS: '1'
  OMP_NUM_THREADS: '1'
  TORCH_CUDA_ARCH_LIST: 12.1a
  FLASHINFER_CUDA_ARCH_LIST: 12.1a
  FLASHINFER_DISABLE_VERSION_CHECK: '1'
  VLLM_DEEP_GEMM_WARMUP: 'skip'
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  VLLM_KZ_TRIM_AFTER_LOAD: '1'
  VLLM_RPC_TIMEOUT: '1800000'
  VLLM_ENGINE_READY_TIMEOUT_S: '3600'
  # NCCL CUMEM/buffers also eat the unified pool — cap them.
  NCCL_CUMEM_ENABLE: '0'
  NCCL_BUFFSIZE: '1048576'
  NCCL_MAX_NCHANNELS: '4'
  NCCL_MIN_NCHANNELS: '4'
  NCCL_IGNORE_CPU_AFFINITY: '1'
  TORCH_NCCL_ASYNC_ERROR_HANDLING: '1'
command: |
  vllm serve {model} \
    --host {host} --port {port} \
    --tensor-parallel-size {tensor_parallel} \
    --pipeline-parallel-size {pipeline_parallel} \
    --max-model-len {max_model_len} \
    --max-num-seqs {max_num_seqs} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --kv-cache-dtype {kv_cache_dtype} \
    --attention-backend TRITON_ATTN \
    --block-size {block_size} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --load-format {load_format} \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser muse \
    --tool-call-parser muse \
    --enable-auto-tool-choice \
    --served-model-name {model} \
    --speculative-config '{"method":"dflash","model":"nvidia/Muse-Spark-Open-816B-A42B-DFlash2","num_speculative_tokens":4}' \
    --moe-backend flashinfer_cutlass
```

For a plain **spec-decode-free** start, delete the `--speculative-config ...` line and skip the
draft steps in §3. Expected plain speed: ~10–15 tok/s/request.

> ⚠️ **Do not use DSpark** (`Inferact/Muse-Spark-Open-DSpark` + `method: dspark`): it was
> observed to produce self-repeating outputs (**endless loops**) on long contexts (§11-a).
> The same cases completed cleanly with DFlash2.

---

## 5. Run

```bash
cd ~ && sparkrun run muse-816b.yaml --dry-run   # plan + VRAM check (relative path! §9-e)
sparkrun run ~/muse-816b.yaml                  # real run
sparkrun logs -f <job-id>                      # Ctrl+C only detaches logs; the job KEEPS RUNNING
```

**Expected boot sequence:**

1. `world_size=8 ... master_addr=<head>` → rendezvous done (otherwise §9-c)
2. `Resolved architecture: MuseSparkForConditionalGeneration` + **`Resolved architecture: DFlash2DraftModel`**
3. `Loading safetensors checkpoint shards: ... 127/127` (~5 min), then the draft (~12 s)
4. `Model loading took ~83 GiB memory` → footprint sanity check (§8)
5. `init engine (profile, create kv cache, warmup model) took ~110–120 s`
6. `GPU KV cache size: ~926,000 tokens`
7. `OpenAI server is ready to accept requests` ✅

If your session detaches: `sparkrun status` → ready-to-use `logs`/`stop` lines.
Do NOT use `pkill -f "sparkrun run"` (it kills your own shell too).

---

## 6. Validate

```bash
bash smoke-test.sh          # customize with M=http://<head>:8000/v1/chat/completions
```

Checklist:
- [ ] Answers arrive; `finish_reason: stop`
- [ ] The `reasoning` field comes back separately (**note**: `content` may be late/empty — §11-b)
- [ ] `SpecDecoding metrics` flows in the log; the acceptance curve must be **decaying**
      (e.g. `0.71, 0.54, 0.39, 0.28, ...`). A steady `1.000, 1.000, 1.000...` = **loop** signature!
- [ ] `journalctl -u earlyoom` is silent (no SIGTERM)

---

## 7. Measured performance (8× GB10, TP8, DFlash2 k=4)

| Scenario | Result |
|---|---|
| Boot (cold, caches warm) | ~9–11 min (weights 5 + init/profile ~2 + capture ~1) |
| KV cache | **~926k tokens** — 7× concurrency at 131k context |
| Decode, 8 concurrent requests | **88–97 tok/s aggregate** (~11 tok/s/req) |
| Spec acceptance curve (slots 1→8, measured at k=8) | 0.70, 0.51, 0.39, 0.29, 0.23, 0.17, 0.13, 0.10 |
| Prefill | ~1900 tok/s (short); 55–70 tok/s (long/concurrent — this governs TTFT) |

---

## 8. Tuning

| Knob | When / how |
|---|---|
| **`num_speculative_tokens` (k)** | **4 is the validated value.** Look at the slot acceptance curve (§7): slots 5–8 verify at only ~10–20% but **every slot costs a full target verification**. k=8→4 raised efficiency (accepted/verified) 0.39 → 0.58. k=5 is a middle ground; check the `Per-position acceptance rate` in your logs against your own workload |
| **`gpu_memory_utilization`** | ⚠️ On GB10 GPU+host share **one 121 GB pool**. 0.85 is at the limit (weights ~81 + peak activation ~15 + graphs ~4 ≈ 100 of 103 GiB). If earlyoom SIGTERMs you, drop to **0.78** |
| **`max_num_batched_tokens`** | 8192 (raised from 4096 to silence the spec-slot warning). If TTFT is poor, try 16384 (it competes with decode under concurrency — test as a single change) |
| **`max_num_seqs`** | 8 → 16/32 for aggregate throughput at the cost of per-stream latency |
| **Sampling** | If you see loops/repetition: request-side `temperature 0.9` + `repetition_penalty 1.1` + `frequency_penalty 0.5` (§9-f) |

---

## 9. Troubleshooting — symptom → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: MuseSpark NVFP4 MoE requires FLASHINFER_TRTLLM or Marlin on Hopper` | patches P1/P2 missing | build with `Dockerfile`; verify patch counts |
| `TrtllmGenBatchedGemmRunner: cubin manifest contains no kernels runnable on sm121` | patches P3/P4 missing | same |
| `expected mat1 and mat2 to have the same dtype, but got: float != BFloat16` (during profile, `qwen3_dflash.py combine_hidden_states`) | **patch P5 missing** — dummy-run aux hidden states are fp32, the DFlash `fc` is bf16 | same (P5 adds the cast) |
| `SpeculativeConfig ValidationError: Invalid repository ID ... DFlash2` | draft not cached + `HF_HUB_OFFLINE` | §3 — `hf download` on the head |
| 10 min of silence then `Timed out ... 1/8 clients joined` | draft on the **head only** — workers die silently | §3 — `rsync -a` to all 8 + verify `config.json/model.safetensors` (check worker `docker logs` for their error) |
| `ValueError: ... 'marlin' is not supported for unquantized MoE` | a **global** `--moe-backend marlin` style override | never set a global `--moe-backend` (stay with `flashinfer_cutlass`) |
| `Adaptive verification replays decode cudagraphs ... FlashAttentionBackend allows none` | `kv_cache_dtype: auto` + dspark-adaptive; flash-attn **v2** = `UNIFORM_BATCH` | `--attention-backend TRITON_ATTN` (**exact enum name**; not `triton`) |
| `Unknown attention backend: 'triton'` | enums take their full name | `TRITON_ATTN` |
| `WorkerProc initialization failed` + `NV_ERR_NO_MEMORY` / earlyoom `sending SIGTERM` | **JIT compile storm** = memory massacre (20+ nvcc/Triton procs + 18 GB Shmem) | the `env:` block with `MAX_JOBS: '2'` + `FLASHINFER_NVCC_THREADS: '1'` is **mandatory**. First 5 min: `ps -eo comm \| grep -cE "nvcc\|cc1plus\|triton"` ≤ 3, `Shmem` < 2 GB, `MemAvailable` > 10 GB |
| Workers die by **hanging** (`kill failed: Timer expired`), machine unresponsive for minutes | earlyoom `-s 50` → the kill never fires and paralysis grows | keep earlyoom at `-m 2 -s 80` (a clean kill beats paralysis) |
| `ResourceBusy` / slot locked | sparkrun job ids are deterministic (derived from the file name) | every run: **a new file name** (`cp a.yaml b.yaml`). Never `pkill -f "sparkrun run"` — use `sparkrun stop <id>` |
| `sparkrun run --dry-run` → `pathlib ... Non-relative patterns are unsupported` | sparkrun 0.4.x dry-run + absolute-path bug | use a relative/plain name: `cd ~ && sparkrun run x.yaml --dry-run` |
| `Container image stale on 7 of 8 host(s), syncing` on **every** run | `container:` uses a mutable `:latest` tag (sparkrun can't trust content identity — docs: "use a digest when content identity must be immutable") + the in-sync marker lives in container labels and is wiped by each run's "Step 1: Cleaning up" | Harmless: `0 B transferred; layers reused`, ~40 s of verification. To get rid of it, pin the image: `container: <image>:2026-10-23` or `container: <image>@sha256:...` |
| Draft model not auto-distributed even though it sits in `--speculative-config` | sparkrun only pre-syncs drafts declared as structured `defaults.speculative_config` (with a `model` field); a raw `--speculative-config '{...}'` string inside `command:` is opaque shell text | move the spec config into `defaults.speculative_config:` (see §3 notes) or rsync manually |
| **Output falls into an endless self-repeating loop** (especially long contexts) | **the DSpark draft** (§11-a) — resolved by DFlash2. If it persists, tune sampling (§8) | use the DFlash2 `--speculative-config` line; request-side `repetition_penalty 1.1, frequency_penalty 0.5` if needed |
| Responses "hang" with an empty stream for minutes | (a) Muse fills the `reasoning` field while `content` stays empty — the client isn't rendering `reasoning`; (b) long-prompt prefill (TTFT) | (a) render `reasoning` client-side and set a generous `max_tokens` (thinking consumes budget!); (b) wait / §8 |
| Long answers write words with numbers attached (`yosunlar38 minik39 ...`) | asking for a word count ("write 350 words") makes Muse count words inside `reasoning` | don't set word-count targets (model behavior — not a loop) |
| Warning: `Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention` | `kv_cache_dtype: fp8` + checkpoint ships no calibration scales (this NVFP4 export has 360 MoE scales but **zero** attention scales) | **`kv_cache_dtype: auto`** — fp8 degrades long generations into repetition |
| `TP MoE finalize requires SM100/SM103/SM107 ... got SM121` | finalize fallback | non-fatal, normal |
| `Draft model ... does not support external multimodal embeddings ... using text-only draft inputs` | known limitation (§11-c) | non-fatal; the draft helps less on image/video prompts |
| `enable_auto_tool_choice` / structured outputs ignored | the Rust frontend (vllm-rs) **silently ignores** those arguments | `VLLM_USE_RUST_FRONTEND: '0'` (Python API — tool-call works) |

**Generic triage sweep (whenever something feels off):**
```bash
# 1) memory/JIT watch (on a worker):
watch -n5 'grep -E "MemAvailable|Shmem|SwapFree" /proc/meminfo; ps -eo comm | grep -cE "nvcc|cc1plus|triton"'
# 2) earlyoom:
journalctl -u earlyoom --since "-30 min" | grep -i "sending"
# 3) worker logs (sparkrun logs only streams node_0):
ssh nvidia@<worker> 'docker logs --tail 60 $(docker ps -qf ancestor=<image> | head -1)'
```

---

## 10. What's the magic? (why 5 patches)

GB10 = **sm_121**; this arch is still young and upstream vLLM Muse code doesn't cover it in
several places. Hence 5 patches:

- **P1/P2 (NVFP4 MoE):** the validator accepted only `FLASHINFER_TRTLLM` (sm10x) or
  `MARLIN`-on-Hopper (sm90). Marlin is actually CUTLASS-based and arch-independent → drop the
  Hopper gate (P1) + generalize the validator to any FLASHINFER flavor (P2). Result:
  `MarlinExperts`.
- **P3 (BF16 MoE):** the selector picks `FLASHINFER_CUTLASS` on sm121 but the class hardcodes
  its enum to `FLASHINFER_TRTLLM` → align them.
- **P4 (BF16 MoE):** the class overrides the oracle's choice with `TrtLlmBf16Experts` in all
  cases; TRT-LLM BF16 cubins cover sm100/103/107 — sm121 is **not** in the manifest → remove
  the hardcode, keep the oracle's `FlashInferExperts`.
- **P5 (DFlash draft):** in `combine_hidden_states` the `self.model.fc` projection is bf16 but
  dummy-run (profile) aux hidden states arrive fp32 → `mat1/mat2 dtype` error. Cast the input
  to the fc weight dtype (also insurance for real inference).
- **`ENTRYPOINT []`:** the base image ships `ENTRYPOINT ["vllm","serve"]` and sparkrun appends
  its command → double `vllm serve` / `unrecognized arguments`.

**Why `kv_cache_dtype: auto` (a):** this NVFP4 export ships 360 `scale` keys — **all of them**
MoE expert weight scales; `k_scale/v_scale/q_scale/prob_scale` are **absent**. vLLM substitutes
unity (1.0) and continues on a warning — so fp8 KV runs *uncalibrated* and degrades long
generation into repetition (observed). auto = bf16 KV = safe.

**Why the env block (b):** prebuilt kernels for sm_121 are limited; first use triggers JIT.
20+ concurrent `nvcc`/`cc1plus` + ~18 GB of `/dev/shm` + 66 GB of empty model params **blow
past** the 121 GB unified pool; a worker wedged in D-state drags the machine down. `MAX_JOBS=2`,
`FLASHINFER_NVCC_THREADS=1`, `TORCH_CUDA_ARCH_LIST=12.1a` & co. shut the storm down (measured:
Shmem 18 GB → <1 GB).

**Why `TRITON_ATTN` (c):** varlen-q replays (like DSpark-adaptive) need per-request different
query lengths. flash-attn **v2** can't (`UNIFORM_BATCH`; only v3 is `ALWAYS`);
`TRITON_ATTN` is `ALWAYS`. (Kept as a safety line even though DFlash2 k=4 is tolerant.)

**Why the DFlash 2 draft (d):** NVIDIA's block-diffusion draft trained for this exact target.
No looping even on long contexts (below); supports block-8; its SWA-2048 window makes it
almost KV-free → ~2× KV capacity and faster boot vs DSpark.

**Spec-decode correctness note:** correct rejection sampling is *lossless* — a draft cannot
force wrong output; the target rejects it. But **inside a loop the target itself repeats**, so
the acceptance ratio is ~1.0 and rejection never fires; that's why acceptance logs show
`1.000×N` during loops. A log alone can't tell "draft vs target" — you need an A/B (§11-a).

---

## 11. Known limitations / behaviors

1. **The DSpark draft (`Inferact/Muse-Spark-Open-DSpark`) loops on long contexts.** Case: with
   ~16.7k tokens of conversation context, a follow-up question produced endlessly repeating
   text (log signature: acceptance pinned at 100% for minutes). The same cases passed cleanly
   with `nvidia/Muse-Spark-Open-816B-A42B-DFlash2`; switching fp8→bf16 KV alone did **not**
   fix it (so the draft behavior is the primary factor). If you insist on DSpark, use
   request-side anti-repetition guards.
2. **The `reasoning` field eats `max_tokens`** (`max_tokens: 400` → 397 went to reasoning,
   `content: None, finish_reason: length`). A client that doesn't render `reasoning` shows an
   empty/stuck response. Set a generous `max_tokens` for long generations.
3. **The draft is text-only by construction — always** (it never receives the target's
   image/video/audio embeddings; logged once at load as "does not support external multimodal
   embeddings ... using text-only draft inputs"). The **cost only shows on multimodal
   requests** (image/video/audio prompts): the draft predicts blind around those tokens, so
   acceptance/speedup drops there. Pure-text requests are completely unaffected.
4. **No fp8 attention scales** exist in this checkpoint — avoid `auto`-less fp8 KV. bf16 KV
   costs ~40% KV capacity, but with the DFlash2 draft 926k tokens is plenty.
5. **GMU 0.85 is at the edge** — drop to 0.78 on constrained systems / long contexts.
6. **The acceptance-rate trap:** a sustained >95% acceptance is not speed, it's a **loop**;
   measure performance on loop-free windows.
7. Mild pattern repetition inside reasoning monologues (`We should ... We should ...`) is
   normal — it doesn't leak into the content output.

---

*Validated on 8× DGX Spark GB10 on 2026-10-23 (sparkrun 0.4.1, vLLM Muse fork
`0.1.dev22954+ga34ca64c1`, FlashInfer 0.7.0.post1, driver 580.173.02, DFlash2 draft k=4).
A full chronological change log of the 23 phases (failures, root causes, fixes) is
maintained separately in the project notes.*