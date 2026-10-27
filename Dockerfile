# =====================================================================
# Muse-Spark-Open-816B-A42B @ 8x DGX Spark (GB10/sm_121) — single-build image
# =====================================================================
# Base: vllm/vllm-openai:muse-spark-open  (Docker Hub — public, arm64+amd64)
#   vLLM Muse fork: vllm 0.1.dev22954+ga34ca64c1, FlashInfer 0.7.0.post1,
#   B12X (sm12x) MoE kernel family included; dflash/dflash2 spec-decode runtime included.
#
# This single Dockerfile reproduces the historical 6-stage chain
# (sparkrun -> marlin -> cutlass -> bf16cutlass -> bf16experts -> dflash2fix)
# as 5 patches applied in one build:
#   P1  NVFP4 MoE: free the Marlin path from the "Hopper only" gate (use_route_major=True)
#   P2  NVFP4 MoE: generalize the validator to accept any FLASHINFER flavor
#   P3  BF16 (Unquantized) MoE: TRT-LLM enum -> FLASHINFER_CUTLASS
#   P4  BF16 MoE: drop the TrtLlmBf16Experts hardcode (no sm121 cubins!)
#   P5  DFlash draft dtype fix: dummy-run aux hidden states are fp32, fc is bf16 -> cast
# Plus ENTRYPOINT [] so sparkrun's appended command runs (double `vllm serve` fix).
#
# Build (head node only — sparkrun distributes the image to all nodes):
#   docker build -t muse-spark-open-dflash2:latest .
# ---------------------------------------------------------------------
FROM vllm/vllm-openai:muse-spark-open

# The base image ships ENTRYPOINT ["vllm","serve"]; sparkrun appends its own command,
# which the entrypoint would swallow as flags -> "unrecognized arguments". Empty it.
ENTRYPOINT []

# (Recommended) conservative defaults from the GB10 lessons learned:
ENV VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0

RUN python3 <<'PY'
import re

# ============================ MoE runner patches (P1-P4) ============================
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/muse_spark/nvidia/muse_moe_runner.py"
src = open(p).read()

# ---------- P1: force the Marlin path on every GPU (Hopper gate bypass) ----------
pattern = re.compile(
    r"self\.use_route_major = \([^)]*?and current_platform\.is_device_capability_family\(90\)[^)]*?\)",
    re.DOTALL,
)
src, n1 = pattern.subn("self.use_route_major = True", src, count=1)

# ---------- P2: open the NVFP4 validator to any FLASHINFER flavor ----------
old2 = """        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
            raise NotImplementedError(
                "MuseSpark NVFP4 MoE requires FLASHINFER_TRTLLM or Marlin "
                f"on Hopper; got {self.nvfp4_backend}."
            )"""
new2 = """        if self.nvfp4_backend not in (NvFp4MoeBackend.FLASHINFER_TRTLLM, NvFp4MoeBackend.FLASHINFER_CUTLASS, NvFp4MoeBackend.FLASHINFER_CUTEDSL, NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED):
            raise NotImplementedError("MuseSpark NVFP4 MoE requires FLASHINFER (any CUTLASS/TRTLLM/CUTEDSL) or Marlin on any arch")"""
n2 = src.count(old2)
src = src.replace(old2, new2)

# ---------- P3: BF16 (Unquantized) MoE enum -> FLASHINFER_CUTLASS ----------
old3 = "self.unquantized_backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM"
new3 = "self.unquantized_backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS"
n3 = src.count(old3)
src = src.replace(old3, new3)

# ---------- P4: drop the TrtLlmBf16Experts hardcode (the sm121 cubin manifest only
#          ships sm100/103/107 -> "cubin manifest contains no kernels runnable on
#          sm121"). Keep the oracle's choice from super().__init__.
old4 = """        self.experts_cls = (
            TrtLlmBf16ExpertsModular if modular else TrtLlmBf16ExpertsMonolithic
        )"""
new4 = """        # sm121 patch: TRT-LLM BF16 cubins target sm100/103/107; keep the oracle's
        # experts_cls selection from super().__init__ (flashinfer_cutlass ->
        # FlashInferExperts).
        pass"""
n4 = src.count(old4)
src = src.replace(old4, new4)
open(p, "w").write(src)
print(f"MoE patch counts: P1={n1} P2={n2} P3={n3} P4={n4}")
if not (n1 == 1 and n2 == 1 and n3 >= 1 and n4 == 1):
    raise SystemExit("PATCH FAILED: unexpected change counts — the base image version may have changed")

# ============================ DFlash draft patch (P5) ============================
q = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py"
qs = open(q).read()
old5 = "        result = self.model.fc(hidden_states)"
new5 = """        # sm121/GB10 patch: dummy-run (profile) aux hidden states may arrive in float32
        # while fc is bf16. Error without this: "expected mat1 and mat2 to have the same
        # dtype, but got: float != BFloat16"
        if hidden_states.dtype != self.model.fc.weight.dtype:
            hidden_states = hidden_states.to(self.model.fc.weight.dtype)
        result = self.model.fc(hidden_states)"""
n5 = qs.count(old5)
if n5 != 1:
    raise SystemExit(f"P5 PATCH FAILED: fc call found {n5} times (expected 1)")
open(q, "w").write(qs.replace(old5, new5))
print(f"DFlash patch count: P5={n5}")

print("Verification:")
for line in open(p).read().splitlines():
    s = line.strip()
    if ("use_route_major = True" in s or "unquantized_backend = UnquantizedMoeBackend" in s
            or "experts_cls = MarlinExperts" in s):
        print(" ", s[:110])
for line in open(q).read().splitlines():
    if "fc.weight.dtype" in line:
        print(" ", line.strip()[:110])
PY