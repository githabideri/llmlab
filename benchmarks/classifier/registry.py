"""classifier/registry.py — the pattern/knowledge table (data, not logic).

Each entry documents which real incident it exists for, so the table stays
honest under review. The failure corpus (fixtures/failures/) is the regression
suite: every entry must classify its fixture to the expected class.

WALL_MIN_GB is the decimal-GB threshold above which a single-device allocation
failure is the *documented* wall shape (Qwen4Exp 125B Q2: a 25.75 GB per-layer
aggregate buffer that cannot fit a 24 GB frame) rather than a generic OOM. It is
data, tuned against the corpus — never an off-the-cuff constant.
"""

WALL_MIN_GB = 24.0   # 24 GB class frames (RTX 3090/4090); the documented wall is >=25.4

CLASSES = (
    "BUILD_DEFECT_ASSERT",   # a toolchain/build assertion fired (root cause class)
    "UNSUPPORTED_ARCH",      # the build lacks the feature for this architecture
    "WRONG_GPU",             # device identity drift — the measurement is unmeasurable
    "WALL_VRAM_FIT",         # the documented wall: allocation above WALL_MIN_GB
    "CUDA_OOM",              # allocation failure below the wall threshold
    "C_GROUP_OOM",           # host/cgroup oom-killer (contained, different cause class)
    "HOST_KERNEL_ERROR",     # Xid/MCE/panic/AER — a host failure, never a cell outcome
    "MODEL_LOAD_FAIL",       # generic load failure, no specific root marker
    "HTTP_EMPTY_200",        # empty-body 200 (vLLM /health contract) hit a body-grepping probe
    "SSE_MALFORMED",         # unparseable stream frame
    "SSE_NO_USAGE",          # stream without a usage object — token closure unavailable
    "TOKEN_MISMATCH",        # client token count diverges from server-side count
    "UNKNOWN",               # nothing matched — this pages the owner, by design
)

# class -> which observed incident(s) forced it into existence (for review)
PROVENANCE = {
    "BUILD_DEFECT_ASSERT": "2026-09-12 dual-3090 B1: meta-backend GGML_ASSERT preceded the OOM",
    "UNSUPPORTED_ARCH": "2026-09-12 B0 control: tensor mode not implemented for an experimental architecture",
    "WRONG_GPU": "2026-09-02 3-GPU stage 1: CUDA FASTEST_FIRST silently ran cells on wrong GPUs",
    "WALL_VRAM_FIT": "2026-09-10 documented wall (25.75 GB single-device buffer on a 24 GB frame)",
    "CUDA_OOM": "boundary cells across 2026-09-02/09-10 ladders",
    "C_GROUP_OOM": "2026-09-10: LXC cgroup oom-killed llama-gguf at 25.6/47.9 GB RSS",
    "HOST_KERNEL_ERROR": "2026-09-08/09-10 unexplained host deaths; Xid/MCE stop-condition gate",
    "MODEL_LOAD_FAIL": "09-12 B0 load-fail chain after the arch rejection",
    "HTTP_EMPTY_200": "2026-09-02 + 09-10: body-grepping /health probes false-negated everything",
    "SSE_MALFORMED": "09-02 stage 1: 'data:' frame parsing bugs in the original client",
    "SSE_NO_USAGE": "09-10: missing stream_options.include_usage left decode rates null",
    "TOKEN_MISMATCH": "09-11: client count vs server-side counter cross-check",
}
