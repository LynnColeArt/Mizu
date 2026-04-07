# Mizu

Mizu is an experimental local inference runtime for multimodal decoder models.

The project is intentionally narrow:

- Apple ANE is a priority target
- CUDA is a first-class peer target
- the public surface stays small through a C ABI
- the control plane is written primarily in Fortran
- the runtime is designed to self-optimize for speed through measured reuse

Current implementation status:

- core Fortran runtime types and lifecycle scaffolding are in place
- the public C header exists in `include/mizu.h`
- model manifests, cache keys, and session flow are implemented as scaffolds
- route-aware optimization and persisted cache metadata are implemented
- runtime backend inventory and capability-probe scaffolding exist for Apple and
  CUDA
- CUDA planner scaffolding now emits route-specific plan and weight-pack records
  and materializes stub payload files under `cache_root`
- CUDA capability probing now prefers a real device bridge when available and
  falls back cleanly when it is not
- CUDA projector, prefill, and decode now run through a backend-owned CUDA
  bridge, with placeholder kernels on NVIDIA hardware and a CPU stub fallback
  when `nvcc` is unavailable
- session staging now preserves attached token content and copied modal bytes,
  along with stable content hashes that feed the current CUDA placeholder path
- live session context identity now survives prefill and advances on decode, so
  repeated CUDA decode steps depend on prior session work instead of counts
  alone
- CUDA prefill now pushes real staged token and modal buffers through the
  backend bridge instead of relying only on staged counts and hashes
- CUDA prefill now emits a persisted backend-owned live-context byte buffer
  into session state, and CUDA decode consumes and updates that buffer across
  steps
- CUDA `park` now materializes a small session-checkpoint artifact when
  `cache_root` is set, and `resume` reloads that checkpoint through the
  runtime cache layer
- runtime workspace reservations now back a real reusable host scratch buffer,
  and the CUDA bridge receives that buffer during stage execution
- `make test` now succeeds from a clean tree without relying on stray Fortran
  module files
- Apple and CUDA execution backends are not implemented yet

Build and test:

- `make test`

Documentation:

- [Architecture](./docs/ARCHITECTURE.md)
- [API Spec](./docs/API_SPEC.md)
- [Project Plan](./docs/PROJECT_PLAN.md)
- [Task List](./docs/TASK_LIST.md)
- [Current State](./docs/CURRENT_STATE.md)
- [Style Guide](./STYLE_GUIDE.md)
