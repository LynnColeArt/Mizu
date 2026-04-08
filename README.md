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
- parked CUDA sessions now offload the resident in-memory context buffer after
  checkpointing, so `resume` is the path that reconstructs active decode state
- CUDA live-context payloads are now versioned, self-describing, and
  checksummed, and decode validates both header and payload integrity before
  consuming restored backend state
- CUDA live-context payloads now also carry the producer artifact identity, so
  decode can reject same-route plan drift and `resume` can reject mismatched
  checkpoint state instead of silently reusing it
- CUDA live-context payloads now use a fixed-size state block with explicit
  decode-state lanes plus a compact summary word, so decode consumes
  structured backend state instead of hashing an opaque byte bag
- those CUDA live-context payloads now expose semantic state as token digest,
  modal digest, packed KV/decode-step counters, and rolling decode state, with
  unit coverage proving decode advances the structured state predictably
- CUDA live-context payloads now widen to a 128-byte windowed state image with
  page-like KV metadata, a recent-token ring, and a state-image digest, so the
  placeholder decode path can evolve against something closer to compact
  backend-owned decode state
- that CUDA live-context image now widens again to 256 bytes and carries
  explicit per-page slot payloads, so decode continuity is represented as a
  small page-backed state image instead of metadata alone
- that CUDA live-context image now widens again to 512 bytes and carries
  compact key and value lane planes plus per-page digests, so page-local decode
  state looks more like a tiny KV-style image than token slots alone
- that same 512-byte CUDA image now also carries per-page tensor-layout records
  for key rows, value rows, lane counts, head blocks, and page generations, so
  decode can preserve untouched page identity while advancing only the page it
  mutates
- that CUDA live-context image now widens again to 640 bytes and carries an
  explicit per-page control table for owner kind, usable capacity, committed
  rows, free rows, epochs, logical page ids, and flags, so decode state now
  looks more like a compact page table than layout metadata alone
- that CUDA live-context image now widens again to 768 bytes and carries an
  explicit per-page tensor descriptor table for storage offsets, committed
  byte spans, capacity byte spans, and row strides, so the compact page image
  now looks more like a tiny tensor-backed page record than a pure summary
- runtime workspace reservations now back a real reusable host scratch buffer,
  and the CUDA bridge receives that buffer during stage execution
- the `Makefile` now rebuilds the contract binaries when the C API Fortran
  sources change, which keeps the public-path tests from silently running stale
  executables
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
