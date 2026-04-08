# Mizu

Mizu is an experimental local inference runtime for multimodal decoder models.

Current reality:

- the runtime contract, cache/reporting machinery, and session lifecycle are
  real
- backend execution is still placeholder/scaffold-level
- `mizu` does not yet run real Qwen/Gemma inference with real packed weights
  and real backend math

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
- the loader now recognizes an optional imported `mizu_import/` bundle with
  validated tensor, modality, and projector inventories
- route-aware optimization and persisted cache metadata are implemented
- runtime backend inventory and capability-probe scaffolding exist for Apple and
  CUDA
- Apple planner scaffolding now exists for ANE and Metal, with route-specific
  plan formats, workspace estimates, and materialized Apple artifact payloads
  behind the current C API metadata path
- Apple now also has a real bridge seam through a compile-safe Objective-C
  implementation on macOS and a non-Apple stub elsewhere, so the public API
  can execute placeholder ANE and Metal projector, prefill, and decode stages
  instead of only emitting Apple planner metadata
- requested backend masks are now intersected with the runtime's detected
  backend inventory at model-open time, so impossible Apple/CUDA routes fail
  early with `MIZU_STATUS_NO_VALID_PLAN` instead of surviving into execution
- Apple session contexts now use the same live-context and park/resume
  checkpoint path as CUDA, including backend-neutral offload and restore rules
  for resident execution state
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
- one narrow multimodal CUDA flow is now validated end to end through the
  public API, including session-state transitions, output readback,
  `park`/`resume`, and fresh-runtime warm reuse against persisted cache state
- imported `mizu_import/` bundle lineage is now retained on the runtime model
  state and emitted into route-specific CUDA and Apple artifact payloads, so
  weight and projector artifacts now carry real imported source-path identity
- imported tensor shapes and dtypes now also produce byte-budget estimates on
  the runtime model state, and those estimates feed backend artifact payloads
  plus weight/projector workspace hints
- CUDA model-load artifacts now go one step further and materialize a narrow
  import-driven weight-pack record with deterministic per-tensor offsets and
  packed-byte totals derived from the imported tensor inventory
- CUDA projector, prefill, and decode artifacts now explicitly depend on that
  packed layout through `pack_ref_*` metadata, and CUDA execution now reads
  that dependency back instead of treating the artifact payload as an opaque
  stage-only blob
- CUDA prefill and decode artifacts now also carry stage-specific
  `pack_use_*` records that name the exact imported tensors selected from the
  packed layout, and CUDA execution now reads those usage summaries back into
  its placeholder execution identity
- those same CUDA stage artifacts now also carry compact numeric
  `pack_dispatch*` records for the first selected packed tensors, and the CUDA
  executor now prefers that compact record before falling back to verbose
  `pack_use*` parsing
- CUDA prefill and decode now also stamp an explicit pack-usage snapshot into
  the live CUDA context payload, so backend-owned session state carries the
  selected imported tensor profile instead of hiding it only inside payload
  strings and cache keys
- that live CUDA context payload now also carries an explicit pack-dispatch
  snapshot for the first selected packed tensors, including packed offsets,
  byte spans, role codes, and layout codes that both bridge variants preserve
  through prefill and decode
- the narrow public CUDA flow now checks stable positive placeholder output plus
  warm-path reproducibility for the same multimodal staged context, while the
  unit suite still pins exact deterministic executor outputs per bridge variant
- runtime workspace reservations now back a real reusable host scratch buffer,
  and the CUDA bridge receives that buffer during stage execution
- the `Makefile` now rebuilds the contract binaries when the C API Fortran
  sources change, which keeps the public-path tests from silently running stale
  executables
- `make test` now succeeds from a clean tree without relying on stray Fortran
  module files and now fails fast if any unit or contract binary fails
- Apple execution now exists as a placeholder bridge/runtime seam rather than a
  real Metal or ANE compute backend, and CUDA execution is still
  placeholder/scaffold-level rather than real transformer math

Build and test:

- `make test`

Documentation:

- [Architecture](./docs/ARCHITECTURE.md)
- [API Spec](./docs/API_SPEC.md)
- [Project Plan](./docs/PROJECT_PLAN.md)
- [Task List](./docs/TASK_LIST.md)
- [Current State](./docs/CURRENT_STATE.md)
- [Importer Layout](./docs/IMPORTER_LAYOUT.md)
- [Placeholder Runtime Status](./docs/PLACEHOLDER_RUNTIME_STATUS.md)
- [Style Guide](./STYLE_GUIDE.md)
