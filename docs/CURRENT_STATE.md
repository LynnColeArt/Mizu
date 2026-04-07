# Mizu Current State

Last updated: 2026-04-07

## Latest Checkpoint

- latest published baseline before this slice: `39e4374` (`Add page-backed
  CUDA decode state slots`)
- current milestone: CUDA live-context payloads now widen into a compact
  512-byte key/value lane image with per-page digests, so decode state carries
  more backend-shaped page content instead of token slots alone
- immediate next target: replace the compact CUDA key/value lane image with a
  more realistic tensor-backed page record or backend-native KV-state payload

## Roadmap Status

- repository bootstrap and core contracts are effectively complete
- the Fortran control plane is well past scaffolding:
  - runtime, model, and session lifecycle exist
  - park and resume are wired
  - workspace reuse exists
  - optimization evidence and cache identity are persisted
- CUDA is the most advanced backend:
  - capability probing exists
  - planner and bridge seams exist
  - projector, prefill, and decode all execute through placeholder CUDA paths
  - backend-owned session state survives prefill, decode, park, and resume
- the cache and self-optimization layers have real shape, but backend-native
  weight and plan caches are still ahead of us
- Apple is still mostly at the scaffolding stage and remains the biggest
  hardware-validation gap
- model import and target-asset mapping are still only partially done

In short:

- the control plane and runtime contract are in good shape
- the CUDA backend is credible as a bring-up path
- real inference math, real packed weights, and Apple execution are still major
  milestones ahead

## What Exists

### Core Contracts

- the project has a Fortran-first source tree under `src/`
- the public C ABI exists in `include/mizu.h`
- runtime, model, and session handles are implemented as opaque C-facing boxes
- status codes, type enums, and execution report records are defined in source

### Runtime Skeleton

- runtime create and destroy are wired through the C bridge
- model open and close are wired through the C bridge
- session open, close, park, resume, prefill, decode, and output read are wired
- manifest loading and validation are implemented
- target fallback manifests exist for the current Qwen and Gemma targets
- runtime create now records a detected backend inventory for Apple and CUDA
- session staging now retains attached token values and copied modal-byte inputs
  long enough for stage execution, along with stable content hashes
- live session context identity now survives prefill and advances after decode,
  so later decode steps can depend on prior staged and emitted content
- live sessions now also retain a persisted backend-owned context byte buffer
  for the active route, starting with CUDA
- `park` and `resume` now materialize and reload a small session-checkpoint
  payload when `cache_root` is configured and route-specific context state
  exists
- parked CUDA sessions now offload that resident context buffer after the
  checkpoint is safely materialized, and `resume` restores it before the
  session becomes live again
- CUDA live-context payloads now include a small versioned, checksummed
  header, and both `resume` and CUDA decode validate header plus payload
  integrity before treating restored bytes as usable backend state
- CUDA live-context payloads now also retain producer-stage and producer-plan
  lineage, so decode rejects incompatible plan drift and resume rejects
  mismatched checkpoint reloads for offloaded CUDA sessions
- CUDA live-context payloads now use a fixed-size state block with explicit
  state lanes and a compact summary word, so decode consumes structured
  backend state instead of folding arbitrary context bytes into one seed
- those structured CUDA state lanes now have stable semantics: token digest,
  modal digest, packed KV/decode-step counters, and a rolling decode-state
  word
- the runtime can now read that semantic CUDA state back through a Fortran-side
  extractor, which keeps tests and future planner logic from treating the
  payload as an opaque blob
- the CUDA live-context payload now reserves 128 bytes instead of 64 and uses
  the extra space for a small windowed state image
- that windowed state image carries:
  - page-like KV metadata for a few compact logical decode pages
  - a recent-token ring
  - a state-image digest over the compact window
- the CUDA live-context payload now reserves 256 bytes instead of 128 so it
  can carry compact per-page slot contents in addition to page metadata
- those slot payloads now give each tracked decode page a small explicit token
  image, which makes the placeholder runtime state look more like compact
  backend decode state than a metadata-only summary
- the CUDA live-context payload now reserves 512 bytes instead of 256 so it
  can widen those page images into compact key and value lane planes, along
  with per-page lane digests
- the runtime can now read those compact KV-style lane planes back through a
  Fortran-side extractor, which makes the widened decode image inspectable in
  tests instead of opaque

### Self-Optimization

- stage selection uses route-neutral optimization identities
- ANE, Metal, and CUDA candidates can be explored under one shared workload key
- exploration is bounded by `exploration_budget`
- repeated work can reuse the measured winner
- optimization evidence is persisted to disk through `optimization_store_v1.txt`

### Build and Backend Scaffolding

- a top-level `Makefile` now builds and runs the current test set through
  `make test`
- backend scaffolding now exists under:
  - `src/backends/apple/`
  - `src/backends/cuda/`
- initial capability probes exist for:
  - Apple Metal
  - Apple ANE via explicit override
  - CUDA via a real CUDA device bridge, with `nvidia-smi` and override fallback
- CUDA planner scaffolding exists for:
  - model load weight-pack records
  - projector plan records
  - prefill plan records
  - decode plan records
- CUDA-selected artifacts can now materialize stub payload files under the
  configured `cache_root`
- CUDA-selected projector, prefill, and decode stages now execute through a
  backend-owned CUDA bridge that launches minimal real kernels on NVIDIA
  hardware
- CUDA placeholder projector and prefill execution now incorporate staged-input
  content hashes instead of relying on counts alone
- CUDA prefill now also receives copied staged token buffers and modal byte
  buffers through the backend bridge, so placeholder execution can depend on
  actual staged tensor content
- CUDA prefill now emits a persisted live-context byte buffer into session
  state
- CUDA placeholder decode execution now consumes that persisted byte buffer
  and updates it across steps instead of depending on a tiny surrogate record
- CUDA decode now advances explicit KV-token and decode-step counters inside
  that persisted context payload, and the compact summary word retains the
  last emitted token plus stop reason
- CUDA decode now also advances the page-like KV window and recent-token ring
  inside the widened state image, so the placeholder backend state has a
  compact but more realistic notion of decode continuity
- CUDA decode now also appends emitted tokens into explicit per-page slot
  payloads inside that state image, so page continuity is represented through
  actual compact payload contents instead of only page fill counters
- CUDA decode now also writes compact value-lane payloads and stable per-page
  lane digests, so unchanged pages retain their own identity while the overall
  state image digest still advances across decode
- runtime workspace reservations now allocate a reusable host scratch buffer
  instead of tracking bytes alone
- CUDA projector, prefill, and decode now receive that runtime workspace buffer
  through the backend bridge and stamp stage-local scratch data into it
- the build now falls back to a CPU CUDA bridge stub when `nvcc` is not present,
  so non-CUDA environments can still build and run the current tests
- `make test` now passes from a clean tree without requiring stray root-level
  `.mod` files from earlier compiler runs
- the contract test binaries now depend on the C API source list in the
  `Makefile`, so `mod_c_api.f90` edits do not leave stale public-path test
  executables behind

### Cache and Artifact Identity

- deterministic cache keys exist for:
  - weights
  - plans
  - sessions
  - multimodal projector inputs
- runtime-scoped in-memory cache presence tracking exists
- persisted artifact presence tracking exists through `artifact_cache_v1.txt`
- persisted artifact records now include backend-specific metadata:
  - backend family
  - execution route
  - stage kind
  - materialization flag
  - payload byte count
  - planned workspace byte count
  - artifact format label
  - payload fingerprint
  - future payload path
- session-store metadata now persists alongside weight, plan, and multimodal
  metadata, so parked-session checkpoint artifacts can be reloaded across
  runtime create/destroy boundaries

### Tests That Pass

- `test_model_manifest_loader`
- `test_cache_keys`
- `test_cache_store`
- `test_optimization_store`
- `test_stage_reports`
- `test_backend_registry`
- `test_runtime_workspace`
- `test_session_staging`
- `test_cuda_planner`
- `test_cuda_executor`
- `test_cuda_artifacts`

## What Is Still Stubbed

- no real Apple backend exists yet
- no real ANE planner or executor exists yet
- no real Metal executor exists yet
- CUDA planner records are still scaffold-level and do not launch kernels
- model load does not build actual packed weights
- plan selection does not materialize backend-native plan payloads
- CUDA projector, prefill, and decode use real but placeholder kernels; they do
  not execute transformer math yet
- decode still does not consume a persisted tensor/KV buffer from prior
  execution state; it now uses a persisted backend-owned context byte buffer
  plus placeholder kernel logic
- Go bindings do not exist yet

## Important Honesty Notes

- persisted artifact metadata is real metadata, not fake cache hits
- route-specific artifact descriptors are persisted even when payload bytes are
  `0`
- `payload_path` currently points at the future artifact location that a real
  backend-owned pack or plan record would use
- `is_materialized = .false.` means the runtime knows the artifact identity and
  route, but does not yet claim to have built the payload
- CUDA-selected artifacts are the first exception: they now write stub planner
  payload files to the persisted artifact location and mark those records as
  materialized
- parked CUDA sessions are now the second exception: they write a small
  checkpoint payload keyed by the live context hash and configuration so
  `resume` can reload backend-owned session state through the runtime cache
- once that checkpoint is materialized, the parked session drops its resident
  CUDA context bytes in memory and relies on `resume` to reconstruct them
- CUDA projector, prefill, and decode now consume those materialized payloads
  through a backend-owned CUDA bridge and launch tiny placeholder kernels to
  prove the runtime-to-device seam
- persisted artifact metadata now carries planned workspace bytes from the
  stage planner, and the runtime now keeps a reusable high-water-mark workspace
  arena around stage execution, backed by a reusable host scratch buffer
- staged token and modal content are now preserved inside session state, but
  only projector, prefill, and the derived live decode context currently fold
  that content identity into the CUDA placeholder execution path
- CUDA prefill now reads real staged token and modal buffers through the bridge,
  but it still uses them for placeholder seed generation rather than real
  transformer activations
- the persisted CUDA context buffer is still a fixed-capacity backend-owned
  surrogate, not a real KV-cache or transformer activation buffer
- the parked-session checkpoint currently persists that same surrogate buffer,
  not a full backend KV-cache image
- decode now explicitly requires a resident CUDA live-context buffer for CUDA
  sessions; parked/offloaded sessions have to come back through `resume`
- CUDA live-context bytes are no longer just unstructured payloads; they now
  carry format/version/kind markers so stale or corrupted context can fail fast
- the structured CUDA context payload is still a compact surrogate for backend
  decode state; it is semantically readable now, but it is not yet a real
  KV-cache image or tensor-backed decode-state record
- the new windowed state image is still intentionally tiny and summary-heavy;
  it behaves more like a compact rehearsal for backend decode state than a
  real device-resident KV layout
- the widened key/value lane planes are still compact synthetic state, not real
  transformer KV tensors or backend-native cache pages
- the per-page lane digests are page-identity aids for the runtime and tests,
  not real backend checksums over device-resident tensor tiles
- Apple ANE detection is still conservative and scaffold-level; it currently
  relies on an explicit environment override instead of validated hardware
  probing

## Most Useful Next Steps

1. Build the Apple capability and planner layer.
2. Materialize route-specific Apple pack and plan payloads behind the existing
   metadata records.
3. Replace the compact CUDA key/value lane image with a more realistic
   tensor-backed page record or backend-owned KV-state payload.
4. Start the thin Go binding once the C ABI settles a bit more.
