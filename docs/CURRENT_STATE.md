# Mizu Current State

Last updated: 2026-04-06

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

### Self-Optimization

- stage selection uses route-neutral optimization identities
- ANE, Metal, and CUDA candidates can be explored under one shared workload key
- exploration is bounded by `exploration_budget`
- repeated work can reuse the measured winner
- optimization evidence is persisted to disk through `optimization_store_v1.txt`

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
  - artifact format label
  - payload fingerprint
  - future payload path

### Tests That Pass

- `test_model_manifest_loader`
- `test_cache_keys`
- `test_cache_store`
- `test_optimization_store`
- `test_stage_reports`

## What Is Still Stubbed

- no real Apple backend exists yet
- no real ANE planner or executor exists yet
- no real Metal executor exists yet
- no real CUDA planner or executor exists yet
- model load does not build actual packed weights
- plan selection does not materialize backend-native plan payloads
- projector, prefill, and decode do not run real kernels yet
- Go bindings do not exist yet
- there is no top-level build entrypoint yet

## Important Honesty Notes

- persisted artifact metadata is real metadata, not fake cache hits
- route-specific artifact descriptors are persisted even when payload bytes are
  `0`
- `payload_path` currently points at the future artifact location that a real
  backend-owned pack or plan record would use
- `is_materialized = .false.` means the runtime knows the artifact identity and
  route, but does not yet claim to have built the payload

## Most Useful Next Steps

1. Build the Apple capability and planner layer.
2. Materialize route-specific Apple pack and plan payloads behind the existing
   metadata records.
3. Add a real build entrypoint so tests and library builds are reproducible.
4. Start the thin Go binding once the C ABI settles a bit more.
