# Unit Test Suites

These suites map directly to the `API-U*` sections in
[docs/API_TEST_MATRIX.md](../../docs/API_TEST_MATRIX.md).

## Planned Suites

- `test_api_abi.c`
  - `API-U001` to `API-U006`
- `test_api_model.c`
  - `API-U010` to `API-U014`
- `test_api_session.c`
  - `API-U020` to `API-U029`
- `test_api_staging.c`
  - `API-U030` to `API-U036`
- `test_api_execution.c`
  - `API-U040` to `API-U049`
- `test_api_routing.c`
  - `API-U050` to `API-U055`
- `test_api_optimizer.c`
  - `API-U060` to `API-U066`
- `test_api_cache.c`
  - `API-U070` to `API-U074`

The fake backend layer should arrive before most of these files get real test
bodies.

## Current Smoke Tests

- `test_model_manifest_loader.f90`
  - validates explicit decoder-only and multimodal manifest fixtures
  - validates malformed manifest rejection
  - validates built-in target fallback
- `test_cache_keys.f90`
  - validates deterministic cache-key generation across plan, weight, session,
    and multimodal key types
- `test_cache_store.f90`
  - validates save/load roundtrips for persisted artifact-cache presence
  - validates route-specific artifact metadata roundtrips for weight, plan,
    session, and multimodal records
  - validates planned workspace bytes survive metadata persistence
  - validates reloaded weight, plan, session, and multimodal keys report hits
- `test_optimization_store.f90`
  - validates runtime-scoped winner selection based on recorded execution
    samples rather than key reuse alone
  - validates save/load roundtrips for persisted optimization evidence
- `test_backend_registry.f90`
  - validates deterministic backend-inventory aggregation independent of local
    hardware
  - validates runtime state retains the aggregated backend mask and descriptors
- `test_runtime_workspace.f90`
  - validates runtime-scoped workspace reservation keeps a reusable high-water
    mark while clearing in-use bytes after release
  - validates the workspace now owns a real reusable host scratch buffer that
    is allocated on reserve and freed on reset
- `test_session_staging.f90`
  - validates attached token content is copied into session staging state
  - validates copied modal bytes and content hashes are retained until clear
  - validates prefill produces a persistent live-context hash and decode
    advances it while retaining emitted tokens
  - validates a backend-owned live-context byte buffer can be stored and
    updated in session state
  - validates an offloaded CUDA live-context buffer makes direct decode invalid
    until the runtime restores residency
- `test_cuda_planner.f90`
  - validates stage-specific CUDA plan candidates for weight-pack, projector,
    prefill, and decode records
  - validates planner payload text includes materialization-relevant metadata
- `test_cuda_executor.f90`
  - validates CUDA projector execution consumes a materialized projector payload
  - validates CUDA prefill execution consumes staged tokens through a
    materialized plan payload with content-aware hashing
  - validates CUDA prefill now changes its workspace scratch when staged token
    and modal tensor contents change, even when the plan shape stays the same
  - validates CUDA prefill emits a persisted context byte buffer and CUDA
    decode consumes that buffer directly
  - validates those context buffers now carry a versioned, checksummed CUDA
    header
  - validates CUDA prefill and decode now fully populate the fixed-size CUDA
    context payload instead of leaving a partial scratch record
  - validates the fixed-size CUDA context payload now carries semantic token
    and modal digests plus explicit KV/decode-step counters
  - validates repeated CUDA decode steps advance those counters and rolling
    decode state predictably
  - validates the widened CUDA context payload now carries a compact windowed
    state image with page-like KV metadata and a recent-token ring
  - validates repeated CUDA decode steps advance that compact windowed state
    image predictably across decode continuity
  - validates the widened CUDA context payload now carries per-page slot
    payloads in addition to page metadata
  - validates repeated CUDA decode steps append emitted tokens into the
    expected page-local slot payloads
  - validates CUDA decode execution produces a deterministic token from a
    materialized decode payload and varies with direct context-buffer identity
  - validates CUDA decode rejects a context produced by a different decode
    artifact, even when the route stays the same
  - validates CUDA decode rejects a corrupted context payload instead of
    consuming invalid restored state
  - validates the CUDA bridge receives and stamps the runtime workspace scratch
    buffer during projector, prefill, and decode
