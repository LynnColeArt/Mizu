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
  - validates route-specific artifact metadata roundtrips for weight, plan, and
    multimodal records
  - validates reloaded weight, plan, and multimodal keys report hits
- `test_optimization_store.f90`
  - validates runtime-scoped winner selection based on recorded execution
    samples rather than key reuse alone
  - validates save/load roundtrips for persisted optimization evidence
