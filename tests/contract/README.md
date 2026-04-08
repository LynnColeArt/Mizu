# Contract Test Suites

These suites cover the `API-C*` portion of
[docs/API_TEST_MATRIX.md](../../docs/API_TEST_MATRIX.md).

## Planned Suites

- `test_header_c_smoke.c`
  - header compiles as C
- `test_header_cpp_smoke.cpp`
  - header compiles as C++
- `test_opaque_handles.c`
  - confirms consumers only see opaque handle declarations
- `test_stage_reports.c`
  - exercises `prefill`, `decode`, `park`, and `resume` through the public API
    and verifies nonzero report plan IDs, route-honest stage reports, in-process
    cache-hit flags, persisted winner reuse across runtimes, and persisted
    route-specific artifact hits for weight, multimodal, and plan stages
  - inspects the persisted artifact cache file to confirm backend-specific
    weight and plan metadata records are written
- `test_cuda_artifacts.c`
  - forces the CUDA route through the public API
  - verifies route-specific CUDA formats are persisted
  - verifies CUDA stub artifact payload files are materialized under `cache_root`
  - verifies the public API can drive CUDA-owned stub prefill and decode paths
  - verifies session-state transitions across staging, prefill, decode,
    `park`, and `resume`
  - verifies decode output can be read back through
    `mizu_session_read_output`
  - verifies CUDA `park` and `resume` materialize a persisted session
    checkpoint artifact under `cache_root`
  - verifies parked CUDA sessions can resume after their active context has
    been offloaded behind that checkpoint
  - verifies resumed CUDA sessions only accept checkpoint state that still
    matches the stored route and producer lineage expectations
  - verifies a fresh runtime can replay the same narrow multimodal CUDA flow
    with warm cache reuse and reproduce the same decode token
  - verifies that narrow public CUDA flow emits the exact current reference
    token for the fixture instead of only a nonzero token
- `test_go_binding_smoke.go`
  - reserved for the first thin Go binding smoke path
