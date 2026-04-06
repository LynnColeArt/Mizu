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
- `test_go_binding_smoke.go`
  - reserved for the first thin Go binding smoke path
