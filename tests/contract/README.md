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
- `test_backend_availability.c`
  - forces backend-availability overrides through the public API
  - verifies `mizu_model_open` fails early with `MIZU_STATUS_NO_VALID_PLAN`
    when the requested backend family is unavailable on the runtime
  - verifies the runtime surfaces a useful last-error string for that case
  - verifies forcing Apple availability makes the same model-open request pass
- `test_stage_reports.c`
  - exercises `prefill`, `decode`, `park`, and `resume` through the public API
    and verifies nonzero report plan IDs, route-honest stage reports, in-process
    cache-hit flags, persisted winner reuse across runtimes, and persisted
    route-specific artifact hits for weight, multimodal, and plan stages
  - verifies the Apple public path can execute with no configured `cache_root`
    by using the virtual-payload bridge path for projector, prefill, and decode
  - inspects the persisted artifact cache file to confirm backend-specific
    weight and plan metadata records are written
  - locks the current Apple planner payload labels for ANE weight/projector/
    prefill and Metal prefill artifacts
- `test_cuda_artifacts.c`
  - forces the CUDA route through the public API
  - verifies route-specific CUDA formats are persisted
  - verifies CUDA artifact payload files are materialized under `cache_root`
  - verifies those materialized CUDA weight and projector artifacts retain
    imported `mizu_import/` source-path lineage
  - verifies the CUDA weight artifact now carries a deterministic import-driven
    pack table with exact packed tensor count, byte total, and stable offsets
  - verifies CUDA projector and prefill artifacts now retain explicit
    dependency on that packed model layout
  - verifies CUDA prefill and decode artifacts now retain stage-specific
    `pack_use_*` records with selected imported tensor names, offsets, and
    byte spans
  - verifies CUDA prefill and decode artifacts now also retain compact numeric
    `pack_dispatch*` records for the first selected packed tensors
  - verifies CUDA prefill and decode artifacts now also retain importer-rooted
    `pack_span*` records for the first selected packed tensors
  - verifies CUDA prefill and decode artifacts now also retain `pack_span_cache`
    references to persisted `.spancache` sidecars under `cache_root`
  - verifies those `.spancache` sidecars now retain staged sample bytes for
    the selected imported tensor spans
  - verifies those `.spancache` sidecars now also retain compact staged
    pack-page records for the selected imported tensor spans
  - verifies those `.spancache` sidecars now also reference pack-owned
    `.packtiles` payloads materialized beside the CUDA weight-pack artifact
  - verifies those pack-owned `.packtiles` payloads now derive their staged
    page/tile records from weight-pack materialization metadata rather than
    sampled importer preview bytes
  - verifies those pack-owned `.packtiles` payloads now also reference a
    dedicated `.packpayload` sibling carrying the staged page/tile bytes
  - verifies compact CUDA `pack_dispatch*` records now carry explicit packed
    entry indices and that the `.spancache` sidecars retain those pack indices
  - verifies those `.spancache` sidecars now also reference dedicated
    `.tilecache` payloads for the selected imported tensor spans
  - verifies those `.tilecache` payloads retain staged tensor-tile records for
    the selected imported tensor spans
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
  - verifies that warm replay still reproduces the same decode token even
    after imported tensor bytes are mutated, as long as the persisted
    `.spancache` sidecars are present
  - verifies the same public CUDA flow still reproduces the same decode token
    after those plan-local `.spancache` and `.tilecache` files are removed, as
    long as the pack-owned weight `.packtiles` cache remains available
  - verifies that narrow public CUDA flow emits a stable positive placeholder
    token and reproduces it for the same staged multimodal context
- `test_go_binding_smoke.go`
  - reserved for the first thin Go binding smoke path
