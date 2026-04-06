# Integration Test Suites

These suites are reserved for the `API-I*` hardware-backed checks in
[docs/API_TEST_MATRIX.md](../../docs/API_TEST_MATRIX.md).

## Planned Areas

- Apple route honesty
- ANE-versus-Metal no-silent-fallback checks
- CUDA lifecycle parity
- multimodal end-to-end projector-plus-decoder validation

These tests should only be wired once the fake-backend unit coverage is in
place and the runtime skeleton can execute a real path.
