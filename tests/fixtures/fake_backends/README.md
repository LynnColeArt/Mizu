# Fake Backend Fixtures

Planned fake backends from [docs/API_TEST_MATRIX.md](../../../docs/API_TEST_MATRIX.md):

- `fake_apple_ane_only`
- `fake_apple_metal_only`
- `fake_apple_split`
- `fake_cuda`
- `fake_dual_plan_backend`

These fixtures should be deterministic and controllable enough to test route
honesty, fallback rules, and optimizer convergence without target hardware.
