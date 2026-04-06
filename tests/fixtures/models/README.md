# Model Fixtures

Planned fixture models from [docs/API_TEST_MATRIX.md](../../../docs/API_TEST_MATRIX.md):

- `fixture_decoder_tiny`
- `fixture_mm_tiny`
- `fixture_mm_preembedded`
- `fixture_bad_manifest`

These fixtures should be small enough to use in unit tests and descriptive
enough to validate projector metadata, placeholder mapping, and model-family
reporting.

Current concrete fixtures:

- `fixture_decoder_tiny/manifest.mizu`
  - explicit decoder-only manifest
- `fixture_mm_tiny/manifest.mizu`
  - explicit multimodal manifest with projector metadata
- `fixture_bad_manifest/manifest.mizu`
  - intentionally malformed manifest for negative-path tests
