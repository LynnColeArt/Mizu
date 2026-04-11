# Import Tooling

Importer and conversion tooling for `mizu` lives here.

The intended job of this directory is:

- read source model assets from upstream layouts
- normalize them into the `manifest.mizu` plus `mizu_import/` bundle shape
- emit stable relative paths and inventory files
- fail loudly when imported assets are incomplete or ambiguous

The first concrete target layout is documented in:

- `docs/IMPORTER_LAYOUT.md`

## HuggingFace Safetensors Smoke Importer

`hf_safetensors_to_mizu.py` converts a local HuggingFace-style safetensors
directory into the Mizu bundle shape without third-party Python dependencies.

Example:

```sh
python3 tools/import/hf_safetensors_to_mizu.py /models/qwen-vl \
  --family qwen3_5 \
  --link-mode symlink
```

The tool writes:

- `<model_root>/manifest.mizu`
- `<model_root>/mizu_import/layout.mizu`
- `<model_root>/mizu_import/tensors.tsv`
- `<model_root>/mizu_import/modalities.tsv`
- `<model_root>/mizu_import/projector.mizu`

By default, safetensors shards are symlinked into `mizu_import/weights/` so the
Fortran loader can keep enforcing safe import-relative paths without copying
large model files. Use `--link-mode copy` for small fixtures or portable test
bundles, and `--force` when intentionally regenerating an existing bundle.

This is an asset-layout smoke importer, not a real inference converter. It
classifies common Qwen/Gemma tensor-name patterns into Mizu tensor roles and
creates stable model/projector identity, giving us a concrete way to start
testing real local assets before backend math is complete.
