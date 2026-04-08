# Import Tooling

Importer and conversion tooling for `mizu` lives here.

The intended job of this directory is:

- read source model assets from upstream layouts
- normalize them into the `manifest.mizu` plus `mizu_import/` bundle shape
- emit stable relative paths and inventory files
- fail loudly when imported assets are incomplete or ambiguous

The first concrete target layout is documented in:

- `docs/IMPORTER_LAYOUT.md`
