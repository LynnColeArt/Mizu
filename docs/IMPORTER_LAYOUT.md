# Importer Layout

This document defines the first concrete imported-model layout for `mizu`.

The goal is simple:

- keep `manifest.mizu` as the root logical contract
- let an optional `mizu_import/` bundle carry imported asset inventories
- make loader validation strict enough that later CUDA and Apple work can trust
  the layout

## Canonical Layout

```text
<model_root>/
  manifest.mizu
  mizu_import/
    layout.mizu
    tensors.tsv
    gguf_tensors.tsv        # optional source-format sidecar
    modalities.tsv
    projector.mizu
    weights/
    projector/
```

`manifest.mizu` remains the logical model manifest.

`mizu_import/` is the imported-asset bundle that enriches or overrides the
logical manifest with concrete asset inventory.

## `layout.mizu`

Supported keys:

- `layout_version`
- `tensor_inventory`
- `modality_inventory`
- `projector_inventory`
- `family`
- `source_model_id`
- `source_revision`
- `source_hash_text`
- `tokenizer`
- `logical_model_hash`
- `model_features`
- `projector_present`
- `projector_slot`
- `projector_placeholder_count`
- `projector_input_dtype`
- `projector_embedding_dtype`
- `projector_revision`

Current required behavior:

- `layout_version` must be `1`
- `tensor_inventory` must resolve to an existing file inside `mizu_import/`
- `modality_inventory` and `projector_inventory` may be omitted with `-`

## `tensors.tsv`

Each non-comment line must have 6 `|`-separated fields:

```text
tensor_name|tensor_role|dtype|layout_name|relative_path|shape
```

Example:

```text
token_embeddings|embedding_table|bf16|row_major|weights/token_embeddings.bin|152064x3584
```

Rules:

- `relative_path` must be relative to `mizu_import/`
- absolute paths are rejected
- parent traversal like `..` is rejected
- the referenced file must exist
- `shape` must contain 1 to `MAX_TENSOR_RANK` positive dimensions

## `modalities.tsv`

Each non-comment line must have 5 `|`-separated fields:

```text
placeholder_ordinal|slot_name|modality_kind|storage_kind|dtype
```

Example:

```text
1|image|image|encoded_bytes|u8
```

## `projector.mizu`

Supported keys:

- `present`
- `slot` or `slot_name`
- `placeholder_count`
- `input_modality_kind`
- `input_dtype`
- `embedding_dtype`
- `revision_identity`
- `artifact_path`

If `present=true`, then:

- `artifact_path` must be a safe relative path under `mizu_import/`
- the referenced artifact file must exist

## Validation Behavior

Current loader behavior is intentionally strict:

- malformed inventory rows return `MIZU_STATUS_INVALID_ARGUMENT`
- missing required inventory files return `MIZU_STATUS_IO_ERROR`
- unsafe import-relative paths return `MIZU_STATUS_INVALID_ARGUMENT`
- missing referenced asset files return `MIZU_STATUS_IO_ERROR`

When a bundle is successfully applied, the loaded manifest reports
`SOURCE_FORMAT_MIZU_IMPORT_BUNDLE`.

## Tooling Location

Importer and conversion tooling lives under `tools/import/`.

That tooling should produce:

- a root `manifest.mizu`
- a validated `mizu_import/` bundle
- stable relative asset paths suitable for later backend pack generation

The first concrete tool is:

- `tools/import/hf_safetensors_to_mizu.py`
- `tools/import/gguf_to_mizu.py`

The safetensors importer reads local HuggingFace-style `.safetensors` headers
directly with the Python standard library, classifies common Qwen/Gemma
tensor-name patterns into Mizu tensor roles, writes the bundle files above, and
symlinks or copies source shards under `mizu_import/weights/` so loader
validation can continue to reject unsafe external paths.

The GGUF importer reads GGUF metadata and tensor-info headers directly, can
pair a model GGUF with an optional mmproj GGUF, writes the same loader-facing
bundle, and adds `gguf_tensors.tsv` as a source-format sidecar that preserves
GGUF tensor type and data-offset details. The current loader ignores that
sidecar; it is there so the next storage-schema pass can distinguish quantized
GGUF storage from the normalized staging dtype used in `tensors.tsv`.
