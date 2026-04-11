#!/usr/bin/env python3
"""Smoke-test the dependency-free GGUF importer."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORTER = REPO_ROOT / "tools" / "import" / "gguf_to_mizu.py"

VALUE_TYPES = {
    "uint32": 4,
    "bool": 7,
    "string": 8,
}

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_K": 12,
    "Q5_K": 13,
}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_gguf_importer_") as temp_root:
        temp_path = Path(temp_root)
        qwen_model = temp_path / "qwen35.gguf"
        qwen_projector = temp_path / "mmproj-qwen35.gguf"
        qwen_output = temp_path / "qwen35_mizu"

        write_gguf(
            qwen_model,
            {
                "general.architecture": ("string", "qwen35"),
                "general.name": ("string", "Qwen3.5 9B"),
                "general.type": ("string", "model"),
                "general.file_type": ("uint32", 15),
                "general.quantization_version": ("uint32", 2),
                "tokenizer.ggml.model": ("string", "gpt2"),
            },
            [
                ("token_embd.weight", [4096, 248320], "Q4_K", 0),
                ("blk.0.attn_qkv.weight", [4096, 8192], "Q5_K", 128),
                ("output_norm.weight", [4096], "F32", 256),
                ("output.weight", [4096, 248320], "Q4_K", 384),
            ],
        )
        write_gguf(
            qwen_projector,
            {
                "general.architecture": ("string", "clip"),
                "general.name": ("string", "Qwen3.5 9B mmproj"),
                "general.type": ("string", "mmproj"),
                "clip.has_vision_encoder": ("bool", True),
                "general.file_type": ("uint32", 1),
                "general.quantization_version": ("uint32", 2),
            },
            [
                ("v.blk.0.attn_qkv.weight", [1152, 3456], "F16", 0),
                ("mm.0.weight", [1152, 4096], "F16", 128),
                ("mm.2.bias", [4096], "F32", 256),
            ],
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(qwen_model),
                "--projector-gguf",
                str(qwen_projector),
                "--output-root",
                str(qwen_output),
                "--link-mode",
                "copy",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode

        expect_file_contains(qwen_output / "manifest.mizu", "family = qwen3_5")
        expect_file_contains(qwen_output / "mizu_import" / "layout.mizu", "gguf_inventory = gguf_tensors.tsv")
        expect_file_contains(qwen_output / "mizu_import" / "layout.mizu", "projector_present = true")
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "token_embd.weight|embedding_table|f16|row_major|weights/qwen35.gguf|4096x248320",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "blk.0.attn_qkv.weight|decoder_stack|f16|packed|weights/qwen35.gguf|4096x8192",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "output.weight|token_projection|f16|row_major|weights/qwen35.gguf|4096x248320",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "tensors.tsv",
            "mm.0.weight|multimodal_projector|f16|packed|weights/mmproj-qwen35.gguf|1152x4096",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "gguf_tensors.tsv",
            "token_embd.weight|model|q4_k|f16|row_major|weights/qwen35.gguf|0|4096x248320",
        )
        expect_file_contains(
            qwen_output / "mizu_import" / "projector" / "projector_assets.mizu",
            "mm.0.weight|weights/mmproj-qwen35.gguf|offset=128|ggml_type=f16",
        )
        expect_path_exists(qwen_output / "mizu_import" / "weights" / "qwen35.gguf")
        expect_path_exists(qwen_output / "mizu_import" / "weights" / "mmproj-qwen35.gguf")

        gemma_model = temp_path / "gemma4.gguf"
        gemma_output = temp_path / "gemma4_mizu"
        write_gguf(
            gemma_model,
            {
                "general.architecture": ("string", "gemma4"),
                "general.name": ("string", "Gemma-4-26B-A4B-It"),
                "general.type": ("string", "model"),
                "general.file_type": ("uint32", 29),
                "general.quantization_version": ("uint32", 2),
                "tokenizer.ggml.model": ("string", "gemma4"),
            },
            [
                ("token_embd.weight", [2816, 262144], "Q5_K", 0),
                ("blk.0.ffn_gate.weight", [2816, 2112], "Q5_K", 128),
                ("output_norm.weight", [2816], "F32", 256),
            ],
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(IMPORTER),
                str(gemma_model),
                "--output-root",
                str(gemma_output),
                "--link-mode",
                "copy",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode

        expect_file_contains(gemma_output / "manifest.mizu", "family = gemma4")
        expect_file_contains(gemma_output / "manifest.mizu", "projector_present = false")
        expect_file_contains(
            gemma_output / "mizu_import" / "tensors.tsv",
            "token_embd.weight|embedding_table|f16|row_major|weights/gemma4.gguf|2816x262144",
        )

    print("test_gguf_to_mizu: PASS")
    return 0


def write_gguf(
    path: Path,
    metadata: dict[str, tuple[str, object]],
    tensors: list[tuple[str, list[int], str, int]],
) -> None:
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", len(tensors)))
        handle.write(struct.pack("<Q", len(metadata)))
        for key, (value_type, value) in metadata.items():
            write_string(handle, key)
            handle.write(struct.pack("<I", VALUE_TYPES[value_type]))
            if value_type == "string":
                write_string(handle, str(value))
            elif value_type == "bool":
                handle.write(struct.pack("<?", bool(value)))
            elif value_type == "uint32":
                handle.write(struct.pack("<I", int(value)))
            else:
                raise AssertionError(f"unsupported metadata type in fixture: {value_type}")

        for name, shape, ggml_type, offset in tensors:
            write_string(handle, name)
            handle.write(struct.pack("<I", len(shape)))
            for dim in shape:
                handle.write(struct.pack("<Q", dim))
            handle.write(struct.pack("<I", GGML_TYPES[ggml_type]))
            handle.write(struct.pack("<Q", offset))

        handle.write(b"\0" * 128)


def write_string(handle: object, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def expect_file_contains(path: Path, needle: str) -> None:
    expect_path_exists(path)
    expect_contains(path.read_text(encoding="utf-8"), needle)


def expect_contains(haystack: str, needle: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"missing expected text: {needle}")


def expect_path_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"missing expected path: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
