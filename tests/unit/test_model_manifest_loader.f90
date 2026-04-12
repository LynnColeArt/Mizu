program test_model_manifest_loader
  use mod_kinds,          only: i32, i64
  use mod_status,         only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_IO_ERROR
  use mod_types,          only: MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                                MIZU_MODEL_FEATURE_NONE, MIZU_MODEL_FEATURE_MULTIMODAL, &
                                MIZU_MODEL_FEATURE_PROJECTOR, SOURCE_FORMAT_MIZU_MANIFEST, &
                                SOURCE_FORMAT_BUILTIN_TARGET, SOURCE_FORMAT_MIZU_IMPORT_BUNDLE
  use mod_model_manifest, only: model_manifest, manifest_tensor_count, &
                                manifest_modality_count
  use mod_model_loader,   only: load_model_manifest_from_root

  implicit none

  type(model_manifest) :: manifest
  integer(i32)         :: status_code

  call expect_manifest_success("tests/fixtures/models/fixture_decoder_tiny", &
    MIZU_MODEL_FAMILY_QWEN3_5, SOURCE_FORMAT_MIZU_MANIFEST, .false., .false.)

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_decoder_tiny", manifest)
  call expect_equal_i64("decoder features stay empty", manifest%model_features, MIZU_MODEL_FEATURE_NONE)
  call expect_equal_i32("decoder tensor count", manifest_tensor_count(manifest), 5_i32)
  call expect_equal_i32("decoder modality count", manifest_modality_count(manifest), 0_i32)

  call expect_manifest_success("tests/fixtures/models/fixture_mm_tiny", &
    MIZU_MODEL_FAMILY_GEMMA4, SOURCE_FORMAT_MIZU_MANIFEST, .true., .true.)

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_mm_tiny", manifest)
  call expect_equal_i64("multimodal feature bit", iand(manifest%model_features, &
    MIZU_MODEL_FEATURE_MULTIMODAL), MIZU_MODEL_FEATURE_MULTIMODAL)
  call expect_equal_i64("projector feature bit", iand(manifest%model_features, &
    MIZU_MODEL_FEATURE_PROJECTOR), MIZU_MODEL_FEATURE_PROJECTOR)
  call expect_equal_i32("multimodal modality count", manifest_modality_count(manifest), 1_i32)

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_bad_manifest", manifest)
  call expect_equal_i32("bad manifest status", status_code, MIZU_STATUS_INVALID_ARGUMENT)

  call expect_manifest_success("tests/fixtures/models/fixture_import_bundle_tiny", &
    MIZU_MODEL_FAMILY_QWEN3_5, SOURCE_FORMAT_MIZU_IMPORT_BUNDLE, .true., .true.)

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_import_bundle_tiny", manifest)
  call expect_equal_i32("import bundle tensor count", manifest_tensor_count(manifest), 5_i32)
  call expect_equal_i32("import bundle modality count", manifest_modality_count(manifest), 1_i32)
  call expect_equal_logical("import bundle should preserve projector presence", manifest%projector%is_present, .true.)
  call expect_equal_text("import bundle should carry tensor source path", &
    trim(manifest%tensors(1)%source_path), "weights/token_embeddings.bin")
  call expect_equal_text("import bundle should carry tensor storage type", &
    trim(manifest%tensors(1)%storage_type), "q4_k")
  call expect_equal_i64("import bundle should carry first tensor source offset", &
    manifest%tensors(1)%source_offset, 0_i64)
  call expect_equal_i64("import bundle should carry second tensor source offset", &
    manifest%tensors(2)%source_offset, 128_i64)
  call expect_equal_i64("import bundle should carry projector tensor source offset", &
    manifest%tensors(5)%source_offset, 512_i64)
  call expect_equal_text("import bundle should carry projector artifact path", &
    trim(manifest%projector%artifact_path), "projector/vision_projector.bin")
  call expect_equal_text("import bundle should carry imported source model id", &
    trim(manifest%provenance%source_model_id), "qwen-3.5-9b-imported")

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_bad_import_bundle", manifest)
  call expect_equal_i32("bad import bundle status", status_code, MIZU_STATUS_IO_ERROR)

  call expect_manifest_success("tests/fixtures/models/qwen_builtin_target", &
    MIZU_MODEL_FAMILY_QWEN3_5, SOURCE_FORMAT_BUILTIN_TARGET, .true., .true.)

  write(*, "(A)") "test_model_manifest_loader: PASS"

contains

  subroutine expect_manifest_success(model_root, expected_family, expected_source_format, &
                                     expect_projector, expect_modalities)
    character(len=*), intent(in) :: model_root
    integer(i32), intent(in)     :: expected_family
    integer(i32), intent(in)     :: expected_source_format
    logical, intent(in)          :: expect_projector
    logical, intent(in)          :: expect_modalities
    integer(i32)                 :: local_status

    local_status = load_model_manifest_from_root(model_root, manifest)
    call expect_equal_i32("manifest load status for " // trim(model_root), local_status, MIZU_STATUS_OK)
    call expect_equal_i32("model family for " // trim(model_root), manifest%model_family, expected_family)
    call expect_equal_i32("source format for " // trim(model_root), &
      manifest%provenance%source_format, expected_source_format)
    call expect_equal_logical("projector presence for " // trim(model_root), &
      manifest%projector%is_present, expect_projector)
    if (expect_modalities) then
      call expect_equal_i32("modality count for " // trim(model_root), manifest_modality_count(manifest), 1_i32)
    end if
  end subroutine expect_manifest_success

  subroutine expect_equal_i32(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i32), intent(in)     :: actual
    integer(i32), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_i64(label, actual, expected)
    use mod_kinds, only: i64
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i64

  subroutine expect_equal_logical(label, actual, expected)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: actual
    logical, intent(in)          :: expected

    if (.not. (actual .eqv. expected)) then
      write(*, "(A,1X,L1,1X,A,1X,L1)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_logical

  subroutine expect_equal_text(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A,1X,A,1X,A,1X,A)") trim(label), trim(actual), "/=", trim(expected)
      error stop 1
    end if
  end subroutine expect_equal_text

end program test_model_manifest_loader
