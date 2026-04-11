program test_mm_cache
  use mod_kinds,          only: i32, i64
  use mod_status,         only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,          only: MIZU_BACKEND_FAMILY_APPLE, MIZU_DTYPE_BF16, &
                                MIZU_EXEC_ROUTE_ANE, MIZU_MODALITY_KIND_IMAGE, &
                                MIZU_STAGE_PROJECTOR
  use mod_model_manifest, only: model_manifest
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_cache_keys,     only: MAX_CACHE_KEY_LEN, invalidation_version_fields, &
                                multimodal_cache_key, build_multimodal_cache_key
  use mod_cache_store,    only: artifact_metadata_record
  use mod_mm_cache,       only: MM_CACHE_KIND_PREPROCESS, MM_CACHE_KIND_PROJECTOR_OUTPUT, &
                                MM_INVALIDATION_NONE, MM_INVALIDATION_KIND_CHANGED, &
                                MM_INVALIDATION_CONTENT_CHANGED, &
                                MM_INVALIDATION_PROJECTOR_CHANGED, &
                                MM_INVALIDATION_EMBEDDING_CHANGED, &
                                MM_INVALIDATION_KEY_CHANGED, MM_INVALIDATION_ARTIFACT_CHANGED, &
                                runtime_mm_cache, mm_cache_record, &
                                initialize_runtime_mm_cache, reset_runtime_mm_cache, &
                                build_preprocess_reuse_key, build_projector_output_reuse_key, &
                                record_mm_cache_entry, lookup_mm_cache_entry, &
                                invalidate_mm_cache_entries, mm_cache_key_is_strict, &
                                mm_cache_invalidation_reason

  implicit none

  type(model_manifest)             :: manifest
  type(model_manifest)             :: projector_changed_manifest
  type(invalidation_version_fields) :: versions
  type(runtime_mm_cache)           :: cache
  type(multimodal_cache_key)       :: apple_key
  type(multimodal_cache_key)       :: cuda_key
  type(multimodal_cache_key)       :: projector_changed_key
  type(multimodal_cache_key)       :: malformed_key
  type(artifact_metadata_record)   :: preprocess_metadata
  type(artifact_metadata_record)   :: projector_metadata
  type(artifact_metadata_record)   :: changed_metadata
  type(mm_cache_record)            :: record
  type(mm_cache_record)            :: preprocess_record
  type(mm_cache_record)            :: projector_record
  character(len=MAX_CACHE_KEY_LEN) :: reuse_key_text
  integer(i32)                     :: status_code
  integer(i32)                     :: reason
  integer(i32)                     :: invalidated_count
  logical                          :: found

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_mm_tiny", manifest)
  call expect_equal_i32("load multimodal fixture", status_code, MIZU_STATUS_OK)

  versions%planner_version = 7_i32
  versions%pack_version = 3_i32
  versions%backend_version = 11_i32

  call build_multimodal_cache_key(manifest, "apple-m2", "image", MIZU_MODALITY_KIND_IMAGE, &
    MIZU_DTYPE_BF16, 32768_i64, apple_key, versions)
  call build_multimodal_cache_key(manifest, "cuda-sm80", "image", MIZU_MODALITY_KIND_IMAGE, &
    MIZU_DTYPE_BF16, 32768_i64, cuda_key, versions)

  projector_changed_manifest = manifest
  projector_changed_manifest%projector%revision_identity = &
    manifest%projector%revision_identity + 1_i64
  call build_multimodal_cache_key(projector_changed_manifest, "apple-m2", "image", &
    MIZU_MODALITY_KIND_IMAGE, MIZU_DTYPE_BF16, 32768_i64, projector_changed_key, versions)

  preprocess_metadata%artifact_format = "image-preprocess-v1"
  preprocess_metadata%payload_fingerprint = "PREA"
  preprocess_metadata%payload_path = "cache/mm/PREA.pre"
  preprocess_metadata%payload_bytes = 32768_i64

  projector_metadata%stage_kind = MIZU_STAGE_PROJECTOR
  projector_metadata%backend_family = MIZU_BACKEND_FAMILY_APPLE
  projector_metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  projector_metadata%is_materialized = .true.
  projector_metadata%payload_bytes = 4096_i64
  projector_metadata%workspace_bytes = 8192_i64
  projector_metadata%artifact_format = "ane-projector-embeddings-v1"
  projector_metadata%payload_fingerprint = "PROJA"
  projector_metadata%payload_path = "cache/mm/PROJA.emb"

  changed_metadata = projector_metadata
  changed_metadata%payload_fingerprint = "PROJB"
  changed_metadata%payload_path = "cache/mm/PROJB.emb"

  call initialize_runtime_mm_cache(cache)
  call expect_true("generated multimodal key should be strict", mm_cache_key_is_strict(apple_key))

  call build_preprocess_reuse_key(apple_key, 111_i64, reuse_key_text, status_code)
  call expect_equal_i32("build preprocess reuse key", status_code, MIZU_STATUS_OK)
  call expect_contains("preprocess reuse key prefix", reuse_key_text, "mm_pre:v")
  call expect_not_contains("preprocess key should be device reusable", reuse_key_text, ":device=")
  call expect_not_contains("preprocess key should ignore projector revision", reuse_key_text, ":proj=")

  call build_projector_output_reuse_key(apple_key, 111_i64, 16_i64, reuse_key_text, status_code)
  call expect_equal_i32("build projector output reuse key", status_code, MIZU_STATUS_OK)
  call expect_contains("projector reuse key prefix", reuse_key_text, "mm_projector:v")
  call expect_contains("projector key should bind device", reuse_key_text, ":device=apple-m2")
  call expect_contains("projector key should bind revision", reuse_key_text, ":proj=")

  call lookup_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PREPROCESS, 111_i64, 0_i64, &
    record, found)
  call expect_false("empty multimodal cache should miss", found)

  call record_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PREPROCESS, 111_i64, 0_i64, &
    preprocess_metadata, status_code)
  call expect_equal_i32("record reusable preprocess entry", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("preprocess entry count", cache%entry_count, 1_i32)

  call lookup_mm_cache_entry(cache, cuda_key, MM_CACHE_KIND_PREPROCESS, 111_i64, 0_i64, &
    preprocess_record, found)
  call expect_true("preprocess output should reuse across devices", found)
  call expect_equal_i64("preprocess lookup should increment hits", preprocess_record%hit_count, 1_i64)
  call expect_equal_i32("preprocess record should preserve cache kind", &
    preprocess_record%cache_kind, MM_CACHE_KIND_PREPROCESS)

  call lookup_mm_cache_entry(cache, cuda_key, MM_CACHE_KIND_PREPROCESS, 222_i64, 0_i64, &
    record, found)
  call expect_false("content-changed preprocess key should miss", found)

  reason = mm_cache_invalidation_reason(preprocess_record, cuda_key, MM_CACHE_KIND_PREPROCESS, &
    222_i64, 0_i64, artifact_metadata_record())
  call expect_equal_i32("preprocess content change reason", reason, MM_INVALIDATION_CONTENT_CHANGED)

  reason = mm_cache_invalidation_reason(preprocess_record, projector_changed_key, &
    MM_CACHE_KIND_PREPROCESS, 111_i64, 0_i64, artifact_metadata_record())
  call expect_equal_i32("preprocess should ignore projector revision changes", &
    reason, MM_INVALIDATION_NONE)

  reason = mm_cache_invalidation_reason(preprocess_record, apple_key, &
    MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, 16_i64, artifact_metadata_record())
  call expect_equal_i32("cache-kind change reason", reason, MM_INVALIDATION_KIND_CHANGED)

  call record_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, &
    16_i64, projector_metadata, status_code)
  call expect_equal_i32("record projector output entry", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("projector entry count", cache%entry_count, 2_i32)

  call lookup_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, &
    16_i64, projector_record, found)
  call expect_true("matching projector output should hit", found)
  call expect_equal_i64("projector lookup should increment hits", projector_record%hit_count, 1_i64)

  call lookup_mm_cache_entry(cache, cuda_key, MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, &
    16_i64, record, found)
  call expect_false("projector output should not reuse across devices", found)

  reason = mm_cache_invalidation_reason(projector_record, cuda_key, &
    MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, 16_i64, artifact_metadata_record())
  call expect_equal_i32("projector device change reason", reason, MM_INVALIDATION_KEY_CHANGED)

  reason = mm_cache_invalidation_reason(projector_record, projector_changed_key, &
    MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, 16_i64, artifact_metadata_record())
  call expect_equal_i32("projector revision change reason", reason, &
    MM_INVALIDATION_PROJECTOR_CHANGED)

  reason = mm_cache_invalidation_reason(projector_record, apple_key, &
    MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, 8_i64, artifact_metadata_record())
  call expect_equal_i32("projector embedding-count change reason", reason, &
    MM_INVALIDATION_EMBEDDING_CHANGED)

  reason = mm_cache_invalidation_reason(projector_record, apple_key, &
    MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, 16_i64, changed_metadata)
  call expect_equal_i32("projector artifact change reason", reason, &
    MM_INVALIDATION_ARTIFACT_CHANGED)

  malformed_key = apple_key
  malformed_key%key_text = ""
  call record_mm_cache_entry(cache, malformed_key, MM_CACHE_KIND_PREPROCESS, 111_i64, &
    0_i64, preprocess_metadata, status_code)
  call expect_equal_i32("malformed multimodal key should be rejected", &
    status_code, MIZU_STATUS_INVALID_ARGUMENT)

  call invalidate_mm_cache_entries(cache, cuda_key, MM_CACHE_KIND_PREPROCESS, 222_i64, &
    0_i64, artifact_metadata_record(), invalidated_count)
  call expect_equal_i32("content invalidation should retire one preprocess entry", &
    invalidated_count, 1_i32)

  call lookup_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PREPROCESS, 111_i64, 0_i64, &
    record, found)
  call expect_false("invalidated preprocess entry should stop hitting", found)

  call lookup_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, &
    16_i64, record, found)
  call expect_true("preprocess invalidation should not retire projector output", found)

  call reset_runtime_mm_cache(cache)
  call lookup_mm_cache_entry(cache, apple_key, MM_CACHE_KIND_PROJECTOR_OUTPUT, 111_i64, &
    16_i64, record, found)
  call expect_false("reset multimodal cache should clear entries", found)

  write(*, "(A)") "test_mm_cache: PASS"

contains

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
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i64

  subroutine expect_contains(label, haystack, needle)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: haystack
    character(len=*), intent(in) :: needle

    if (index(haystack, needle) <= 0) then
      write(*, "(A)") trim(label) // " missing: " // trim(needle)
      error stop 1
    end if
  end subroutine expect_contains

  subroutine expect_not_contains(label, haystack, needle)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: haystack
    character(len=*), intent(in) :: needle

    if (index(haystack, needle) > 0) then
      write(*, "(A)") trim(label) // " unexpectedly contained: " // trim(needle)
      error stop 1
    end if
  end subroutine expect_not_contains

  subroutine expect_true(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (.not. condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_true

  subroutine expect_false(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_false

end program test_mm_cache
