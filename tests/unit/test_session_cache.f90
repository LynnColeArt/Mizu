program test_session_cache
  use mod_kinds,          only: i32, i64
  use mod_status,         only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,          only: MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                                MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_CUDA, MIZU_STAGE_DECODE, &
                                MIZU_STAGE_PARK, session_config, session_state
  use mod_model_manifest, only: model_manifest
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_cache_keys,     only: invalidation_version_fields, session_cache_key, build_session_cache_key
  use mod_cache_store,    only: artifact_metadata_record
  use mod_session,        only: initialize_session_state
  use mod_session_cache,  only: runtime_session_cache, session_cache_record, &
                                initialize_runtime_session_cache, reset_runtime_session_cache, &
                                record_session_cache_entry, lookup_session_cache_entry, &
                                mark_session_cache_entry_live, evict_one_inactive_session_cache_entry, &
                                session_cache_key_is_strict, session_cache_record_is_evictable, &
                                session_cache_retention_score

  implicit none

  type(model_manifest)             :: manifest
  type(invalidation_version_fields) :: versions
  type(runtime_session_cache)      :: cache
  type(session_cache_key)          :: small_key
  type(session_cache_key)          :: large_key
  type(session_cache_key)          :: live_key
  type(session_cache_key)          :: uncheckpointed_key
  type(session_cache_key)          :: route_changed_key
  type(session_cache_key)          :: malformed_key
  type(session_state)              :: session
  type(artifact_metadata_record)   :: metadata
  type(session_cache_record)       :: record
  type(session_cache_record)       :: small_record
  type(session_cache_record)       :: large_record
  type(session_cache_record)       :: evicted_record
  integer(i32)                     :: status_code
  logical                          :: found

  status_code = load_model_manifest_from_root("tests/fixtures/models/fixture_mm_tiny", manifest)
  call expect_equal_i32("load multimodal fixture", status_code, MIZU_STATUS_OK)

  versions%planner_version = 7_i32
  versions%pack_version = 3_i32
  versions%backend_version = 11_i32

  call build_session_cache_key(manifest, "checkpoint", MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, &
    4096_i64, 128_i64, small_key, versions)
  call build_session_cache_key(manifest, "checkpoint", MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, &
    8192_i64, 128_i64, large_key, versions)
  call build_session_cache_key(manifest, "checkpoint", MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, &
    4096_i64, 256_i64, live_key, versions)
  call build_session_cache_key(manifest, "checkpoint", MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA, &
    2048_i64, 128_i64, uncheckpointed_key, versions)
  call build_session_cache_key(manifest, "checkpoint", MIZU_BACKEND_FAMILY_APPLE, MIZU_EXEC_ROUTE_ANE, &
    4096_i64, 128_i64, route_changed_key, versions)

  metadata%stage_kind = MIZU_STAGE_PARK
  metadata%backend_family = MIZU_BACKEND_FAMILY_CUDA
  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  metadata%is_materialized = .true.
  metadata%payload_bytes = 512_i64
  metadata%workspace_bytes = 0_i64
  metadata%artifact_format = "cuda_session_checkpoint_v1"
  metadata%payload_fingerprint = "SESSIONA"
  metadata%payload_path = "cache/sessions/SESSIONA.session"

  call initialize_runtime_session_cache(cache)
  call expect_true("generated session key should be strict", session_cache_key_is_strict(small_key))

  call make_session(session, 64_i64, 1001_i64, 16_i32, .true., .true., .false.)
  call lookup_session_cache_entry(cache, small_key, record, found)
  call expect_false("empty session cache should miss", found)

  call record_session_cache_entry(cache, small_key, session, metadata, status_code)
  call expect_equal_i32("record strict parked session", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("session cache entry count", cache%entry_count, 1_i32)

  call lookup_session_cache_entry(cache, small_key, small_record, found)
  call expect_true("matching strict session key should hit", found)
  call expect_equal_i64("session cache hit should increment count", small_record%hit_count, 1_i64)
  call expect_equal_i64("session cache should preserve kv tokens", small_record%kv_token_count, 64_i64)
  call expect_true("parked checkpointed resident session should be evictable", &
    session_cache_record_is_evictable(small_record))

  call lookup_session_cache_entry(cache, route_changed_key, record, found)
  call expect_false("route-changed session key should miss", found)

  metadata%execution_route = MIZU_EXEC_ROUTE_ANE
  call record_session_cache_entry(cache, small_key, session, metadata, status_code)
  call expect_equal_i32("mismatched session metadata should be rejected", &
    status_code, MIZU_STATUS_INVALID_ARGUMENT)
  call expect_equal_i32("rejected session metadata should not add entries", cache%entry_count, 1_i32)

  malformed_key = small_key
  malformed_key%key_text = ""
  call record_session_cache_entry(cache, malformed_key, session, artifact_metadata_record(), status_code)
  call expect_equal_i32("malformed session key should be rejected", status_code, MIZU_STATUS_INVALID_ARGUMENT)

  metadata%execution_route = MIZU_EXEC_ROUTE_CUDA
  metadata%payload_fingerprint = "SESSIONB"
  metadata%payload_path = "cache/sessions/SESSIONB.session"
  call make_session(session, 4096_i64, 2002_i64, 64_i32, .true., .true., .false.)
  call record_session_cache_entry(cache, large_key, session, metadata, status_code)
  call expect_equal_i32("record larger parked session", status_code, MIZU_STATUS_OK)
  call lookup_session_cache_entry(cache, large_key, large_record, found)
  call expect_true("larger parked session should hit", found)
  call expect_true("larger kv session should have stronger retention", &
    session_cache_retention_score(large_record) > session_cache_retention_score(small_record))

  call make_session(session, 1_i64, 3003_i64, 16_i32, .false., .true., .false.)
  call record_session_cache_entry(cache, live_key, session, metadata, status_code, is_live=.true.)
  call expect_equal_i32("record live session entry", status_code, MIZU_STATUS_OK)
  call lookup_session_cache_entry(cache, live_key, record, found)
  call expect_true("live session should hit", found)
  call expect_false("live session should not be evictable", session_cache_record_is_evictable(record))

  metadata%is_materialized = .false.
  metadata%payload_path = ""
  metadata%payload_fingerprint = ""
  call make_session(session, 2_i64, 4004_i64, 16_i32, .true., .true., .false.)
  call record_session_cache_entry(cache, uncheckpointed_key, session, metadata, status_code)
  call expect_equal_i32("record uncheckpointed resident session", status_code, MIZU_STATUS_OK)
  call lookup_session_cache_entry(cache, uncheckpointed_key, record, found)
  call expect_true("uncheckpointed session should hit", found)
  call expect_false("resident session without checkpoint should not be evictable", &
    session_cache_record_is_evictable(record))

  call evict_one_inactive_session_cache_entry(cache, evicted_record, found)
  call expect_true("one inactive checkpointed session should evict", found)
  call expect_equal_string("eviction should choose weakest retained key", &
    evicted_record%key%key_text, small_key%key_text)
  call expect_true("evicted record should be marked evicted", evicted_record%is_evicted)
  call expect_false("evicted record should no longer be resident", evicted_record%is_resident)

  call lookup_session_cache_entry(cache, small_key, record, found)
  call expect_true("evicted entry should remain discoverable by identity", found)
  call expect_true("evicted entry should stay marked evicted", record%is_evicted)
  call expect_false("evicted entry should not be evictable again", session_cache_record_is_evictable(record))

  call mark_session_cache_entry_live(cache, large_key, .true., status_code)
  call expect_equal_i32("mark large session live", status_code, MIZU_STATUS_OK)
  call evict_one_inactive_session_cache_entry(cache, evicted_record, found)
  call expect_false("only live or unsafe entries should remain after eviction", found)

  call reset_runtime_session_cache(cache)
  call lookup_session_cache_entry(cache, large_key, record, found)
  call expect_false("reset session cache should clear entries", found)

  write(*, "(A)") "test_session_cache: PASS"

contains

  subroutine make_session(session, kv_token_count, live_context_hash, context_byte_count, &
                          is_parked, is_resident, is_evicted)
    type(session_state), intent(out) :: session
    integer(i64), intent(in)         :: kv_token_count
    integer(i64), intent(in)         :: live_context_hash
    integer(i32), intent(in)         :: context_byte_count
    logical, intent(in)              :: is_parked
    logical, intent(in)              :: is_resident
    logical, intent(in)              :: is_evicted

    call initialize_session_state(session, session_config(max_context_tokens=8192_i64, max_decode_tokens=128_i64))
    session%handle%value = live_context_hash
    session%kv_token_count = kv_token_count
    session%live_context_hash = live_context_hash
    session%live_context_backend_family = MIZU_BACKEND_FAMILY_CUDA
    session%live_context_execution_route = MIZU_EXEC_ROUTE_CUDA
    session%live_context_producer_stage = MIZU_STAGE_DECODE
    session%live_context_artifact_hash = live_context_hash + 99_i64
    session%live_context_byte_count = context_byte_count
    session%has_resident_live_context = is_resident
    session%has_live_context = .true.
    session%is_parked = is_parked
    session%is_evicted = is_evicted
  end subroutine make_session

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

  subroutine expect_equal_string(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A)") trim(label) // " mismatch"
      error stop 1
    end if
  end subroutine expect_equal_string

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

end program test_session_cache
