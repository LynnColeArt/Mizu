module mod_mm_cache
  use mod_kinds,       only: i32, i64
  use mod_status,      only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,       only: MIZU_BACKEND_FAMILY_NONE, MIZU_DTYPE_UNKNOWN, &
                             MIZU_EXEC_ROUTE_NONE, MIZU_MODALITY_KIND_UNKNOWN, &
                             MIZU_STAGE_NONE, MIZU_STAGE_PROJECTOR
  use mod_cache_keys,  only: MAX_CACHE_KEY_LEN, multimodal_cache_key
  use mod_cache_store, only: artifact_metadata_record, artifact_metadata_is_defined

  implicit none

  private
  public :: MM_CACHE_KIND_UNKNOWN, MM_CACHE_KIND_PREPROCESS, MM_CACHE_KIND_PROJECTOR_OUTPUT
  public :: MM_INVALIDATION_NONE, MM_INVALIDATION_ENTRY_INVALID, MM_INVALIDATION_KIND_CHANGED
  public :: MM_INVALIDATION_CONTENT_CHANGED, MM_INVALIDATION_PROJECTOR_CHANGED
  public :: MM_INVALIDATION_EMBEDDING_CHANGED, MM_INVALIDATION_KEY_CHANGED
  public :: MM_INVALIDATION_ARTIFACT_CHANGED
  public :: mm_cache_record, runtime_mm_cache
  public :: initialize_runtime_mm_cache, reset_runtime_mm_cache
  public :: build_preprocess_reuse_key, build_projector_output_reuse_key
  public :: record_mm_cache_entry, lookup_mm_cache_entry
  public :: invalidate_mm_cache_entries
  public :: mm_cache_key_is_strict, mm_cache_record_is_reusable
  public :: mm_cache_invalidation_reason

  integer(i32), parameter :: MM_CACHE_KIND_UNKNOWN = 0_i32
  integer(i32), parameter :: MM_CACHE_KIND_PREPROCESS = 1_i32
  integer(i32), parameter :: MM_CACHE_KIND_PROJECTOR_OUTPUT = 2_i32

  integer(i32), parameter :: MM_INVALIDATION_NONE = 0_i32
  integer(i32), parameter :: MM_INVALIDATION_ENTRY_INVALID = 1_i32
  integer(i32), parameter :: MM_INVALIDATION_KIND_CHANGED = 2_i32
  integer(i32), parameter :: MM_INVALIDATION_CONTENT_CHANGED = 3_i32
  integer(i32), parameter :: MM_INVALIDATION_PROJECTOR_CHANGED = 4_i32
  integer(i32), parameter :: MM_INVALIDATION_EMBEDDING_CHANGED = 5_i32
  integer(i32), parameter :: MM_INVALIDATION_KEY_CHANGED = 6_i32
  integer(i32), parameter :: MM_INVALIDATION_ARTIFACT_CHANGED = 7_i32

  integer(i32), parameter :: INITIAL_MM_CACHE_CAPACITY = 16_i32

  type :: mm_cache_record
    type(multimodal_cache_key)      :: key
    integer(i32)                    :: cache_kind = MM_CACHE_KIND_UNKNOWN
    integer(i64)                    :: content_hash = 0_i64
    integer(i64)                    :: embedding_count = 0_i64
    character(len=MAX_CACHE_KEY_LEN) :: reuse_key_text = ""
    type(artifact_metadata_record)  :: artifact_metadata
    integer(i64)                    :: hit_count = 0_i64
    integer(i64)                    :: last_access_tick = 0_i64
    integer(i32)                    :: invalidation_reason = MM_INVALIDATION_NONE
    logical                         :: is_invalid = .false.
  end type mm_cache_record

  type :: runtime_mm_cache
    integer(i32) :: entry_count = 0_i32
    integer(i64) :: clock_tick = 0_i64
    type(mm_cache_record), allocatable :: entries(:)
  end type runtime_mm_cache

contains

  subroutine initialize_runtime_mm_cache(cache)
    type(runtime_mm_cache), intent(out) :: cache

    cache = runtime_mm_cache()
  end subroutine initialize_runtime_mm_cache

  subroutine reset_runtime_mm_cache(cache)
    type(runtime_mm_cache), intent(inout) :: cache

    if (allocated(cache%entries)) deallocate(cache%entries)
    cache%entry_count = 0_i32
    cache%clock_tick = 0_i64
  end subroutine reset_runtime_mm_cache

  subroutine build_preprocess_reuse_key(key, content_hash, reuse_key_text, status_code)
    type(multimodal_cache_key), intent(in) :: key
    integer(i64), intent(in)               :: content_hash
    character(len=*), intent(out)          :: reuse_key_text
    integer(i32), intent(out)              :: status_code

    reuse_key_text = ""
    if (.not. mm_cache_key_is_strict(key) .or. content_hash == 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    reuse_key_text = "mm_pre:v" // trim(i32_to_text(key%versions%schema_version)) // &
      ":abi=" // trim(i32_to_text(key%versions%abi_version)) // &
      ":model=" // trim(i64_to_text(key%logical_model_hash)) // &
      ":fam=" // trim(i32_to_text(key%model_family)) // &
      ":kind=" // trim(i32_to_text(key%modality_kind)) // &
      ":dtype=" // trim(i32_to_text(key%dtype)) // &
      ":bytes=" // trim(i64_to_text(key%byte_count)) // &
      ":slot=" // trim(key%slot_name) // &
      ":content=" // trim(i64_to_text(content_hash))
    status_code = MIZU_STATUS_OK
  end subroutine build_preprocess_reuse_key

  subroutine build_projector_output_reuse_key(key, content_hash, embedding_count, &
                                              reuse_key_text, status_code)
    type(multimodal_cache_key), intent(in) :: key
    integer(i64), intent(in)               :: content_hash
    integer(i64), intent(in)               :: embedding_count
    character(len=*), intent(out)          :: reuse_key_text
    integer(i32), intent(out)              :: status_code

    reuse_key_text = ""
    if (.not. mm_cache_key_is_strict(key) .or. content_hash == 0_i64 .or. &
        embedding_count <= 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    reuse_key_text = "mm_projector:v" // trim(i32_to_text(key%versions%schema_version)) // &
      ":abi=" // trim(i32_to_text(key%versions%abi_version)) // &
      ":planner=" // trim(i32_to_text(key%versions%planner_version)) // &
      ":model=" // trim(i64_to_text(key%logical_model_hash)) // &
      ":proj=" // trim(i64_to_text(key%projector_revision)) // &
      ":fam=" // trim(i32_to_text(key%model_family)) // &
      ":kind=" // trim(i32_to_text(key%modality_kind)) // &
      ":dtype=" // trim(i32_to_text(key%dtype)) // &
      ":bytes=" // trim(i64_to_text(key%byte_count)) // &
      ":device=" // trim(key%device_key) // &
      ":slot=" // trim(key%slot_name) // &
      ":content=" // trim(i64_to_text(content_hash)) // &
      ":emb=" // trim(i64_to_text(embedding_count))
    status_code = MIZU_STATUS_OK
  end subroutine build_projector_output_reuse_key

  subroutine record_mm_cache_entry(cache, key, cache_kind, content_hash, embedding_count, &
                                   artifact_metadata, status_code)
    type(runtime_mm_cache), intent(inout)        :: cache
    type(multimodal_cache_key), intent(in)       :: key
    integer(i32), intent(in)                     :: cache_kind
    integer(i64), intent(in)                     :: content_hash
    integer(i64), intent(in)                     :: embedding_count
    type(artifact_metadata_record), intent(in)   :: artifact_metadata
    integer(i32), intent(out)                    :: status_code
    character(len=MAX_CACHE_KEY_LEN)             :: reuse_key_text
    integer(i32)                                 :: entry_index
    integer(i64)                                 :: existing_hits

    call build_mm_reuse_key(key, cache_kind, content_hash, embedding_count, &
      reuse_key_text, status_code)
    if (status_code /= MIZU_STATUS_OK .or. &
        .not. metadata_matches_cache_kind(artifact_metadata, cache_kind)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    entry_index = ensure_entry_index(cache, trim(reuse_key_text))
    existing_hits = cache%entries(entry_index)%hit_count

    cache%entries(entry_index)%key = key
    cache%entries(entry_index)%cache_kind = cache_kind
    cache%entries(entry_index)%content_hash = content_hash
    cache%entries(entry_index)%embedding_count = embedding_count
    cache%entries(entry_index)%reuse_key_text = trim(reuse_key_text)
    cache%entries(entry_index)%artifact_metadata = artifact_metadata
    cache%entries(entry_index)%hit_count = existing_hits
    cache%entries(entry_index)%last_access_tick = next_cache_tick(cache)
    cache%entries(entry_index)%invalidation_reason = MM_INVALIDATION_NONE
    cache%entries(entry_index)%is_invalid = .false.

    status_code = MIZU_STATUS_OK
  end subroutine record_mm_cache_entry

  subroutine lookup_mm_cache_entry(cache, key, cache_kind, content_hash, embedding_count, &
                                   record, found)
    type(runtime_mm_cache), intent(inout)  :: cache
    type(multimodal_cache_key), intent(in) :: key
    integer(i32), intent(in)               :: cache_kind
    integer(i64), intent(in)               :: content_hash
    integer(i64), intent(in)               :: embedding_count
    type(mm_cache_record), intent(out)     :: record
    logical, intent(out)                   :: found
    character(len=MAX_CACHE_KEY_LEN)       :: reuse_key_text
    integer(i32)                           :: status_code
    integer(i32)                           :: entry_index

    record = mm_cache_record()
    found = .false.
    call build_mm_reuse_key(key, cache_kind, content_hash, embedding_count, &
      reuse_key_text, status_code)
    if (status_code /= MIZU_STATUS_OK) return

    entry_index = find_entry_index(cache, trim(reuse_key_text))
    if (entry_index <= 0_i32) return
    if (.not. mm_cache_record_is_reusable(cache%entries(entry_index), key, cache_kind, &
        content_hash, embedding_count)) return

    cache%entries(entry_index)%hit_count = cache%entries(entry_index)%hit_count + 1_i64
    cache%entries(entry_index)%last_access_tick = next_cache_tick(cache)
    record = cache%entries(entry_index)
    found = .true.
  end subroutine lookup_mm_cache_entry

  subroutine invalidate_mm_cache_entries(cache, key, cache_kind, content_hash, embedding_count, &
                                         artifact_metadata, invalidated_count)
    type(runtime_mm_cache), intent(inout)        :: cache
    type(multimodal_cache_key), intent(in)       :: key
    integer(i32), intent(in)                     :: cache_kind
    integer(i64), intent(in)                     :: content_hash
    integer(i64), intent(in)                     :: embedding_count
    type(artifact_metadata_record), intent(in)   :: artifact_metadata
    integer(i32), intent(out)                    :: invalidated_count
    integer(i32)                                 :: index
    integer(i32)                                 :: reason

    invalidated_count = 0_i32
    if (.not. allocated(cache%entries)) return
    if (.not. valid_cache_kind(cache_kind)) return
    if (.not. mm_cache_key_is_strict(key)) return

    do index = 1_i32, cache%entry_count
      if (cache%entries(index)%is_invalid) cycle
      if (.not. same_cache_family(cache%entries(index), key, cache_kind)) cycle

      reason = mm_cache_invalidation_reason(cache%entries(index), key, cache_kind, &
        content_hash, embedding_count, artifact_metadata)
      if (reason == MM_INVALIDATION_NONE) cycle

      cache%entries(index)%is_invalid = .true.
      cache%entries(index)%invalidation_reason = reason
      cache%entries(index)%last_access_tick = next_cache_tick(cache)
      invalidated_count = invalidated_count + 1_i32
    end do
  end subroutine invalidate_mm_cache_entries

  pure logical function mm_cache_key_is_strict(key) result(is_strict)
    type(multimodal_cache_key), intent(in) :: key

    is_strict = len_trim(key%key_text) > 0 .and. &
      key%versions%schema_version > 0_i32 .and. &
      key%versions%abi_version > 0_i32 .and. &
      key%logical_model_hash /= 0_i64 .and. &
      key%projector_revision /= 0_i64 .and. &
      key%modality_kind /= MIZU_MODALITY_KIND_UNKNOWN .and. &
      key%dtype /= MIZU_DTYPE_UNKNOWN .and. &
      key%byte_count > 0_i64 .and. &
      len_trim(key%device_key) > 0 .and. &
      len_trim(key%slot_name) > 0 .and. &
      index(key%key_text, "mm:v") == 1 .and. &
      index(key%key_text, ":abi=" // trim(i32_to_text(key%versions%abi_version))) > 0 .and. &
      index(key%key_text, ":planner=" // trim(i32_to_text(key%versions%planner_version))) > 0 .and. &
      index(key%key_text, ":model=" // trim(i64_to_text(key%logical_model_hash))) > 0 .and. &
      index(key%key_text, ":proj=" // trim(i64_to_text(key%projector_revision))) > 0 .and. &
      index(key%key_text, ":kind=" // trim(i32_to_text(key%modality_kind))) > 0 .and. &
      index(key%key_text, ":dtype=" // trim(i32_to_text(key%dtype))) > 0 .and. &
      index(key%key_text, ":bytes=" // trim(i64_to_text(key%byte_count))) > 0 .and. &
      index(key%key_text, ":device=" // trim(key%device_key)) > 0 .and. &
      index(key%key_text, ":slot=" // trim(key%slot_name)) > 0
  end function mm_cache_key_is_strict

  pure logical function mm_cache_record_is_reusable(record, key, cache_kind, content_hash, &
                                                    embedding_count) result(is_reusable)
    type(mm_cache_record), intent(in)       :: record
    type(multimodal_cache_key), intent(in)  :: key
    integer(i32), intent(in)                :: cache_kind
    integer(i64), intent(in)                :: content_hash
    integer(i64), intent(in)                :: embedding_count

    is_reusable = mm_cache_invalidation_reason(record, key, cache_kind, content_hash, &
      embedding_count, artifact_metadata_record()) == MM_INVALIDATION_NONE
  end function mm_cache_record_is_reusable

  pure integer(i32) function mm_cache_invalidation_reason(record, key, cache_kind, content_hash, &
                                                          embedding_count, artifact_metadata) &
                                                          result(reason)
    type(mm_cache_record), intent(in)         :: record
    type(multimodal_cache_key), intent(in)    :: key
    integer(i32), intent(in)                  :: cache_kind
    integer(i64), intent(in)                  :: content_hash
    integer(i64), intent(in)                  :: embedding_count
    type(artifact_metadata_record), intent(in) :: artifact_metadata

    if (record%is_invalid) then
      reason = MM_INVALIDATION_ENTRY_INVALID
      return
    end if

    if (record%cache_kind /= cache_kind .or. .not. valid_cache_kind(cache_kind)) then
      reason = MM_INVALIDATION_KIND_CHANGED
      return
    end if

    if (.not. mm_cache_key_is_strict(key)) then
      reason = MM_INVALIDATION_KEY_CHANGED
      return
    end if

    if (content_hash == 0_i64 .or. record%content_hash /= content_hash) then
      reason = MM_INVALIDATION_CONTENT_CHANGED
      return
    end if

    select case (cache_kind)
    case (MM_CACHE_KIND_PREPROCESS)
      if (.not. preprocessing_key_matches(record%key, key)) then
        reason = MM_INVALIDATION_KEY_CHANGED
        return
      end if
    case (MM_CACHE_KIND_PROJECTOR_OUTPUT)
      if (record%key%projector_revision /= key%projector_revision) then
        reason = MM_INVALIDATION_PROJECTOR_CHANGED
        return
      end if
      if (record%embedding_count /= embedding_count .or. embedding_count <= 0_i64) then
        reason = MM_INVALIDATION_EMBEDDING_CHANGED
        return
      end if
      if (.not. projector_output_key_matches(record%key, key)) then
        reason = MM_INVALIDATION_KEY_CHANGED
        return
      end if
    case default
      reason = MM_INVALIDATION_KIND_CHANGED
      return
    end select

    if (metadata_invalidates(record%artifact_metadata, artifact_metadata)) then
      reason = MM_INVALIDATION_ARTIFACT_CHANGED
      return
    end if

    reason = MM_INVALIDATION_NONE
  end function mm_cache_invalidation_reason

  subroutine build_mm_reuse_key(key, cache_kind, content_hash, embedding_count, reuse_key_text, &
                                status_code)
    type(multimodal_cache_key), intent(in) :: key
    integer(i32), intent(in)               :: cache_kind
    integer(i64), intent(in)               :: content_hash
    integer(i64), intent(in)               :: embedding_count
    character(len=*), intent(out)          :: reuse_key_text
    integer(i32), intent(out)              :: status_code

    reuse_key_text = ""
    select case (cache_kind)
    case (MM_CACHE_KIND_PREPROCESS)
      call build_preprocess_reuse_key(key, content_hash, reuse_key_text, status_code)
    case (MM_CACHE_KIND_PROJECTOR_OUTPUT)
      call build_projector_output_reuse_key(key, content_hash, embedding_count, &
        reuse_key_text, status_code)
    case default
      status_code = MIZU_STATUS_INVALID_ARGUMENT
    end select
  end subroutine build_mm_reuse_key

  pure logical function valid_cache_kind(cache_kind) result(is_valid)
    integer(i32), intent(in) :: cache_kind

    is_valid = cache_kind == MM_CACHE_KIND_PREPROCESS .or. &
      cache_kind == MM_CACHE_KIND_PROJECTOR_OUTPUT
  end function valid_cache_kind

  pure logical function metadata_matches_cache_kind(metadata, cache_kind) result(matches)
    type(artifact_metadata_record), intent(in) :: metadata
    integer(i32), intent(in)                   :: cache_kind

    matches = valid_cache_kind(cache_kind)
    if (.not. matches) return

    if (metadata%stage_kind /= MIZU_STAGE_NONE .and. &
        metadata%stage_kind /= MIZU_STAGE_PROJECTOR) matches = .false.
  end function metadata_matches_cache_kind

  pure logical function metadata_invalidates(record_metadata, candidate_metadata) result(invalidates)
    type(artifact_metadata_record), intent(in) :: record_metadata
    type(artifact_metadata_record), intent(in) :: candidate_metadata

    invalidates = .false.
    if (.not. artifact_metadata_is_defined(candidate_metadata)) return

    if (candidate_metadata%stage_kind /= MIZU_STAGE_NONE .and. &
        record_metadata%stage_kind /= MIZU_STAGE_NONE .and. &
        candidate_metadata%stage_kind /= record_metadata%stage_kind) invalidates = .true.
    if (candidate_metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
        record_metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
        candidate_metadata%backend_family /= record_metadata%backend_family) invalidates = .true.
    if (candidate_metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
        record_metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
        candidate_metadata%execution_route /= record_metadata%execution_route) invalidates = .true.
    if (candidate_metadata%payload_bytes > 0_i64 .and. record_metadata%payload_bytes > 0_i64 .and. &
        candidate_metadata%payload_bytes /= record_metadata%payload_bytes) invalidates = .true.
    if (len_trim(candidate_metadata%artifact_format) > 0 .and. &
        len_trim(record_metadata%artifact_format) > 0 .and. &
        trim(candidate_metadata%artifact_format) /= trim(record_metadata%artifact_format)) invalidates = .true.
    if (len_trim(candidate_metadata%payload_fingerprint) > 0 .and. &
        len_trim(record_metadata%payload_fingerprint) > 0 .and. &
        trim(candidate_metadata%payload_fingerprint) /= &
        trim(record_metadata%payload_fingerprint)) invalidates = .true.
    if (len_trim(candidate_metadata%payload_path) > 0 .and. &
        len_trim(record_metadata%payload_path) > 0 .and. &
        trim(candidate_metadata%payload_path) /= trim(record_metadata%payload_path)) invalidates = .true.
  end function metadata_invalidates

  pure logical function same_cache_family(record, key, cache_kind) result(matches)
    type(mm_cache_record), intent(in)       :: record
    type(multimodal_cache_key), intent(in)  :: key
    integer(i32), intent(in)                :: cache_kind

    matches = .false.
    if (record%cache_kind /= cache_kind) return
    if (.not. mm_cache_key_is_strict(key)) return

    select case (cache_kind)
    case (MM_CACHE_KIND_PREPROCESS)
      matches = preprocessing_key_matches(record%key, key)
    case (MM_CACHE_KIND_PROJECTOR_OUTPUT)
      matches = projector_family_matches(record%key, key)
    case default
      matches = .false.
    end select
  end function same_cache_family

  pure logical function preprocessing_key_matches(record_key, requested_key) result(matches)
    type(multimodal_cache_key), intent(in) :: record_key
    type(multimodal_cache_key), intent(in) :: requested_key

    matches = record_key%versions%schema_version == requested_key%versions%schema_version .and. &
      record_key%versions%abi_version == requested_key%versions%abi_version .and. &
      record_key%logical_model_hash == requested_key%logical_model_hash .and. &
      record_key%model_family == requested_key%model_family .and. &
      record_key%modality_kind == requested_key%modality_kind .and. &
      record_key%dtype == requested_key%dtype .and. &
      record_key%byte_count == requested_key%byte_count .and. &
      trim(record_key%slot_name) == trim(requested_key%slot_name)
  end function preprocessing_key_matches

  pure logical function projector_family_matches(record_key, requested_key) result(matches)
    type(multimodal_cache_key), intent(in) :: record_key
    type(multimodal_cache_key), intent(in) :: requested_key

    matches = record_key%versions%schema_version == requested_key%versions%schema_version .and. &
      record_key%versions%abi_version == requested_key%versions%abi_version .and. &
      record_key%logical_model_hash == requested_key%logical_model_hash .and. &
      record_key%model_family == requested_key%model_family .and. &
      record_key%modality_kind == requested_key%modality_kind .and. &
      record_key%dtype == requested_key%dtype .and. &
      record_key%byte_count == requested_key%byte_count .and. &
      trim(record_key%device_key) == trim(requested_key%device_key) .and. &
      trim(record_key%slot_name) == trim(requested_key%slot_name)
  end function projector_family_matches

  pure logical function projector_output_key_matches(record_key, requested_key) result(matches)
    type(multimodal_cache_key), intent(in) :: record_key
    type(multimodal_cache_key), intent(in) :: requested_key

    matches = projector_family_matches(record_key, requested_key) .and. &
      record_key%versions%planner_version == requested_key%versions%planner_version .and. &
      record_key%projector_revision == requested_key%projector_revision
  end function projector_output_key_matches

  integer(i32) function ensure_entry_index(cache, reuse_key_text) result(entry_index)
    type(runtime_mm_cache), intent(inout) :: cache
    character(len=*), intent(in)          :: reuse_key_text

    entry_index = find_entry_index(cache, reuse_key_text)
    if (entry_index > 0_i32) return

    call ensure_entry_capacity(cache, cache%entry_count + 1_i32)
    cache%entry_count = cache%entry_count + 1_i32
    entry_index = cache%entry_count
    cache%entries(entry_index) = mm_cache_record()
    cache%entries(entry_index)%reuse_key_text = trim(reuse_key_text)
  end function ensure_entry_index

  integer(i32) function find_entry_index(cache, reuse_key_text) result(entry_index)
    type(runtime_mm_cache), intent(in) :: cache
    character(len=*), intent(in)       :: reuse_key_text
    integer(i32)                       :: index

    entry_index = 0_i32
    if (len_trim(reuse_key_text) == 0) return
    if (.not. allocated(cache%entries)) return

    do index = 1_i32, cache%entry_count
      if (trim(cache%entries(index)%reuse_key_text) == trim(reuse_key_text)) then
        entry_index = index
        return
      end if
    end do
  end function find_entry_index

  subroutine ensure_entry_capacity(cache, required_capacity)
    type(runtime_mm_cache), intent(inout) :: cache
    integer(i32), intent(in)              :: required_capacity
    type(mm_cache_record), allocatable    :: resized_entries(:)
    integer(i32)                          :: new_capacity

    if (.not. allocated(cache%entries)) then
      allocate(cache%entries(max(INITIAL_MM_CACHE_CAPACITY, required_capacity)))
      cache%entries = mm_cache_record()
      return
    end if

    if (size(cache%entries) >= required_capacity) return

    new_capacity = max(required_capacity, int(size(cache%entries), kind=i32) * 2_i32)
    allocate(resized_entries(new_capacity))
    resized_entries = mm_cache_record()
    if (cache%entry_count > 0_i32) resized_entries(1:cache%entry_count) = &
      cache%entries(1:cache%entry_count)
    call move_alloc(resized_entries, cache%entries)
  end subroutine ensure_entry_capacity

  integer(i64) function next_cache_tick(cache) result(tick_value)
    type(runtime_mm_cache), intent(inout) :: cache

    cache%clock_tick = cache%clock_tick + 1_i64
    tick_value = cache%clock_tick
  end function next_cache_tick

  pure function i32_to_text(value) result(text)
    integer(i32), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i32_to_text

  pure function i64_to_text(value) result(text)
    integer(i64), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i64_to_text

end module mod_mm_cache
