module mod_plan_cache
  use mod_kinds,       only: i32, i64, MAX_TENSOR_RANK
  use mod_status,      only: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK
  use mod_types,       only: MIZU_BACKEND_FAMILY_NONE, MIZU_EXEC_ROUTE_NONE, MIZU_STAGE_NONE
  use mod_cache_keys,  only: MAX_CACHE_KEY_LEN, plan_cache_key
  use mod_cache_store, only: artifact_metadata_record

  implicit none

  private
  public :: plan_cache_record, runtime_plan_cache
  public :: initialize_runtime_plan_cache, reset_runtime_plan_cache
  public :: record_plan_cache_entry, lookup_plan_cache_entry
  public :: plan_cache_key_is_strict

  integer(i32), parameter :: INITIAL_PLAN_CACHE_CAPACITY = 16_i32

  type :: plan_cache_record
    type(plan_cache_key)             :: key
    integer(i64)                     :: plan_id = 0_i64
    character(len=MAX_CACHE_KEY_LEN) :: candidate_key_text = ""
    type(artifact_metadata_record)   :: artifact_metadata
    integer(i64)                     :: hit_count = 0_i64
  end type plan_cache_record

  type :: runtime_plan_cache
    integer(i32) :: entry_count = 0_i32
    type(plan_cache_record), allocatable :: entries(:)
  end type runtime_plan_cache

contains

  subroutine initialize_runtime_plan_cache(cache)
    type(runtime_plan_cache), intent(out) :: cache

    cache = runtime_plan_cache()
  end subroutine initialize_runtime_plan_cache

  subroutine reset_runtime_plan_cache(cache)
    type(runtime_plan_cache), intent(inout) :: cache

    if (allocated(cache%entries)) deallocate(cache%entries)
    cache%entry_count = 0_i32
  end subroutine reset_runtime_plan_cache

  subroutine record_plan_cache_entry(cache, key, plan_id, artifact_metadata, status_code, &
                                     candidate_key_text)
    type(runtime_plan_cache), intent(inout)       :: cache
    type(plan_cache_key), intent(in)              :: key
    integer(i64), intent(in)                      :: plan_id
    type(artifact_metadata_record), intent(in)    :: artifact_metadata
    integer(i32), intent(out)                     :: status_code
    character(len=*), intent(in), optional        :: candidate_key_text
    integer(i32)                                  :: entry_index
    integer(i64)                                  :: existing_hits

    if (.not. plan_cache_key_is_strict(key) .or. plan_id == 0_i64 .or. &
        .not. metadata_matches_key(artifact_metadata, key)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    entry_index = ensure_entry_index(cache, trim(key%key_text))
    existing_hits = cache%entries(entry_index)%hit_count

    cache%entries(entry_index)%key = key
    cache%entries(entry_index)%plan_id = plan_id
    cache%entries(entry_index)%artifact_metadata = artifact_metadata
    cache%entries(entry_index)%hit_count = existing_hits
    cache%entries(entry_index)%candidate_key_text = ""
    if (present(candidate_key_text)) then
      cache%entries(entry_index)%candidate_key_text = trim(candidate_key_text)
    end if

    status_code = MIZU_STATUS_OK
  end subroutine record_plan_cache_entry

  subroutine lookup_plan_cache_entry(cache, key, record, found)
    type(runtime_plan_cache), intent(inout) :: cache
    type(plan_cache_key), intent(in)        :: key
    type(plan_cache_record), intent(out)    :: record
    logical, intent(out)                    :: found
    integer(i32)                            :: entry_index

    record = plan_cache_record()
    found = .false.
    if (.not. plan_cache_key_is_strict(key)) return

    entry_index = find_entry_index(cache, trim(key%key_text))
    if (entry_index <= 0_i32) return

    cache%entries(entry_index)%hit_count = cache%entries(entry_index)%hit_count + 1_i64
    record = cache%entries(entry_index)
    found = .true.
  end subroutine lookup_plan_cache_entry

  pure logical function plan_cache_key_is_strict(key) result(is_strict)
    type(plan_cache_key), intent(in) :: key

    is_strict = len_trim(key%key_text) > 0 .and. &
      key%stage_kind /= MIZU_STAGE_NONE .and. &
      key%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
      key%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
      key%rank >= 0_i32 .and. key%rank <= MAX_TENSOR_RANK .and. &
      len_trim(key%device_key) > 0 .and. &
      len_trim(key%pack_format) > 0 .and. &
      index(key%key_text, "plan:v") == 1 .and. &
      index(key%key_text, ":stage=" // trim(i32_to_text(key%stage_kind))) > 0 .and. &
      index(key%key_text, ":backend=" // trim(i32_to_text(key%backend_family))) > 0 .and. &
      index(key%key_text, ":route=" // trim(i32_to_text(key%execution_route))) > 0 .and. &
      index(key%key_text, ":device=" // trim(key%device_key)) > 0 .and. &
      index(key%key_text, ":pack=" // trim(key%pack_format)) > 0 .and. &
      index(key%key_text, ":shape=") > 0
  end function plan_cache_key_is_strict

  pure logical function metadata_matches_key(metadata, key) result(matches)
    type(artifact_metadata_record), intent(in) :: metadata
    type(plan_cache_key), intent(in)           :: key

    matches = .true.
    if (metadata%stage_kind /= MIZU_STAGE_NONE .and. metadata%stage_kind /= key%stage_kind) matches = .false.
    if (metadata%backend_family /= MIZU_BACKEND_FAMILY_NONE .and. &
        metadata%backend_family /= key%backend_family) matches = .false.
    if (metadata%execution_route /= MIZU_EXEC_ROUTE_NONE .and. &
        metadata%execution_route /= key%execution_route) matches = .false.
  end function metadata_matches_key

  integer(i32) function ensure_entry_index(cache, key_text) result(entry_index)
    type(runtime_plan_cache), intent(inout) :: cache
    character(len=*), intent(in)            :: key_text

    entry_index = find_entry_index(cache, key_text)
    if (entry_index > 0_i32) return

    call ensure_entry_capacity(cache, cache%entry_count + 1_i32)
    cache%entry_count = cache%entry_count + 1_i32
    entry_index = cache%entry_count
    cache%entries(entry_index) = plan_cache_record()
    cache%entries(entry_index)%key%key_text = trim(key_text)
  end function ensure_entry_index

  integer(i32) function find_entry_index(cache, key_text) result(entry_index)
    type(runtime_plan_cache), intent(in) :: cache
    character(len=*), intent(in)         :: key_text
    integer(i32)                         :: index

    entry_index = 0_i32
    if (len_trim(key_text) == 0) return
    if (.not. allocated(cache%entries)) return

    do index = 1_i32, cache%entry_count
      if (trim(cache%entries(index)%key%key_text) == trim(key_text)) then
        entry_index = index
        return
      end if
    end do
  end function find_entry_index

  subroutine ensure_entry_capacity(cache, required_capacity)
    type(runtime_plan_cache), intent(inout) :: cache
    integer(i32), intent(in)                :: required_capacity
    type(plan_cache_record), allocatable    :: resized_entries(:)
    integer(i32)                            :: new_capacity

    if (.not. allocated(cache%entries)) then
      allocate(cache%entries(max(INITIAL_PLAN_CACHE_CAPACITY, required_capacity)))
      cache%entries = plan_cache_record()
      return
    end if

    if (size(cache%entries) >= required_capacity) return

    new_capacity = max(required_capacity, int(size(cache%entries), kind=i32) * 2_i32)
    allocate(resized_entries(new_capacity))
    resized_entries = plan_cache_record()
    if (cache%entry_count > 0_i32) resized_entries(1:cache%entry_count) = &
      cache%entries(1:cache%entry_count)
    call move_alloc(resized_entries, cache%entries)
  end subroutine ensure_entry_capacity

  pure function i32_to_text(value) result(text)
    integer(i32), intent(in) :: value
    character(len=32)        :: text

    write(text, "(I0)") value
  end function i32_to_text

end module mod_plan_cache
