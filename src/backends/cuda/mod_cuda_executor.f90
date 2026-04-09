module mod_cuda_executor
  use iso_c_binding,     only: c_ptr, c_null_ptr
  use mod_kinds,          only: i8, i32, i64, MAX_PATH_LEN
  use mod_status,         only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                                MIZU_STATUS_INVALID_STATE
  use mod_types,          only: MIZU_STOP_REASON_NONE, MIZU_STAGE_NONE, MIZU_STAGE_PREFILL, &
                                MIZU_STAGE_DECODE
  use mod_cuda_bridge,    only: launch_cuda_projector, launch_cuda_prefill, launch_cuda_decode
  use mod_model_manifest, only: hash_text64

  implicit none

  private
  public :: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode
  public :: cuda_context_bytes_are_valid, extract_cuda_context_lineage
  public :: extract_cuda_context_state_snapshot
  public :: extract_cuda_context_window_snapshot
  public :: extract_cuda_context_kv_lane_snapshot
  public :: extract_cuda_context_kv_layout_snapshot
  public :: extract_cuda_context_page_control_snapshot
  public :: extract_cuda_context_page_tensor_snapshot
  public :: extract_cuda_context_pack_usage_snapshot
  public :: extract_cuda_context_pack_dispatch_snapshot
  public :: extract_cuda_context_slot_snapshot

  integer(i32), parameter :: MAX_CUDA_PACK_DISPATCH_ENTRIES = 4_i32
  integer(i32), parameter :: MAX_CUDA_SPAN_SAMPLE_BYTES = 64_i32
  integer(i32), parameter :: MAX_CUDA_PACK_PAGE_WORDS = 8_i32
  integer(i32), parameter :: MAX_CUDA_PACK_TILE_BYTES = 32_i32

  type :: cuda_pack_usage_profile
    integer(i64) :: usage_hash = 0_i64
    integer(i64) :: usage_bytes = 0_i64
    integer(i64) :: first_pack_offset = 0_i64
    integer(i64) :: last_pack_offset = 0_i64
    integer(i64) :: last_pack_bytes = 0_i64
    integer(i32) :: usage_count = 0_i32
    integer(i64) :: entry_offsets(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i64
    integer(i64) :: entry_bytes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i64
    integer(i32) :: role_codes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i32
    integer(i32) :: layout_codes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i32
    integer(i64) :: entry_span_hashes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i64
    integer(i64) :: entry_span_bytes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i64
    integer(i32) :: entry_span_sample_sizes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i32
    integer(i8)  :: entry_span_samples(MAX_CUDA_SPAN_SAMPLE_BYTES, MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i8
    integer(i64) :: entry_page_hashes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i64
    integer(i32) :: entry_page_word_counts(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i32
    integer(i32) :: entry_page_words(MAX_CUDA_PACK_PAGE_WORDS, MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i32
    integer(i64) :: entry_tile_hashes(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i64
    integer(i32) :: entry_tile_byte_counts(MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i32
    integer(i8)  :: entry_tile_bytes(MAX_CUDA_PACK_TILE_BYTES, MAX_CUDA_PACK_DISPATCH_ENTRIES) = 0_i8
    character(len=MAX_PATH_LEN) :: entry_span_paths(MAX_CUDA_PACK_DISPATCH_ENTRIES) = ""
    character(len=MAX_PATH_LEN) :: span_root = ""
    logical      :: has_usage = .false.
  end type cuda_pack_usage_profile

contains

  subroutine execute_cuda_projector(cache_root, artifact_path, modal_byte_count, placeholder_count, &
                                    modal_content_hash, embedding_count, status_code, workspace_buffer, &
                                    workspace_bytes)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i64), intent(in)     :: modal_byte_count
    integer(i32), intent(in)     :: placeholder_count
    integer(i64), intent(in)     :: modal_content_hash
    integer(i64), intent(out)    :: embedding_count
    integer(i32), intent(out)    :: status_code
    type(c_ptr), intent(in), optional :: workspace_buffer
    integer(i64), intent(in), optional :: workspace_bytes
    character(len=1024)          :: payload_text
    integer(i64)                 :: payload_hash
    integer(i64)                 :: pack_dependency_hash
    integer(i64)                 :: workspace_bytes_local
    logical                      :: loaded_ok
    logical                      :: has_pack_dependency
    type(c_ptr)                  :: workspace_buffer_local

    embedding_count = 0_i64
    if (len_trim(cache_root) == 0 .or. len_trim(artifact_path) == 0) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call load_cuda_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    if (.not. loaded_ok) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    if (index(payload_text, "stage=2") <= 0) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    call extract_payload_pack_dependency_hash(payload_text, pack_dependency_hash, has_pack_dependency)
    payload_hash = combine_positive_hash64(positive_hash64(trim(payload_text)), modal_content_hash)
    if (has_pack_dependency) payload_hash = combine_positive_hash64(payload_hash, pack_dependency_hash)
    workspace_buffer_local = c_null_ptr
    workspace_bytes_local = 0_i64
    if (present(workspace_buffer)) workspace_buffer_local = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_local = workspace_bytes
    call launch_cuda_projector(payload_hash, max(0_i64, modal_byte_count), placeholder_count, embedding_count, &
      status_code, workspace_buffer_local, workspace_bytes_local)
  end subroutine execute_cuda_projector

  subroutine execute_cuda_prefill(cache_root, artifact_path, staged_tokens, staged_modal_count, &
                                  token_content_hash, modal_content_hash, consumed_token_count, status_code, &
                                  workspace_buffer, workspace_bytes, token_values, modal_bytes, &
                                  context_bytes, context_byte_count, context_artifact_hash)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i64), intent(in)     :: staged_tokens
    integer(i32), intent(in)     :: staged_modal_count
    integer(i64), intent(in)     :: token_content_hash
    integer(i64), intent(in)     :: modal_content_hash
    integer(i64), intent(out)    :: consumed_token_count
    integer(i32), intent(out)    :: status_code
    type(c_ptr), intent(in), optional :: workspace_buffer
    integer(i64), intent(in), optional :: workspace_bytes
    integer(i32), intent(in), optional :: token_values(:)
    integer(i8), intent(in), optional  :: modal_bytes(:)
    integer(i8), intent(out)           :: context_bytes(:)
    integer(i32), intent(out)          :: context_byte_count
    integer(i64), intent(out), optional :: context_artifact_hash
    character(len=1024)          :: payload_text
    integer(i64)                 :: artifact_hash
    integer(i64)                 :: payload_hash
    integer(i64)                 :: pack_dependency_hash
    integer(i64)                 :: workspace_bytes_local
    type(cuda_pack_usage_profile) :: pack_usage
    logical                      :: loaded_ok
    logical                      :: loaded_cached_spans
    logical                      :: has_pack_dependency
    type(c_ptr)                  :: workspace_buffer_local

    consumed_token_count = 0_i64
    if (present(context_artifact_hash)) context_artifact_hash = 0_i64
    if (len_trim(cache_root) == 0 .or. len_trim(artifact_path) == 0) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call load_cuda_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    if (.not. loaded_ok) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    if (index(payload_text, "stage=3") <= 0) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    if (present(token_values)) then
      if (int(size(token_values), kind=i64) /= max(0_i64, staged_tokens)) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
    end if
    if (present(modal_bytes)) then
      if (staged_modal_count <= 0_i32 .and. size(modal_bytes) > 0) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
    end if

    artifact_hash = positive_hash64(trim(payload_text))
    call extract_payload_pack_dependency_hash(payload_text, pack_dependency_hash, has_pack_dependency)
    call extract_payload_pack_usage_profile(payload_text, pack_usage)
    call hydrate_cached_pack_span_profile(cache_root, artifact_path, payload_text, pack_usage, loaded_cached_spans)
    if (.not. loaded_cached_spans) call hydrate_payload_pack_span_profile(pack_usage)
    if (has_pack_dependency) artifact_hash = combine_positive_hash64(artifact_hash, pack_dependency_hash)
    payload_hash = artifact_hash
    payload_hash = combine_positive_hash64(payload_hash, token_content_hash)
    payload_hash = combine_positive_hash64(payload_hash, modal_content_hash)
    workspace_buffer_local = c_null_ptr
    workspace_bytes_local = 0_i64
    if (present(workspace_buffer)) workspace_buffer_local = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_local = workspace_bytes
    call launch_cuda_prefill(payload_hash, artifact_hash, pack_usage%usage_hash, pack_usage%usage_bytes, &
      pack_usage%first_pack_offset, pack_usage%last_pack_offset, pack_usage%last_pack_bytes, &
      pack_usage%usage_count, pack_usage%entry_offsets, pack_usage%entry_bytes, pack_usage%role_codes, &
      pack_usage%layout_codes, pack_usage%entry_span_hashes, pack_usage%entry_span_bytes, &
      pack_usage%entry_page_hashes, pack_usage%entry_page_word_counts, pack_usage%entry_page_words, &
      pack_usage%entry_tile_hashes, pack_usage%entry_tile_byte_counts, pack_usage%entry_tile_bytes, &
      pack_usage%entry_span_sample_sizes, pack_usage%entry_span_samples, &
      max(0_i64, staged_tokens), staged_modal_count, &
      consumed_token_count, status_code, workspace_buffer_local, workspace_bytes_local, token_values, &
      modal_bytes, context_bytes, context_byte_count)
    if (present(context_artifact_hash)) context_artifact_hash = merge(artifact_hash, 0_i64, status_code == MIZU_STATUS_OK)
  end subroutine execute_cuda_prefill

  subroutine execute_cuda_decode(cache_root, artifact_path, kv_before, token_budget, emitted_token_count, &
                                 token_value, stop_reason, status_code, workspace_buffer, workspace_bytes, &
                                 context_bytes, context_byte_count, updated_context_bytes, &
                                 updated_context_byte_count, context_artifact_hash)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i64), intent(in)     :: kv_before
    integer(i64), intent(in)     :: token_budget
    integer(i64), intent(out)    :: emitted_token_count
    integer(i32), intent(out)    :: token_value
    integer(i32), intent(out)    :: stop_reason
    integer(i32), intent(out)    :: status_code
    type(c_ptr), intent(in), optional :: workspace_buffer
    integer(i64), intent(in), optional :: workspace_bytes
    integer(i8), intent(in)            :: context_bytes(:)
    integer(i32), intent(in)           :: context_byte_count
    integer(i8), intent(out)           :: updated_context_bytes(:)
    integer(i32), intent(out)          :: updated_context_byte_count
    integer(i64), intent(out), optional :: context_artifact_hash
    character(len=1024)          :: payload_text
    integer(i64)                 :: artifact_hash
    integer(i64)                 :: payload_hash
    integer(i64)                 :: pack_dependency_hash
    integer(i64)                 :: workspace_bytes_local
    integer(i64)                 :: current_context_artifact_hash
    integer(i32)                 :: current_context_stage
    type(cuda_pack_usage_profile) :: pack_usage
    logical                      :: loaded_ok
    logical                      :: loaded_cached_spans
    logical                      :: lineage_known
    logical                      :: has_pack_dependency
    type(c_ptr)                  :: workspace_buffer_local

    emitted_token_count = 0_i64
    token_value = 0_i32
    stop_reason = MIZU_STOP_REASON_NONE
    current_context_stage = MIZU_STAGE_NONE
    current_context_artifact_hash = 0_i64
    lineage_known = .false.
    if (present(context_artifact_hash)) context_artifact_hash = 0_i64

    if (len_trim(cache_root) == 0 .or. len_trim(artifact_path) == 0 .or. token_budget <= 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call load_cuda_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    if (.not. loaded_ok) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    if (index(payload_text, "stage=4") <= 0) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    artifact_hash = positive_hash64(trim(payload_text))
    call extract_payload_pack_dependency_hash(payload_text, pack_dependency_hash, has_pack_dependency)
    call extract_payload_pack_usage_profile(payload_text, pack_usage)
    call hydrate_cached_pack_span_profile(cache_root, artifact_path, payload_text, pack_usage, loaded_cached_spans)
    if (.not. loaded_cached_spans) call hydrate_payload_pack_span_profile(pack_usage)
    if (has_pack_dependency) artifact_hash = combine_positive_hash64(artifact_hash, pack_dependency_hash)
    payload_hash = artifact_hash
    call extract_cuda_context_lineage(context_bytes, context_byte_count, current_context_stage, &
      current_context_artifact_hash, lineage_known)
    if (lineage_known .and. current_context_stage == MIZU_STAGE_DECODE) then
      if (current_context_artifact_hash /= artifact_hash) then
        status_code = MIZU_STATUS_INVALID_STATE
        return
      end if
    end if
    workspace_buffer_local = c_null_ptr
    workspace_bytes_local = 0_i64
    if (present(workspace_buffer)) workspace_buffer_local = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_local = workspace_bytes
    call launch_cuda_decode(payload_hash, artifact_hash, pack_usage%usage_hash, pack_usage%usage_bytes, &
      pack_usage%first_pack_offset, pack_usage%last_pack_offset, pack_usage%last_pack_bytes, &
      pack_usage%usage_count, pack_usage%entry_offsets, pack_usage%entry_bytes, pack_usage%role_codes, &
      pack_usage%layout_codes, pack_usage%entry_span_hashes, pack_usage%entry_span_bytes, &
      pack_usage%entry_page_hashes, pack_usage%entry_page_word_counts, pack_usage%entry_page_words, &
      pack_usage%entry_tile_hashes, pack_usage%entry_tile_byte_counts, pack_usage%entry_tile_bytes, &
      pack_usage%entry_span_sample_sizes, pack_usage%entry_span_samples, &
      kv_before, token_budget, emitted_token_count, token_value, &
      stop_reason, status_code, workspace_buffer_local, workspace_bytes_local, context_bytes, &
      context_byte_count, updated_context_bytes, updated_context_byte_count)
    if (present(context_artifact_hash)) context_artifact_hash = merge(artifact_hash, 0_i64, status_code == MIZU_STATUS_OK)
  end subroutine execute_cuda_decode

  subroutine load_cuda_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    character(len=*), intent(in)  :: cache_root
    character(len=*), intent(in)  :: artifact_path
    character(len=*), intent(out) :: payload_text
    logical, intent(out)          :: loaded_ok
    character(len=MAX_PATH_LEN)   :: full_path
    integer(i32)                  :: unit_id
    integer(i32)                  :: ios
    logical                       :: exists

    payload_text = ""
    loaded_ok = .false.

    full_path = join_cache_root_with_payload_path(cache_root, artifact_path)
    if (len_trim(full_path) == 0) return

    inquire(file=trim(full_path), exist=exists)
    if (.not. exists) return

    open(newunit=unit_id, file=trim(full_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) return
    read(unit_id, "(A)", iostat=ios) payload_text
    close(unit_id)
    if (ios /= 0_i32) return

    loaded_ok = (len_trim(payload_text) > 0)
  end subroutine load_cuda_artifact_payload

  function join_cache_root_with_payload_path(cache_root, artifact_path) result(full_path)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    character(len=MAX_PATH_LEN)  :: full_path
    integer                      :: root_len

    full_path = ""
    if (len_trim(cache_root) == 0 .or. len_trim(artifact_path) == 0) return

    root_len = len_trim(cache_root)
    if (cache_root(root_len:root_len) == "/") then
      full_path = trim(cache_root) // trim(artifact_path)
    else
      full_path = trim(cache_root) // "/" // trim(artifact_path)
    end if
  end function join_cache_root_with_payload_path

  function build_pack_span_cache_artifact_path(artifact_path) result(cache_path)
    character(len=*), intent(in) :: artifact_path
    character(len=MAX_PATH_LEN)  :: cache_path

    cache_path = ""
    if (len_trim(artifact_path) == 0) return
    write(cache_path, '(A,".spancache")') trim(artifact_path)
  end function build_pack_span_cache_artifact_path

  integer(i64) function positive_hash64(text) result(hash_value)
    character(len=*), intent(in) :: text

    hash_value = iand(hash_text64(text), int(z'7FFFFFFFFFFFFFFF', kind=i64))
    if (hash_value == 0_i64) hash_value = 1_i64
  end function positive_hash64

  integer(i64) function combine_positive_hash64(base_hash, content_hash) result(hash_value)
    integer(i64), intent(in) :: base_hash
    integer(i64), intent(in) :: content_hash
    integer(i64)             :: mixed_hash

    mixed_hash = ieor(max(1_i64, base_hash), content_hash + int(z'9E3779B97F4A7C15', kind=i64))
    mixed_hash = ieor(mixed_hash, shiftr(mixed_hash, 30))
    mixed_hash = mixed_hash * int(z'BF58476D1CE4E5B9', kind=i64)
    mixed_hash = ieor(mixed_hash, shiftr(mixed_hash, 27))
    mixed_hash = mixed_hash * int(z'94D049BB133111EB', kind=i64)
    hash_value = iand(ieor(mixed_hash, shiftr(mixed_hash, 31)), int(z'7FFFFFFFFFFFFFFF', kind=i64))
    if (hash_value == 0_i64) hash_value = 1_i64
  end function combine_positive_hash64

  subroutine extract_payload_pack_dependency_hash(payload_text, dependency_hash, has_dependency)
    character(len=*), intent(in) :: payload_text
    integer(i64), intent(out)    :: dependency_hash
    logical, intent(out)         :: has_dependency
    character(len=64)            :: pack_hash_text
    character(len=64)            :: pack_dispatch_hash_text
    character(len=64)            :: pack_bytes_text
    character(len=64)            :: pack_use_hash_text
    character(len=64)            :: pack_use_bytes_text
    character(len=64)            :: pack_use_count_text
    logical                      :: found_hash
    logical                      :: found_dispatch_hash
    logical                      :: found_bytes
    logical                      :: found_use_hash
    logical                      :: found_use_bytes
    logical                      :: found_use_count

    dependency_hash = 0_i64
    has_dependency = .false.
    pack_hash_text = ""
    pack_dispatch_hash_text = ""
    pack_bytes_text = ""
    pack_use_hash_text = ""
    pack_use_bytes_text = ""
    pack_use_count_text = ""

    call extract_payload_field_text(payload_text, "pack_ref_hash=", pack_hash_text, found_hash)
    if (.not. found_hash) call extract_payload_field_text(payload_text, "weight_pack_hash=", pack_hash_text, found_hash)
    call extract_payload_field_text(payload_text, "pack_dispatch_hash=", pack_dispatch_hash_text, found_dispatch_hash)
    call extract_payload_field_text(payload_text, "pack_ref_bytes=", pack_bytes_text, found_bytes)
    if (.not. found_bytes) call extract_payload_field_text(payload_text, "weight_pack_bytes=", pack_bytes_text, found_bytes)
    call extract_payload_field_text(payload_text, "pack_use_hash=", pack_use_hash_text, found_use_hash)
    call extract_payload_field_text(payload_text, "pack_use_bytes=", pack_use_bytes_text, found_use_bytes)
    call extract_payload_field_text(payload_text, "pack_use_count=", pack_use_count_text, found_use_count)

    if (found_hash) then
      dependency_hash = combine_positive_hash64(max(1_i64, dependency_hash), positive_hash64(trim(pack_hash_text)))
      has_dependency = .true.
    end if
    if (found_dispatch_hash) then
      dependency_hash = combine_positive_hash64(max(1_i64, dependency_hash), &
        positive_hash64(trim(pack_dispatch_hash_text)))
      has_dependency = .true.
    end if
    if (found_bytes) then
      dependency_hash = combine_positive_hash64(max(1_i64, dependency_hash), positive_hash64(trim(pack_bytes_text)))
      has_dependency = .true.
    end if
    if (found_use_hash) then
      dependency_hash = combine_positive_hash64(max(1_i64, dependency_hash), positive_hash64(trim(pack_use_hash_text)))
      has_dependency = .true.
    end if
    if (found_use_bytes) then
      dependency_hash = combine_positive_hash64(max(1_i64, dependency_hash), positive_hash64(trim(pack_use_bytes_text)))
      has_dependency = .true.
    end if
    if (found_use_count) then
      dependency_hash = combine_positive_hash64(max(1_i64, dependency_hash), positive_hash64(trim(pack_use_count_text)))
      has_dependency = .true.
    end if
  end subroutine extract_payload_pack_dependency_hash

  subroutine extract_payload_pack_usage_profile(payload_text, pack_usage)
    character(len=*), intent(in)        :: payload_text
    type(cuda_pack_usage_profile), intent(out) :: pack_usage
    character(len=32)                   :: field_key
    character(len=256)                  :: usage_entry
    integer(i32)                        :: entry_index
    integer(i64)                        :: entry_offset
    integer(i64)                        :: entry_bytes
    integer(i32)                        :: role_code
    integer(i32)                        :: layout_code
    logical                             :: found_compact
    logical                             :: found_entry
    logical                             :: parsed_ok

    pack_usage = cuda_pack_usage_profile()
    if (len_trim(payload_text) == 0) return
    call extract_compact_pack_usage_profile(payload_text, pack_usage, found_compact)
    if (found_compact) return
    pack_usage = cuda_pack_usage_profile()

    do entry_index = 1_i32, 16_i32
      write(field_key, '("pack_use",I0,"=")') entry_index
      usage_entry = ""
      call extract_payload_field_text(payload_text, trim(field_key), usage_entry, found_entry)
      if (.not. found_entry) exit
      call extract_payload_usage_entry(trim(usage_entry), entry_offset, entry_bytes, role_code, layout_code, parsed_ok)
      if (.not. parsed_ok) cycle

      pack_usage%usage_count = pack_usage%usage_count + 1_i32
      pack_usage%usage_bytes = pack_usage%usage_bytes + entry_bytes
      if (pack_usage%usage_count <= MAX_CUDA_PACK_DISPATCH_ENTRIES) then
        pack_usage%entry_offsets(pack_usage%usage_count) = entry_offset
        pack_usage%entry_bytes(pack_usage%usage_count) = entry_bytes
        pack_usage%role_codes(pack_usage%usage_count) = role_code
        pack_usage%layout_codes(pack_usage%usage_count) = layout_code
      end if
      if (pack_usage%usage_count == 1_i32) pack_usage%first_pack_offset = entry_offset
      pack_usage%last_pack_offset = entry_offset
      pack_usage%last_pack_bytes = entry_bytes
      pack_usage%usage_hash = combine_positive_hash64(max(1_i64, pack_usage%usage_hash), &
        positive_hash64(trim(usage_entry)))
    end do

    pack_usage%has_usage = (pack_usage%usage_count > 0_i32)
  end subroutine extract_payload_pack_usage_profile

  subroutine extract_compact_pack_usage_profile(payload_text, pack_usage, found_compact)
    character(len=*), intent(in)                 :: payload_text
    type(cuda_pack_usage_profile), intent(inout) :: pack_usage
    logical, intent(out)                         :: found_compact
    character(len=64)                            :: field_text
    character(len=128)                           :: entry_text
    character(len=MAX_PATH_LEN)                  :: path_text
    integer(i32)                                 :: entry_index
    integer(i32)                                 :: entry_limit
    integer(i64)                                 :: parsed_i64
    integer(i64)                                 :: entry_offset
    integer(i64)                                 :: entry_bytes
    integer(i32)                                 :: role_code
    integer(i32)                                 :: layout_code
    logical                                      :: found_count
    logical                                      :: found_bytes
    logical                                      :: found_hash
    logical                                      :: found_first_offset
    logical                                      :: found_last_offset
    logical                                      :: found_last_bytes
    logical                                      :: found_dispatch_count
    logical                                      :: found_span_root
    logical                                      :: found_path
    logical                                      :: found_sample_bytes
    logical                                      :: found_entry
    logical                                      :: has_compact_markers
    logical                                      :: parsed_ok

    found_compact = .false.
    has_compact_markers = .false.
    if (len_trim(payload_text) == 0) return

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_use_count=", field_text, found_count)
    if (found_count) then
      if (parse_i64_text(field_text, parsed_i64)) then
        pack_usage%usage_count = max(0_i32, int(parsed_i64, kind=i32))
      end if
    end if

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_use_bytes=", field_text, found_bytes)
    if (found_bytes) then
      if (parse_i64_text(field_text, parsed_i64)) then
        pack_usage%usage_bytes = max(0_i64, parsed_i64)
      end if
    end if

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_use_hash=", field_text, found_hash)
    if (found_hash) then
      pack_usage%usage_hash = positive_hash64(trim(field_text))
    end if

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_use_first_offset=", field_text, found_first_offset)
    if (found_first_offset) then
      if (parse_i64_text(field_text, parsed_i64)) then
        pack_usage%first_pack_offset = max(0_i64, parsed_i64)
        has_compact_markers = .true.
      end if
    end if

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_use_last_offset=", field_text, found_last_offset)
    if (found_last_offset) then
      if (parse_i64_text(field_text, parsed_i64)) then
        pack_usage%last_pack_offset = max(0_i64, parsed_i64)
        has_compact_markers = .true.
      end if
    end if

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_use_last_bytes=", field_text, found_last_bytes)
    if (found_last_bytes) then
      if (parse_i64_text(field_text, parsed_i64)) then
        pack_usage%last_pack_bytes = max(0_i64, parsed_i64)
        has_compact_markers = .true.
      end if
    end if

    field_text = ""
    call extract_payload_field_text(payload_text, "pack_dispatch_count=", field_text, found_dispatch_count)
    entry_limit = 0_i32
    if (found_dispatch_count) then
      if (parse_i64_text(field_text, parsed_i64)) then
        entry_limit = max(0_i32, min(MAX_CUDA_PACK_DISPATCH_ENTRIES, int(parsed_i64, kind=i32)))
        has_compact_markers = .true.
      end if
    end if

    pack_usage%span_root = ""
    path_text = ""
    call extract_payload_field_text(payload_text, "pack_span_root=", path_text, found_span_root)
    if (found_span_root) pack_usage%span_root = trim(path_text)

    do entry_index = 1_i32, entry_limit
      write(field_text, '("pack_dispatch",I0,"=")') entry_index
      entry_text = ""
      call extract_payload_field_text(payload_text, trim(field_text), entry_text, found_entry)
      if (.not. found_entry) exit
      call extract_payload_dispatch_entry(trim(entry_text), entry_offset, entry_bytes, role_code, layout_code, parsed_ok)
      if (.not. parsed_ok) cycle
      pack_usage%entry_offsets(entry_index) = entry_offset
      pack_usage%entry_bytes(entry_index) = entry_bytes
      pack_usage%role_codes(entry_index) = role_code
      pack_usage%layout_codes(entry_index) = layout_code

      if (len_trim(pack_usage%span_root) > 0) then
        write(field_text, '("pack_span",I0,"=")') entry_index
        entry_text = ""
        call extract_payload_field_text(payload_text, trim(field_text), entry_text, found_entry)
        if (found_entry) then
          path_text = ""
          call extract_pipe_field(trim(entry_text), 1_i32, path_text, found_path)
          field_text = ""
          call extract_inline_numeric_field(trim(entry_text), "sample_bytes=", field_text, found_sample_bytes)
          if (found_path) pack_usage%entry_span_paths(entry_index) = trim(path_text)
          if (found_sample_bytes) then
            if (parse_i64_text(field_text, parsed_i64)) then
              pack_usage%entry_span_bytes(entry_index) = max(0_i64, parsed_i64)
            end if
          end if
        end if
      end if
    end do

    if (pack_usage%usage_count > 0_i32 .and. pack_usage%usage_hash == 0_i64) then
      pack_usage%usage_hash = positive_hash64(trim(payload_text))
    end if
    found_compact = has_compact_markers
    pack_usage%has_usage = (pack_usage%usage_count > 0_i32)
  end subroutine extract_compact_pack_usage_profile

  subroutine hydrate_payload_pack_span_profile(pack_usage)
    type(cuda_pack_usage_profile), intent(inout) :: pack_usage
    integer(i32)                                 :: entry_index
    integer(i64)                                 :: span_hash
    integer(i64)                                 :: actual_sample_bytes
    integer(i8)                                  :: span_sample_bytes(MAX_CUDA_SPAN_SAMPLE_BYTES)
    integer(i64)                                 :: page_hash
    integer(i32)                                 :: page_word_count
    integer(i32)                                 :: page_words(MAX_CUDA_PACK_PAGE_WORDS)
    integer(i64)                                 :: tile_hash
    integer(i32)                                 :: tile_byte_count
    integer(i8)                                  :: tile_bytes(MAX_CUDA_PACK_TILE_BYTES)

    if (len_trim(pack_usage%span_root) == 0) return

    do entry_index = 1_i32, MAX_CUDA_PACK_DISPATCH_ENTRIES
      if (len_trim(pack_usage%entry_span_paths(entry_index)) == 0) cycle
      call resolve_import_span_record(trim(pack_usage%span_root), trim(pack_usage%entry_span_paths(entry_index)), &
        pack_usage%entry_span_bytes(entry_index), span_hash, actual_sample_bytes, span_sample_bytes)
      pack_usage%entry_span_hashes(entry_index) = span_hash
      if (actual_sample_bytes > 0_i64) then
        pack_usage%entry_span_bytes(entry_index) = actual_sample_bytes
        pack_usage%entry_span_sample_sizes(entry_index) = min(MAX_CUDA_SPAN_SAMPLE_BYTES, int(actual_sample_bytes, kind=i32))
        pack_usage%entry_span_samples(:, entry_index) = 0_i8
        if (pack_usage%entry_span_sample_sizes(entry_index) > 0_i32) then
          pack_usage%entry_span_samples(1:pack_usage%entry_span_sample_sizes(entry_index), entry_index) = &
            span_sample_bytes(1:pack_usage%entry_span_sample_sizes(entry_index))
        end if
      end if
      call build_cuda_pack_page_record(pack_usage%entry_offsets(entry_index), pack_usage%entry_bytes(entry_index), &
        pack_usage%role_codes(entry_index), pack_usage%layout_codes(entry_index), span_hash, &
        pack_usage%entry_span_samples(:, entry_index), int(pack_usage%entry_span_sample_sizes(entry_index), kind=i64), &
        page_hash, page_word_count, page_words)
      pack_usage%entry_page_hashes(entry_index) = page_hash
      pack_usage%entry_page_word_counts(entry_index) = page_word_count
      pack_usage%entry_page_words(:, entry_index) = 0_i32
      if (page_word_count > 0_i32) then
        pack_usage%entry_page_words(1:page_word_count, entry_index) = page_words(1:page_word_count)
      end if
      call build_cuda_pack_tile_record(pack_usage%entry_offsets(entry_index), pack_usage%entry_bytes(entry_index), &
        pack_usage%role_codes(entry_index), pack_usage%layout_codes(entry_index), &
        pack_usage%entry_span_samples(:, entry_index), int(pack_usage%entry_span_sample_sizes(entry_index), kind=i64), &
        tile_hash, tile_byte_count, tile_bytes)
      pack_usage%entry_tile_hashes(entry_index) = tile_hash
      pack_usage%entry_tile_byte_counts(entry_index) = tile_byte_count
      pack_usage%entry_tile_bytes(:, entry_index) = 0_i8
      if (tile_byte_count > 0_i32) then
        pack_usage%entry_tile_bytes(1:tile_byte_count, entry_index) = tile_bytes(1:tile_byte_count)
      end if
    end do
  end subroutine hydrate_payload_pack_span_profile

  subroutine hydrate_cached_pack_span_profile(cache_root, artifact_path, payload_text, pack_usage, loaded_cached)
    character(len=*), intent(in)                 :: cache_root
    character(len=*), intent(in)                 :: artifact_path
    character(len=*), intent(in)                 :: payload_text
    type(cuda_pack_usage_profile), intent(inout) :: pack_usage
    logical, intent(out)                         :: loaded_cached
    character(len=1024)                          :: cache_text
    character(len=MAX_PATH_LEN)                  :: cache_path
    character(len=64)                            :: key_text
    character(len=64)                            :: value_text
    integer(i32)                                 :: entry_index
    integer(i64)                                 :: parsed_i64
    logical                                      :: found_cache_path
    logical                                      :: found_hash
    logical                                      :: found_bytes
    logical                                      :: found_page_hash
    logical                                      :: found_page_words
    logical                                      :: found_page_hex
    logical                                      :: found_tile_hash
    logical                                      :: found_tile_bytes
    logical                                      :: found_tile_hex
    logical                                      :: loaded_ok
    logical                                      :: required_entry_found
    integer(i64)                                 :: page_hash
    integer(i32)                                 :: page_word_count
    integer(i32)                                 :: page_words(MAX_CUDA_PACK_PAGE_WORDS)
    integer(i64)                                 :: tile_hash
    integer(i32)                                 :: tile_byte_count
    integer(i8)                                  :: tile_bytes(MAX_CUDA_PACK_TILE_BYTES)

    loaded_cached = .false.
    if (len_trim(cache_root) == 0) return

    cache_path = ""
    call extract_payload_field_text(payload_text, "pack_span_cache=", cache_path, found_cache_path)
    if (.not. found_cache_path) cache_path = build_pack_span_cache_artifact_path(artifact_path)
    if (len_trim(cache_path) == 0) return

    call load_cuda_artifact_payload(cache_root, trim(cache_path), cache_text, loaded_ok)
    if (.not. loaded_ok) return
    if (index(cache_text, "kind=cuda_pack_span_cache_v1") <= 0 .and. &
        index(cache_text, "kind=cuda_pack_span_cache_v2") <= 0 .and. &
        index(cache_text, "kind=cuda_pack_span_cache_v3") <= 0) return

    required_entry_found = .false.
    do entry_index = 1_i32, MAX_CUDA_PACK_DISPATCH_ENTRIES
      if (len_trim(pack_usage%entry_span_paths(entry_index)) == 0) cycle
      required_entry_found = .true.

      write(key_text, '("entry",I0,"_hash=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_hash)
      if (.not. found_hash) return
      if (.not. parse_i64_text(value_text, parsed_i64)) return
      if (parsed_i64 <= 0_i64) return
      pack_usage%entry_span_hashes(entry_index) = parsed_i64

      write(key_text, '("entry",I0,"_bytes=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_bytes)
      if (found_bytes) then
        if (.not. parse_i64_text(value_text, parsed_i64)) return
        if (parsed_i64 > 0_i64) pack_usage%entry_span_bytes(entry_index) = parsed_i64
      end if

      write(key_text, '("entry",I0,"_sample_hex=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_bytes)
      if (found_bytes) then
        call decode_hex_to_span_bytes(trim(value_text), pack_usage%entry_span_samples(:, entry_index), &
          pack_usage%entry_span_sample_sizes(entry_index))
      end if

      write(key_text, '("entry",I0,"_page_hash=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_page_hash)
      page_hash = 0_i64
      if (found_page_hash) then
        if (.not. parse_i64_text(value_text, page_hash)) return
      end if

      write(key_text, '("entry",I0,"_page_words=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_page_words)
      page_word_count = 0_i32
      if (found_page_words) then
        if (.not. parse_i64_text(value_text, parsed_i64)) return
        page_word_count = max(0_i32, min(MAX_CUDA_PACK_PAGE_WORDS, int(parsed_i64, kind=i32)))
      end if

      write(key_text, '("entry",I0,"_page_hex=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_page_hex)
      page_words = 0_i32
      if (found_page_words .and. found_page_hex .and. page_word_count > 0_i32) then
        call decode_hex_to_page_words(trim(value_text), page_words, page_word_count)
      else if (pack_usage%entry_span_sample_sizes(entry_index) > 0_i32) then
        call build_cuda_pack_page_record(pack_usage%entry_offsets(entry_index), pack_usage%entry_bytes(entry_index), &
          pack_usage%role_codes(entry_index), pack_usage%layout_codes(entry_index), &
          pack_usage%entry_span_hashes(entry_index), pack_usage%entry_span_samples(:, entry_index), &
          int(pack_usage%entry_span_sample_sizes(entry_index), kind=i64), page_hash, page_word_count, page_words)
      end if
      pack_usage%entry_page_hashes(entry_index) = max(0_i64, page_hash)
      pack_usage%entry_page_word_counts(entry_index) = max(0_i32, page_word_count)
      pack_usage%entry_page_words(:, entry_index) = 0_i32
      if (pack_usage%entry_page_word_counts(entry_index) > 0_i32) then
        pack_usage%entry_page_words(1:pack_usage%entry_page_word_counts(entry_index), entry_index) = &
          page_words(1:pack_usage%entry_page_word_counts(entry_index))
      end if

      write(key_text, '("entry",I0,"_tile_hash=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_tile_hash)
      tile_hash = 0_i64
      if (found_tile_hash) then
        if (.not. parse_i64_text(value_text, tile_hash)) return
      end if

      write(key_text, '("entry",I0,"_tile_bytes=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_tile_bytes)
      tile_byte_count = 0_i32
      if (found_tile_bytes) then
        if (.not. parse_i64_text(value_text, parsed_i64)) return
        tile_byte_count = max(0_i32, min(MAX_CUDA_PACK_TILE_BYTES, int(parsed_i64, kind=i32)))
      end if

      write(key_text, '("entry",I0,"_tile_hex=")') entry_index
      value_text = ""
      call extract_payload_field_text(cache_text, trim(key_text), value_text, found_tile_hex)
      tile_bytes = 0_i8
      if (found_tile_bytes .and. found_tile_hex .and. tile_byte_count > 0_i32) then
        call decode_hex_to_tile_bytes(trim(value_text), tile_bytes, tile_byte_count)
      else if (pack_usage%entry_span_sample_sizes(entry_index) > 0_i32) then
        call build_cuda_pack_tile_record(pack_usage%entry_offsets(entry_index), pack_usage%entry_bytes(entry_index), &
          pack_usage%role_codes(entry_index), pack_usage%layout_codes(entry_index), &
          pack_usage%entry_span_samples(:, entry_index), int(pack_usage%entry_span_sample_sizes(entry_index), kind=i64), &
          tile_hash, tile_byte_count, tile_bytes)
      end if
      pack_usage%entry_tile_hashes(entry_index) = max(0_i64, tile_hash)
      pack_usage%entry_tile_byte_counts(entry_index) = max(0_i32, tile_byte_count)
      pack_usage%entry_tile_bytes(:, entry_index) = 0_i8
      if (pack_usage%entry_tile_byte_counts(entry_index) > 0_i32) then
        pack_usage%entry_tile_bytes(1:pack_usage%entry_tile_byte_counts(entry_index), entry_index) = &
          tile_bytes(1:pack_usage%entry_tile_byte_counts(entry_index))
      end if
    end do

    loaded_cached = required_entry_found
  end subroutine hydrate_cached_pack_span_profile

  subroutine extract_payload_usage_entry(usage_entry, pack_offset, pack_bytes, role_code, layout_code, parsed_ok)
    character(len=*), intent(in) :: usage_entry
    integer(i64), intent(out)    :: pack_offset
    integer(i64), intent(out)    :: pack_bytes
    integer(i32), intent(out)    :: role_code
    integer(i32), intent(out)    :: layout_code
    logical, intent(out)         :: parsed_ok
    character(len=64)            :: offset_text
    character(len=64)            :: bytes_text
    character(len=64)            :: role_text
    character(len=64)            :: layout_text
    logical                      :: found_offset
    logical                      :: found_bytes
    logical                      :: found_role
    logical                      :: found_layout

    pack_offset = 0_i64
    pack_bytes = 0_i64
    role_code = 0_i32
    layout_code = 0_i32
    parsed_ok = .false.
    if (len_trim(usage_entry) == 0) return

    call extract_pipe_field(usage_entry, 2_i32, role_text, found_role)
    call extract_inline_numeric_field(usage_entry, "offset=", offset_text, found_offset)
    call extract_inline_numeric_field(usage_entry, "bytes=", bytes_text, found_bytes)
    call extract_inline_numeric_field(usage_entry, "layout=", layout_text, found_layout)
    if (.not. found_offset .or. .not. found_bytes) return
    if (.not. parse_i64_text(offset_text, pack_offset)) return
    if (.not. parse_i64_text(bytes_text, pack_bytes)) return
    if (found_role) role_code = pack_role_code(trim(role_text))
    if (found_layout) layout_code = pack_layout_code(trim(layout_text))
    parsed_ok = (pack_bytes > 0_i64 .and. pack_offset >= 0_i64)
  end subroutine extract_payload_usage_entry

  subroutine extract_payload_dispatch_entry(dispatch_entry, pack_offset, pack_bytes, role_code, layout_code, parsed_ok)
    character(len=*), intent(in) :: dispatch_entry
    integer(i64), intent(out)    :: pack_offset
    integer(i64), intent(out)    :: pack_bytes
    integer(i32), intent(out)    :: role_code
    integer(i32), intent(out)    :: layout_code
    logical, intent(out)         :: parsed_ok
    character(len=64)            :: offset_text
    character(len=64)            :: bytes_text
    character(len=64)            :: role_text
    character(len=64)            :: layout_text
    integer(i64)                 :: parsed_i64
    logical                      :: found_offset
    logical                      :: found_bytes
    logical                      :: found_role
    logical                      :: found_layout

    pack_offset = 0_i64
    pack_bytes = 0_i64
    role_code = 0_i32
    layout_code = 0_i32
    parsed_ok = .false.
    if (len_trim(dispatch_entry) == 0) return

    call extract_inline_numeric_field(dispatch_entry, "offset=", offset_text, found_offset)
    call extract_inline_numeric_field(dispatch_entry, "bytes=", bytes_text, found_bytes)
    call extract_inline_numeric_field(dispatch_entry, "role=", role_text, found_role)
    call extract_inline_numeric_field(dispatch_entry, "layout=", layout_text, found_layout)
    if (.not. found_offset .or. .not. found_bytes .or. .not. found_role .or. .not. found_layout) return
    if (.not. parse_i64_text(offset_text, pack_offset)) return
    if (.not. parse_i64_text(bytes_text, pack_bytes)) return
    if (.not. parse_i64_text(role_text, parsed_i64)) return
    role_code = int(parsed_i64, kind=i32)
    if (.not. parse_i64_text(layout_text, parsed_i64)) return
    layout_code = int(parsed_i64, kind=i32)
    parsed_ok = (pack_bytes > 0_i64 .and. pack_offset >= 0_i64)
  end subroutine extract_payload_dispatch_entry

  subroutine resolve_import_span_record(span_root, span_path, requested_sample_bytes, span_hash, actual_sample_bytes, &
                                        sample_bytes)
    character(len=*), intent(in) :: span_root
    character(len=*), intent(in) :: span_path
    integer(i64), intent(in)     :: requested_sample_bytes
    integer(i64), intent(out)    :: span_hash
    integer(i64), intent(out)    :: actual_sample_bytes
    integer(i8), intent(out)     :: sample_bytes(:)
    character(len=MAX_PATH_LEN)  :: full_path
    integer(i32)                 :: unit_id
    integer(i32)                 :: ios
    integer(i64)                 :: sample_count
    integer(i64)                 :: file_size
    logical                      :: exists
    integer(i64)                 :: stored_count
    integer(i8), allocatable     :: sample_buffer(:)

    span_hash = 0_i64
    actual_sample_bytes = 0_i64
    sample_bytes = 0_i8
    if (len_trim(span_root) == 0 .or. len_trim(span_path) == 0) return

    full_path = join_import_span_path(span_root, span_path)
    span_hash = positive_hash64(trim(full_path))
    inquire(file=trim(full_path), exist=exists, size=file_size)
    if (.not. exists) return

    sample_count = max(0_i64, min(max(1_i64, requested_sample_bytes), max(0_i64, file_size)))
    if (sample_count <= 0_i64) return

    allocate(sample_buffer(sample_count))
    sample_buffer = 0_i8
    open(newunit=unit_id, file=trim(full_path), status="old", access="stream", form="unformatted", &
      action="read", iostat=ios)
    if (ios /= 0_i32) then
      deallocate(sample_buffer)
      return
    end if
    read(unit_id, iostat=ios) sample_buffer
    close(unit_id)
    if (ios /= 0_i32) then
      deallocate(sample_buffer)
      return
    end if

    actual_sample_bytes = sample_count
    span_hash = combine_positive_hash64(max(1_i64, span_hash), hash_i8_buffer64(sample_buffer, sample_count))
    stored_count = min(sample_count, int(size(sample_bytes), kind=i64))
    if (stored_count > 0_i64) sample_bytes(1:stored_count) = sample_buffer(1:stored_count)
    deallocate(sample_buffer)
  end subroutine resolve_import_span_record

  subroutine decode_hex_to_span_bytes(hex_text, span_bytes, stored_count)
    character(len=*), intent(in) :: hex_text
    integer(i8), intent(out)     :: span_bytes(:)
    integer(i32), intent(out)    :: stored_count
    integer(i32)                 :: decode_index
    integer(i32)                 :: byte_value
    integer(i32)                 :: high_nibble
    integer(i32)                 :: low_nibble
    integer(i32)                 :: hex_count

    span_bytes = 0_i8
    stored_count = 0_i32
    if (len_trim(hex_text) <= 1) return

    hex_count = min(len_trim(hex_text) / 2_i32, size(span_bytes))
    do decode_index = 1_i32, hex_count
      high_nibble = hex_digit_value(hex_text((2 * decode_index) - 1:(2 * decode_index) - 1))
      low_nibble = hex_digit_value(hex_text(2 * decode_index:2 * decode_index))
      if (high_nibble < 0_i32 .or. low_nibble < 0_i32) exit
      byte_value = (16_i32 * high_nibble) + low_nibble
      if (byte_value > 127_i32) then
        span_bytes(decode_index) = int(byte_value - 256_i32, kind=i8)
      else
        span_bytes(decode_index) = int(byte_value, kind=i8)
      end if
      stored_count = decode_index
    end do
  end subroutine decode_hex_to_span_bytes

  subroutine build_cuda_pack_page_record(pack_offset, pack_bytes, role_code, layout_code, span_hash, &
                                         sample_bytes, actual_sample_bytes, page_hash, page_word_count, page_words)
    integer(i64), intent(in) :: pack_offset
    integer(i64), intent(in) :: pack_bytes
    integer(i32), intent(in) :: role_code
    integer(i32), intent(in) :: layout_code
    integer(i64), intent(in) :: span_hash
    integer(i8), intent(in)  :: sample_bytes(:)
    integer(i64), intent(in) :: actual_sample_bytes
    integer(i64), intent(out) :: page_hash
    integer(i32), intent(out) :: page_word_count
    integer(i32), intent(out) :: page_words(:)
    integer(i32)              :: stored_sample_bytes
    integer(i32)              :: preview_word_1
    integer(i32)              :: preview_word_2
    integer(i32)              :: preview_word_3
    integer(i32)              :: preview_word_4
    integer(i32)              :: control_word
    integer(i32)              :: word_index

    page_hash = 0_i64
    page_word_count = 0_i32
    page_words = 0_i32
    if (span_hash <= 0_i64 .or. pack_bytes <= 0_i64) return

    stored_sample_bytes = int(max(0_i64, min(actual_sample_bytes, int(size(sample_bytes), kind=i64))), kind=i32)
    preview_word_1 = pack_sample_word(sample_bytes, stored_sample_bytes, 1_i32)
    preview_word_2 = pack_sample_word(sample_bytes, stored_sample_bytes, 2_i32)
    preview_word_3 = pack_sample_word(sample_bytes, stored_sample_bytes, 3_i32)
    preview_word_4 = pack_sample_word(sample_bytes, stored_sample_bytes, 4_i32)
    control_word = ior(iand(role_code, int(z'000000FF', kind=i32)), &
      ishft(iand(layout_code, int(z'000000FF', kind=i32)), 8))
    control_word = ior(control_word, ishft(iand(stored_sample_bytes, int(z'000000FF', kind=i32)), 16))
    control_word = ior(control_word, ishft(4_i32, 24))

    page_word_count = min(MAX_CUDA_PACK_PAGE_WORDS, int(size(page_words), kind=i32))
    if (page_word_count < 8_i32) return

    page_words(1) = int(iand(pack_offset, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(2) = int(iand(pack_bytes, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(3) = control_word
    page_words(4) = int(iand(span_hash, int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(5) = int(iand(shiftr(span_hash, 32), int(z'FFFFFFFF', kind=i64)), kind=i32)
    page_words(6) = preview_word_1
    page_words(7) = preview_word_2
    page_words(8) = ieor(preview_word_3, ishft(preview_word_4, 1))

    page_hash = max(1_i64, span_hash)
    do word_index = 1_i32, 8_i32
      page_hash = combine_positive_hash64(max(1_i64, page_hash), &
        iand(int(page_words(word_index), kind=i64), int(z'FFFFFFFF', kind=i64)))
    end do
  end subroutine build_cuda_pack_page_record

  pure integer(i32) function pack_sample_word(sample_bytes, stored_sample_bytes, word_index) result(word_value)
    integer(i8), intent(in)  :: sample_bytes(:)
    integer(i32), intent(in) :: stored_sample_bytes
    integer(i32), intent(in) :: word_index
    integer(i32)             :: byte_offset
    integer(i32)             :: byte_index
    integer(i32)             :: byte_value

    word_value = 0_i32
    if (word_index <= 0_i32) return

    byte_offset = (word_index - 1_i32) * 4_i32
    do byte_index = 0_i32, 3_i32
      if ((byte_offset + byte_index + 1_i32) > stored_sample_bytes) exit
      if ((byte_offset + byte_index + 1_i32) > int(size(sample_bytes), kind=i32)) exit
      byte_value = int(sample_bytes(byte_offset + byte_index + 1_i32), kind=i32)
      if (byte_value < 0_i32) byte_value = byte_value + 256_i32
      word_value = ior(word_value, ishft(byte_value, 8 * byte_index))
    end do
  end function pack_sample_word

  subroutine decode_hex_to_page_words(hex_text, page_words, stored_count)
    character(len=*), intent(in) :: hex_text
    integer(i32), intent(out)    :: page_words(:)
    integer(i32), intent(inout)  :: stored_count
    integer(i8)                  :: packed_bytes(MAX_CUDA_PACK_PAGE_WORDS * 4)
    integer(i32)                 :: byte_count
    integer(i32)                 :: word_index

    page_words = 0_i32
    packed_bytes = 0_i8
    if (stored_count <= 0_i32) return

    call decode_hex_to_span_bytes(hex_text, packed_bytes, byte_count)
    stored_count = max(0_i32, min(stored_count, min(int(size(page_words), kind=i32), byte_count / 4_i32)))
    do word_index = 1_i32, stored_count
      page_words(word_index) = unpack_le_i32(packed_bytes(((word_index - 1_i32) * 4_i32) + 1: &
        ((word_index - 1_i32) * 4_i32) + 4_i32))
    end do
  end subroutine decode_hex_to_page_words

  subroutine build_cuda_pack_tile_record(pack_offset, pack_bytes, role_code, layout_code, sample_bytes, &
                                         actual_sample_bytes, tile_hash, tile_byte_count, tile_bytes)
    integer(i64), intent(in) :: pack_offset
    integer(i64), intent(in) :: pack_bytes
    integer(i32), intent(in) :: role_code
    integer(i32), intent(in) :: layout_code
    integer(i8), intent(in)  :: sample_bytes(:)
    integer(i64), intent(in) :: actual_sample_bytes
    integer(i64), intent(out) :: tile_hash
    integer(i32), intent(out) :: tile_byte_count
    integer(i8), intent(out)  :: tile_bytes(:)
    integer(i32)              :: stored_sample_bytes
    integer(i32)              :: byte_index
    integer(i32)              :: source_index
    integer(i32)              :: start_index

    tile_hash = 0_i64
    tile_byte_count = 0_i32
    tile_bytes = 0_i8
    if (pack_bytes <= 0_i64) return

    stored_sample_bytes = int(max(0_i64, min(actual_sample_bytes, int(size(sample_bytes), kind=i64))), kind=i32)
    if (stored_sample_bytes <= 0_i32) return

    tile_byte_count = min(MAX_CUDA_PACK_TILE_BYTES, min(stored_sample_bytes, int(size(tile_bytes), kind=i32)))
    if (tile_byte_count <= 0_i32) return

    start_index = 1_i32 + mod(int(modulo(pack_offset, int(max(1_i32, stored_sample_bytes), kind=i64)), kind=i32), &
      max(1_i32, stored_sample_bytes))
    do byte_index = 1_i32, tile_byte_count
      source_index = select_cuda_pack_tile_source_index(layout_code, byte_index, stored_sample_bytes, start_index)
      tile_bytes(byte_index) = sample_bytes(source_index)
    end do

    tile_hash = positive_hash64("cuda_pack_tile")
    tile_hash = combine_positive_hash64(tile_hash, iand(pack_offset, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    tile_hash = combine_positive_hash64(tile_hash, iand(pack_bytes, int(z'7FFFFFFFFFFFFFFF', kind=i64)))
    tile_hash = combine_positive_hash64(tile_hash, int(role_code, kind=i64) + 257_i64)
    tile_hash = combine_positive_hash64(tile_hash, int(layout_code, kind=i64) + 1025_i64)
    tile_hash = combine_positive_hash64(tile_hash, hash_i8_buffer64(tile_bytes, int(tile_byte_count, kind=i64)))
  end subroutine build_cuda_pack_tile_record

  pure integer(i32) function select_cuda_pack_tile_source_index(layout_code, byte_index, stored_sample_bytes, &
                                                                start_index) result(source_index)
    integer(i32), intent(in) :: layout_code
    integer(i32), intent(in) :: byte_index
    integer(i32), intent(in) :: stored_sample_bytes
    integer(i32), intent(in) :: start_index
    integer(i32)             :: half_count
    integer(i32)             :: local_index

    source_index = 1_i32
    if (stored_sample_bytes <= 0_i32) return

    select case (layout_code)
    case (2_i32)
      half_count = max(1_i32, (stored_sample_bytes + 1_i32) / 2_i32)
      if (mod(byte_index, 2_i32) == 1_i32) then
        local_index = (byte_index + 1_i32) / 2_i32
      else
        local_index = half_count + (byte_index / 2_i32)
      end if
    case (3_i32)
      local_index = 1_i32 + mod((byte_index - 1_i32) * 2_i32, stored_sample_bytes)
    case default
      local_index = byte_index
    end select

    source_index = 1_i32 + mod((start_index - 1_i32) + (local_index - 1_i32), stored_sample_bytes)
  end function select_cuda_pack_tile_source_index

  subroutine decode_hex_to_tile_bytes(hex_text, tile_bytes, stored_count)
    character(len=*), intent(in) :: hex_text
    integer(i8), intent(out)     :: tile_bytes(:)
    integer(i32), intent(inout)  :: stored_count
    integer(i32)                 :: decoded_count

    decoded_count = 0_i32
    call decode_hex_to_span_bytes(hex_text, tile_bytes, decoded_count)
    stored_count = max(0_i32, min(stored_count, decoded_count))
  end subroutine decode_hex_to_tile_bytes

  pure integer(i32) function unpack_le_i32(byte_values) result(word_value)
    integer(i8), intent(in) :: byte_values(:)
    integer(i32)            :: byte_index
    integer(i32)            :: byte_value

    word_value = 0_i32
    do byte_index = 1_i32, min(4_i32, int(size(byte_values), kind=i32))
      byte_value = int(byte_values(byte_index), kind=i32)
      if (byte_value < 0_i32) byte_value = byte_value + 256_i32
      word_value = ior(word_value, ishft(byte_value, 8 * (byte_index - 1_i32)))
    end do
  end function unpack_le_i32

  pure integer(i32) function hex_digit_value(hex_char) result(digit_value)
    character(len=*), intent(in) :: hex_char
    integer(i32)                 :: ascii_code

    digit_value = -1_i32
    if (len_trim(hex_char) <= 0) return

    ascii_code = iachar(hex_char(1:1))
    select case (ascii_code)
    case (iachar("0"):iachar("9"))
      digit_value = ascii_code - iachar("0")
    case (iachar("A"):iachar("F"))
      digit_value = 10_i32 + ascii_code - iachar("A")
    case (iachar("a"):iachar("f"))
      digit_value = 10_i32 + ascii_code - iachar("a")
    case default
      digit_value = -1_i32
    end select
  end function hex_digit_value

  pure function join_import_span_path(span_root, span_path) result(full_path)
    character(len=*), intent(in) :: span_root
    character(len=*), intent(in) :: span_path
    character(len=MAX_PATH_LEN)  :: full_path
    integer(i32)                 :: root_len

    full_path = ""
    if (len_trim(span_root) == 0 .or. len_trim(span_path) == 0) return
    if (span_path(1:1) == "/") then
      full_path = trim(span_path)
      return
    end if

    root_len = len_trim(span_root)
    if (span_root(root_len:root_len) == "/") then
      full_path = trim(span_root) // trim(span_path)
    else
      full_path = trim(span_root) // "/" // trim(span_path)
    end if
  end function join_import_span_path

  subroutine extract_pipe_field(source_text, field_index, value_text, found)
    character(len=*), intent(in)  :: source_text
    integer(i32), intent(in)      :: field_index
    character(len=*), intent(out) :: value_text
    logical, intent(out)          :: found
    integer(i32)                  :: start_index
    integer(i32)                  :: pipe_index
    integer(i32)                  :: current_field
    integer(i32)                  :: value_len

    value_text = ""
    found = .false.
    if (len_trim(source_text) == 0 .or. field_index <= 0_i32) return

    start_index = 1_i32
    current_field = 1_i32
    do
      pipe_index = index(source_text(start_index:len_trim(source_text)), "|")
      if (current_field == field_index) then
        if (pipe_index <= 0) then
          value_len = min(len_trim(source_text) - start_index + 1_i32, len(value_text))
        else
          value_len = min(pipe_index - 1_i32, len(value_text))
        end if
        if (value_len > 0_i32) then
          value_text(1:value_len) = source_text(start_index:start_index + value_len - 1_i32)
          found = (len_trim(value_text) > 0)
        end if
        return
      end if
      if (pipe_index <= 0) exit
      start_index = start_index + pipe_index
      current_field = current_field + 1_i32
      if (start_index > len_trim(source_text)) exit
    end do
  end subroutine extract_pipe_field

  pure integer(i32) function pack_role_code(role_text) result(role_code)
    character(len=*), intent(in) :: role_text

    select case (trim(role_text))
    case ("embedding_table")
      role_code = 1_i32
    case ("decoder_stack")
      role_code = 2_i32
    case ("normalization")
      role_code = 3_i32
    case ("token_projection")
      role_code = 4_i32
    case ("multimodal_projector")
      role_code = 5_i32
    case default
      role_code = 0_i32
    end select
  end function pack_role_code

  pure integer(i32) function pack_layout_code(layout_text) result(layout_code)
    character(len=*), intent(in) :: layout_text

    select case (trim(layout_text))
    case ("row_major")
      layout_code = 1_i32
    case ("packed")
      layout_code = 2_i32
    case ("vector")
      layout_code = 3_i32
    case default
      layout_code = 0_i32
    end select
  end function pack_layout_code

  integer(i64) function hash_i8_buffer64(buffer, buffer_count) result(hash_value)
    integer(i8), intent(in)  :: buffer(:)
    integer(i64), intent(in) :: buffer_count
    integer(i64)             :: index_byte

    hash_value = positive_hash64("cuda_import_span")
    if (buffer_count <= 0_i64) return
    do index_byte = 1_i64, min(buffer_count, int(size(buffer), kind=i64))
      hash_value = combine_positive_hash64(max(1_i64, hash_value), int(buffer(index_byte), kind=i64) + 257_i64)
    end do
  end function hash_i8_buffer64

  subroutine extract_inline_numeric_field(source_text, key_text, value_text, found)
    character(len=*), intent(in)  :: source_text
    character(len=*), intent(in)  :: key_text
    character(len=*), intent(out) :: value_text
    logical, intent(out)          :: found
    integer(i32)                  :: start_index
    integer(i32)                  :: value_start
    integer(i32)                  :: remaining_len
    integer(i32)                  :: separator_index
    integer(i32)                  :: copy_len

    value_text = ""
    found = .false.
    if (len_trim(source_text) == 0 .or. len_trim(key_text) == 0) return

    start_index = index(source_text, trim(key_text))
    if (start_index <= 0) return

    value_start = start_index + len_trim(key_text)
    if (value_start > len_trim(source_text)) return

    remaining_len = len_trim(source_text) - value_start + 1_i32
    separator_index = index(source_text(value_start:value_start + remaining_len - 1_i32), "|")
    if (separator_index <= 0) then
      copy_len = min(len_trim(source_text) - value_start + 1_i32, len(value_text))
    else
      copy_len = min(separator_index - 1_i32, len(value_text))
    end if
    if (copy_len <= 0) return

    value_text(1:copy_len) = source_text(value_start:value_start + copy_len - 1_i32)
    found = (len_trim(value_text) > 0)
  end subroutine extract_inline_numeric_field

  logical function parse_i64_text(text, value_out) result(parsed_ok)
    character(len=*), intent(in) :: text
    integer(i64), intent(out)    :: value_out
    integer(i32)                 :: ios

    value_out = 0_i64
    parsed_ok = .false.
    if (len_trim(text) == 0) return
    read(text, *, iostat=ios) value_out
    parsed_ok = (ios == 0_i32)
  end function parse_i64_text

  subroutine extract_payload_field_text(payload_text, key_text, value_text, found)
    character(len=*), intent(in)  :: payload_text
    character(len=*), intent(in)  :: key_text
    character(len=*), intent(out) :: value_text
    logical, intent(out)          :: found
    integer(i32)                  :: start_index
    integer(i32)                  :: value_start
    integer(i32)                  :: remaining_len
    integer(i32)                  :: separator_index
    integer(i32)                  :: copy_len

    value_text = ""
    found = .false.
    if (len_trim(payload_text) == 0 .or. len_trim(key_text) == 0) return

    start_index = index(payload_text, trim(key_text))
    if (start_index <= 0) return

    value_start = start_index + len_trim(key_text)
    if (value_start > len_trim(payload_text)) return

    remaining_len = len_trim(payload_text) - value_start + 1_i32
    separator_index = index(payload_text(value_start:value_start + remaining_len - 1_i32), ";")
    if (separator_index <= 0) then
      copy_len = min(len_trim(payload_text) - value_start + 1_i32, len(value_text))
    else
      copy_len = min(separator_index - 1_i32, len(value_text))
    end if
    if (copy_len <= 0) return

    value_text(1:copy_len) = payload_text(value_start:value_start + copy_len - 1_i32)
    found = (len_trim(value_text) > 0)
  end subroutine extract_payload_field_text

  pure logical function cuda_context_bytes_are_valid(context_bytes, context_byte_count) result(is_valid)
    integer(i8), intent(in) :: context_bytes(:)
    integer(i32), intent(in) :: context_byte_count
    integer(i32), parameter :: HEADER_SIZE = 16_i32
    integer(i32), parameter :: VERSION_1 = 1_i32
    integer(i32), parameter :: KIND_PREFILL = 1_i32
    integer(i32), parameter :: KIND_DECODE = 2_i32
    integer(i32)            :: stored_count
    integer(i64)            :: expected_checksum
    integer(i64)            :: actual_checksum

    is_valid = .false.
    if (context_byte_count < HEADER_SIZE) return
    if (int(size(context_bytes), kind=i32) < context_byte_count) return
    if (int(context_bytes(1), kind=i32) /= iachar("M")) return
    if (int(context_bytes(2), kind=i32) /= iachar("Z")) return
    if (int(context_bytes(3), kind=i32) /= iachar("C")) return
    if (int(context_bytes(4), kind=i32) /= iachar("T")) return
    if (int(context_bytes(5), kind=i32) /= VERSION_1) return
    if (int(context_bytes(6), kind=i32) /= KIND_PREFILL .and. &
        int(context_bytes(6), kind=i32) /= KIND_DECODE) return
    stored_count = decode_context_u16le(context_bytes(7), context_bytes(8))
    if (stored_count /= context_byte_count) return
    expected_checksum = decode_context_u32le(context_bytes(9), context_bytes(10), context_bytes(11), &
      context_bytes(12))
    actual_checksum = compute_context_checksum32(context_bytes, stored_count)
    if (actual_checksum /= expected_checksum) return
    is_valid = .true.
  end function cuda_context_bytes_are_valid

  pure subroutine extract_cuda_context_lineage(context_bytes, context_byte_count, producer_stage, artifact_hash, &
                                               lineage_known)
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: context_byte_count
    integer(i32), intent(out) :: producer_stage
    integer(i64), intent(out) :: artifact_hash
    logical, intent(out)      :: lineage_known
    integer(i32)              :: producer_kind

    producer_stage = MIZU_STAGE_NONE
    artifact_hash = 0_i64
    lineage_known = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return

    producer_kind = int(context_bytes(6), kind=i32)
    select case (producer_kind)
    case (1_i32)
      producer_stage = MIZU_STAGE_PREFILL
    case (2_i32)
      producer_stage = MIZU_STAGE_DECODE
    case default
      producer_stage = MIZU_STAGE_NONE
    end select
    if (context_byte_count < 56_i32) return

    artifact_hash = decode_context_u64le(context_bytes(49), context_bytes(50), context_bytes(51), context_bytes(52), &
      context_bytes(53), context_bytes(54), context_bytes(55), context_bytes(56))
    lineage_known = (producer_stage /= MIZU_STAGE_NONE .and. artifact_hash /= 0_i64)
  end subroutine extract_cuda_context_lineage

  pure subroutine extract_cuda_context_state_snapshot(context_bytes, context_byte_count, producer_stage, artifact_hash, &
                                                      token_digest, modal_digest, kv_token_count, decode_step_count, &
                                                      rolling_state_digest, summary_primary_count, &
                                                      summary_secondary_count, summary_control_a, &
                                                      summary_control_b, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: producer_stage
    integer(i64), intent(out) :: artifact_hash
    integer(i64), intent(out) :: token_digest
    integer(i64), intent(out) :: modal_digest
    integer(i64), intent(out) :: kv_token_count
    integer(i64), intent(out) :: decode_step_count
    integer(i64), intent(out) :: rolling_state_digest
    integer(i64), intent(out) :: summary_primary_count
    integer(i64), intent(out) :: summary_secondary_count
    integer(i32), intent(out) :: summary_control_a
    integer(i32), intent(out) :: summary_control_b
    logical, intent(out)      :: snapshot_valid
    integer(i64), parameter   :: MASK_16 = int(z'FFFF', kind=i64)
    integer(i64), parameter   :: MASK_32 = int(z'FFFFFFFF', kind=i64)
    integer(i32)              :: producer_kind
    integer(i64)              :: counter_word
    integer(i64)              :: summary_word

    producer_stage = MIZU_STAGE_NONE
    artifact_hash = 0_i64
    token_digest = 0_i64
    modal_digest = 0_i64
    kv_token_count = 0_i64
    decode_step_count = 0_i64
    rolling_state_digest = 0_i64
    summary_primary_count = 0_i64
    summary_secondary_count = 0_i64
    summary_control_a = 0_i32
    summary_control_b = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 64_i32) return

    producer_kind = int(context_bytes(6), kind=i32)
    select case (producer_kind)
    case (1_i32)
      producer_stage = MIZU_STAGE_PREFILL
    case (2_i32)
      producer_stage = MIZU_STAGE_DECODE
    case default
      producer_stage = MIZU_STAGE_NONE
    end select

    token_digest = decode_context_u64le(context_bytes(17), context_bytes(18), context_bytes(19), context_bytes(20), &
      context_bytes(21), context_bytes(22), context_bytes(23), context_bytes(24))
    modal_digest = decode_context_u64le(context_bytes(25), context_bytes(26), context_bytes(27), context_bytes(28), &
      context_bytes(29), context_bytes(30), context_bytes(31), context_bytes(32))
    counter_word = decode_context_u64le(context_bytes(33), context_bytes(34), context_bytes(35), context_bytes(36), &
      context_bytes(37), context_bytes(38), context_bytes(39), context_bytes(40))
    rolling_state_digest = decode_context_u64le(context_bytes(41), context_bytes(42), context_bytes(43), &
      context_bytes(44), context_bytes(45), context_bytes(46), context_bytes(47), context_bytes(48))
    artifact_hash = decode_context_u64le(context_bytes(49), context_bytes(50), context_bytes(51), context_bytes(52), &
      context_bytes(53), context_bytes(54), context_bytes(55), context_bytes(56))
    summary_word = decode_context_u64le(context_bytes(57), context_bytes(58), context_bytes(59), context_bytes(60), &
      context_bytes(61), context_bytes(62), context_bytes(63), context_bytes(64))

    kv_token_count = iand(counter_word, MASK_32)
    decode_step_count = iand(shiftr(counter_word, 32), MASK_32)
    summary_primary_count = iand(summary_word, MASK_16)
    summary_secondary_count = iand(shiftr(summary_word, 16), MASK_16)
    summary_control_a = int(iand(shiftr(summary_word, 32), MASK_16), kind=i32)
    summary_control_b = int(iand(shiftr(summary_word, 48), MASK_16), kind=i32)
    snapshot_valid = .true.
  end subroutine extract_cuda_context_state_snapshot

  pure subroutine extract_cuda_context_window_snapshot(context_bytes, context_byte_count, page_anchors, &
                                                       page_token_counts, page_kinds, current_page_index, &
                                                       valid_page_count, recent_tokens, recent_token_count, &
                                                       state_image_digest, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i64), intent(out) :: page_anchors(:)
    integer(i64), intent(out) :: page_token_counts(:)
    integer(i32), intent(out) :: page_kinds(:)
    integer(i32), intent(out) :: current_page_index
    integer(i32), intent(out) :: valid_page_count
    integer(i32), intent(out) :: recent_tokens(:)
    integer(i32), intent(out) :: recent_token_count
    integer(i64), intent(out) :: state_image_digest
    logical, intent(out)      :: snapshot_valid
    integer(i64), parameter   :: MASK_16 = int(z'FFFF', kind=i64)
    integer(i64)              :: page_word
    integer(i64)              :: window_meta
    integer(i32)              :: entry_index
    integer(i32)              :: page_limit
    integer(i32)              :: token_limit

    page_anchors = 0_i64
    page_token_counts = 0_i64
    page_kinds = 0_i32
    current_page_index = 0_i32
    valid_page_count = 0_i32
    recent_tokens = 0_i32
    recent_token_count = 0_i32
    state_image_digest = 0_i64
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 128_i32) return

    page_limit = min(4_i32, min(int(size(page_anchors), kind=i32), min(int(size(page_token_counts), kind=i32), &
      int(size(page_kinds), kind=i32))))
    do entry_index = 1_i32, page_limit
      page_word = decode_context_u64le(context_bytes(65 + ((entry_index - 1_i32) * 8)), &
        context_bytes(66 + ((entry_index - 1_i32) * 8)), context_bytes(67 + ((entry_index - 1_i32) * 8)), &
        context_bytes(68 + ((entry_index - 1_i32) * 8)), context_bytes(69 + ((entry_index - 1_i32) * 8)), &
        context_bytes(70 + ((entry_index - 1_i32) * 8)), context_bytes(71 + ((entry_index - 1_i32) * 8)), &
        context_bytes(72 + ((entry_index - 1_i32) * 8)))
      page_anchors(entry_index) = iand(page_word, MASK_16)
      page_token_counts(entry_index) = iand(shiftr(page_word, 16), MASK_16)
      page_kinds(entry_index) = int(iand(shiftr(page_word, 48), MASK_16), kind=i32)
    end do

    token_limit = min(4_i32, int(size(recent_tokens), kind=i32))
    do entry_index = 1_i32, token_limit
      recent_tokens(entry_index) = int(decode_context_u32le(context_bytes(97 + ((entry_index - 1_i32) * 4)), &
        context_bytes(98 + ((entry_index - 1_i32) * 4)), context_bytes(99 + ((entry_index - 1_i32) * 4)), &
        context_bytes(100 + ((entry_index - 1_i32) * 4))), kind=i32)
    end do

    window_meta = decode_context_u64le(context_bytes(113), context_bytes(114), context_bytes(115), context_bytes(116), &
      context_bytes(117), context_bytes(118), context_bytes(119), context_bytes(120))
    current_page_index = int(iand(window_meta, MASK_16), kind=i32)
    valid_page_count = int(iand(shiftr(window_meta, 16), MASK_16), kind=i32)
    recent_token_count = int(iand(shiftr(window_meta, 32), MASK_16), kind=i32)
    state_image_digest = decode_context_u64le(context_bytes(121), context_bytes(122), context_bytes(123), &
      context_bytes(124), context_bytes(125), context_bytes(126), context_bytes(127), context_bytes(128))
    snapshot_valid = .true.
  end subroutine extract_cuda_context_window_snapshot

  pure subroutine extract_cuda_context_slot_snapshot(context_bytes, context_byte_count, page_slot_tokens, &
                                                     snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: page_slot_tokens(:, :)
    logical, intent(out)      :: snapshot_valid
    integer(i32)              :: page_index
    integer(i32)              :: slot_index
    integer(i32)              :: page_limit
    integer(i32)              :: slot_limit
    integer(i32)              :: slot_offset

    page_slot_tokens = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 256_i32) return

    slot_limit = min(8_i32, int(size(page_slot_tokens, dim=1), kind=i32))
    page_limit = min(4_i32, int(size(page_slot_tokens, dim=2), kind=i32))
    do page_index = 1_i32, page_limit
      do slot_index = 1_i32, slot_limit
        slot_offset = 129_i32 + (((page_index - 1_i32) * 8_i32 + (slot_index - 1_i32)) * 4_i32)
        page_slot_tokens(slot_index, page_index) = int(decode_context_u32le(context_bytes(slot_offset), &
          context_bytes(slot_offset + 1_i32), context_bytes(slot_offset + 2_i32), &
          context_bytes(slot_offset + 3_i32)), kind=i32)
      end do
    end do
    snapshot_valid = .true.
  end subroutine extract_cuda_context_slot_snapshot

  pure subroutine extract_cuda_context_kv_lane_snapshot(context_bytes, context_byte_count, page_key_lanes, &
                                                        page_value_lanes, page_lane_digests, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: page_key_lanes(:, :)
    integer(i32), intent(out) :: page_value_lanes(:, :)
    integer(i64), intent(out) :: page_lane_digests(:)
    logical, intent(out)      :: snapshot_valid
    integer(i32)              :: page_index
    integer(i32)              :: slot_index
    integer(i32)              :: page_limit
    integer(i32)              :: slot_limit
    integer(i32)              :: lane_offset
    integer(i32)              :: digest_offset

    page_key_lanes = 0_i32
    page_value_lanes = 0_i32
    page_lane_digests = 0_i64
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 416_i32) return

    slot_limit = min(8_i32, min(int(size(page_key_lanes, dim=1), kind=i32), int(size(page_value_lanes, dim=1), kind=i32)))
    page_limit = min(4_i32, min(int(size(page_key_lanes, dim=2), kind=i32), min(int(size(page_value_lanes, dim=2), &
      kind=i32), int(size(page_lane_digests), kind=i32))))
    do page_index = 1_i32, page_limit
      do slot_index = 1_i32, slot_limit
        lane_offset = 129_i32 + (((page_index - 1_i32) * 8_i32 + (slot_index - 1_i32)) * 4_i32)
        page_key_lanes(slot_index, page_index) = int(decode_context_u32le(context_bytes(lane_offset), &
          context_bytes(lane_offset + 1_i32), context_bytes(lane_offset + 2_i32), &
          context_bytes(lane_offset + 3_i32)), kind=i32)
        lane_offset = 257_i32 + (((page_index - 1_i32) * 8_i32 + (slot_index - 1_i32)) * 4_i32)
        page_value_lanes(slot_index, page_index) = int(decode_context_u32le(context_bytes(lane_offset), &
          context_bytes(lane_offset + 1_i32), context_bytes(lane_offset + 2_i32), &
          context_bytes(lane_offset + 3_i32)), kind=i32)
      end do

      digest_offset = 385_i32 + ((page_index - 1_i32) * 8_i32)
      page_lane_digests(page_index) = decode_context_u64le(context_bytes(digest_offset), &
        context_bytes(digest_offset + 1_i32), context_bytes(digest_offset + 2_i32), &
        context_bytes(digest_offset + 3_i32), context_bytes(digest_offset + 4_i32), &
        context_bytes(digest_offset + 5_i32), context_bytes(digest_offset + 6_i32), &
        context_bytes(digest_offset + 7_i32))
    end do
    snapshot_valid = .true.
  end subroutine extract_cuda_context_kv_lane_snapshot

  pure subroutine extract_cuda_context_kv_layout_snapshot(context_bytes, context_byte_count, page_key_rows, &
                                                          page_key_lane_counts, page_value_rows, &
                                                          page_value_lane_counts, page_head_blocks, &
                                                          page_generations, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: page_key_rows(:)
    integer(i32), intent(out) :: page_key_lane_counts(:)
    integer(i32), intent(out) :: page_value_rows(:)
    integer(i32), intent(out) :: page_value_lane_counts(:)
    integer(i32), intent(out) :: page_head_blocks(:)
    integer(i32), intent(out) :: page_generations(:)
    logical, intent(out)      :: snapshot_valid
    integer(i32)              :: page_index
    integer(i32)              :: page_limit
    integer(i32)              :: layout_offset

    page_key_rows = 0_i32
    page_key_lane_counts = 0_i32
    page_value_rows = 0_i32
    page_value_lane_counts = 0_i32
    page_head_blocks = 0_i32
    page_generations = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 512_i32) return

    page_limit = min(4_i32, int(size(page_key_rows), kind=i32))
    page_limit = min(page_limit, int(size(page_key_lane_counts), kind=i32))
    page_limit = min(page_limit, int(size(page_value_rows), kind=i32))
    page_limit = min(page_limit, int(size(page_value_lane_counts), kind=i32))
    page_limit = min(page_limit, int(size(page_head_blocks), kind=i32))
    page_limit = min(page_limit, int(size(page_generations), kind=i32))
    do page_index = 1_i32, page_limit
      layout_offset = 417_i32 + ((page_index - 1_i32) * 24_i32)
      page_key_rows(page_index) = int(decode_context_u32le(context_bytes(layout_offset), &
        context_bytes(layout_offset + 1_i32), context_bytes(layout_offset + 2_i32), &
        context_bytes(layout_offset + 3_i32)), kind=i32)
      page_key_lane_counts(page_index) = int(decode_context_u32le(context_bytes(layout_offset + 4_i32), &
        context_bytes(layout_offset + 5_i32), context_bytes(layout_offset + 6_i32), &
        context_bytes(layout_offset + 7_i32)), kind=i32)
      page_value_rows(page_index) = int(decode_context_u32le(context_bytes(layout_offset + 8_i32), &
        context_bytes(layout_offset + 9_i32), context_bytes(layout_offset + 10_i32), &
        context_bytes(layout_offset + 11_i32)), kind=i32)
      page_value_lane_counts(page_index) = int(decode_context_u32le(context_bytes(layout_offset + 12_i32), &
        context_bytes(layout_offset + 13_i32), context_bytes(layout_offset + 14_i32), &
        context_bytes(layout_offset + 15_i32)), kind=i32)
      page_head_blocks(page_index) = int(decode_context_u32le(context_bytes(layout_offset + 16_i32), &
        context_bytes(layout_offset + 17_i32), context_bytes(layout_offset + 18_i32), &
        context_bytes(layout_offset + 19_i32)), kind=i32)
      page_generations(page_index) = int(decode_context_u32le(context_bytes(layout_offset + 20_i32), &
        context_bytes(layout_offset + 21_i32), context_bytes(layout_offset + 22_i32), &
        context_bytes(layout_offset + 23_i32)), kind=i32)
    end do
    snapshot_valid = .true.
  end subroutine extract_cuda_context_kv_layout_snapshot

  pure subroutine extract_cuda_context_page_control_snapshot(context_bytes, context_byte_count, page_owner_kinds, &
                                                             page_usable_capacities, page_committed_tokens, &
                                                             page_free_slots, page_epochs, page_recycle_epochs, &
                                                             page_logical_ids, page_flags, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: page_owner_kinds(:)
    integer(i32), intent(out) :: page_usable_capacities(:)
    integer(i32), intent(out) :: page_committed_tokens(:)
    integer(i32), intent(out) :: page_free_slots(:)
    integer(i32), intent(out) :: page_epochs(:)
    integer(i32), intent(out) :: page_recycle_epochs(:)
    integer(i32), intent(out) :: page_logical_ids(:)
    integer(i32), intent(out) :: page_flags(:)
    logical, intent(out)      :: snapshot_valid
    integer(i32)              :: page_index
    integer(i32)              :: page_limit
    integer(i32)              :: control_offset

    page_owner_kinds = 0_i32
    page_usable_capacities = 0_i32
    page_committed_tokens = 0_i32
    page_free_slots = 0_i32
    page_epochs = 0_i32
    page_recycle_epochs = 0_i32
    page_logical_ids = 0_i32
    page_flags = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 640_i32) return

    page_limit = min(4_i32, int(size(page_owner_kinds), kind=i32))
    page_limit = min(page_limit, int(size(page_usable_capacities), kind=i32))
    page_limit = min(page_limit, int(size(page_committed_tokens), kind=i32))
    page_limit = min(page_limit, int(size(page_free_slots), kind=i32))
    page_limit = min(page_limit, int(size(page_epochs), kind=i32))
    page_limit = min(page_limit, int(size(page_recycle_epochs), kind=i32))
    page_limit = min(page_limit, int(size(page_logical_ids), kind=i32))
    page_limit = min(page_limit, int(size(page_flags), kind=i32))
    do page_index = 1_i32, page_limit
      control_offset = 513_i32 + ((page_index - 1_i32) * 32_i32)
      page_owner_kinds(page_index) = int(decode_context_u32le(context_bytes(control_offset), &
        context_bytes(control_offset + 1_i32), context_bytes(control_offset + 2_i32), &
        context_bytes(control_offset + 3_i32)), kind=i32)
      page_usable_capacities(page_index) = int(decode_context_u32le(context_bytes(control_offset + 4_i32), &
        context_bytes(control_offset + 5_i32), context_bytes(control_offset + 6_i32), &
        context_bytes(control_offset + 7_i32)), kind=i32)
      page_committed_tokens(page_index) = int(decode_context_u32le(context_bytes(control_offset + 8_i32), &
        context_bytes(control_offset + 9_i32), context_bytes(control_offset + 10_i32), &
        context_bytes(control_offset + 11_i32)), kind=i32)
      page_free_slots(page_index) = int(decode_context_u32le(context_bytes(control_offset + 12_i32), &
        context_bytes(control_offset + 13_i32), context_bytes(control_offset + 14_i32), &
        context_bytes(control_offset + 15_i32)), kind=i32)
      page_epochs(page_index) = int(decode_context_u32le(context_bytes(control_offset + 16_i32), &
        context_bytes(control_offset + 17_i32), context_bytes(control_offset + 18_i32), &
        context_bytes(control_offset + 19_i32)), kind=i32)
      page_recycle_epochs(page_index) = int(decode_context_u32le(context_bytes(control_offset + 20_i32), &
        context_bytes(control_offset + 21_i32), context_bytes(control_offset + 22_i32), &
        context_bytes(control_offset + 23_i32)), kind=i32)
      page_logical_ids(page_index) = int(decode_context_u32le(context_bytes(control_offset + 24_i32), &
        context_bytes(control_offset + 25_i32), context_bytes(control_offset + 26_i32), &
        context_bytes(control_offset + 27_i32)), kind=i32)
      page_flags(page_index) = int(decode_context_u32le(context_bytes(control_offset + 28_i32), &
        context_bytes(control_offset + 29_i32), context_bytes(control_offset + 30_i32), &
        context_bytes(control_offset + 31_i32)), kind=i32)
    end do
    snapshot_valid = .true.
  end subroutine extract_cuda_context_page_control_snapshot

  pure subroutine extract_cuda_context_page_tensor_snapshot(context_bytes, context_byte_count, &
                                                            page_key_storage_offsets, page_key_committed_bytes, &
                                                            page_key_capacity_bytes, page_key_row_stride_bytes, &
                                                            page_value_storage_offsets, page_value_committed_bytes, &
                                                            page_value_capacity_bytes, page_value_row_stride_bytes, &
                                                            snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: page_key_storage_offsets(:)
    integer(i32), intent(out) :: page_key_committed_bytes(:)
    integer(i32), intent(out) :: page_key_capacity_bytes(:)
    integer(i32), intent(out) :: page_key_row_stride_bytes(:)
    integer(i32), intent(out) :: page_value_storage_offsets(:)
    integer(i32), intent(out) :: page_value_committed_bytes(:)
    integer(i32), intent(out) :: page_value_capacity_bytes(:)
    integer(i32), intent(out) :: page_value_row_stride_bytes(:)
    logical, intent(out)      :: snapshot_valid
    integer(i32)              :: page_index
    integer(i32)              :: page_limit
    integer(i32)              :: tensor_offset

    page_key_storage_offsets = 0_i32
    page_key_committed_bytes = 0_i32
    page_key_capacity_bytes = 0_i32
    page_key_row_stride_bytes = 0_i32
    page_value_storage_offsets = 0_i32
    page_value_committed_bytes = 0_i32
    page_value_capacity_bytes = 0_i32
    page_value_row_stride_bytes = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < 768_i32) return

    page_limit = min(4_i32, int(size(page_key_storage_offsets), kind=i32))
    page_limit = min(page_limit, int(size(page_key_committed_bytes), kind=i32))
    page_limit = min(page_limit, int(size(page_key_capacity_bytes), kind=i32))
    page_limit = min(page_limit, int(size(page_key_row_stride_bytes), kind=i32))
    page_limit = min(page_limit, int(size(page_value_storage_offsets), kind=i32))
    page_limit = min(page_limit, int(size(page_value_committed_bytes), kind=i32))
    page_limit = min(page_limit, int(size(page_value_capacity_bytes), kind=i32))
    page_limit = min(page_limit, int(size(page_value_row_stride_bytes), kind=i32))
    do page_index = 1_i32, page_limit
      tensor_offset = 641_i32 + ((page_index - 1_i32) * 32_i32)
      page_key_storage_offsets(page_index) = int(decode_context_u32le(context_bytes(tensor_offset), &
        context_bytes(tensor_offset + 1_i32), context_bytes(tensor_offset + 2_i32), &
        context_bytes(tensor_offset + 3_i32)), kind=i32)
      page_key_committed_bytes(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 4_i32), &
        context_bytes(tensor_offset + 5_i32), context_bytes(tensor_offset + 6_i32), &
        context_bytes(tensor_offset + 7_i32)), kind=i32)
      page_key_capacity_bytes(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 8_i32), &
        context_bytes(tensor_offset + 9_i32), context_bytes(tensor_offset + 10_i32), &
        context_bytes(tensor_offset + 11_i32)), kind=i32)
      page_key_row_stride_bytes(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 12_i32), &
        context_bytes(tensor_offset + 13_i32), context_bytes(tensor_offset + 14_i32), &
        context_bytes(tensor_offset + 15_i32)), kind=i32)
      page_value_storage_offsets(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 16_i32), &
        context_bytes(tensor_offset + 17_i32), context_bytes(tensor_offset + 18_i32), &
        context_bytes(tensor_offset + 19_i32)), kind=i32)
      page_value_committed_bytes(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 20_i32), &
        context_bytes(tensor_offset + 21_i32), context_bytes(tensor_offset + 22_i32), &
        context_bytes(tensor_offset + 23_i32)), kind=i32)
      page_value_capacity_bytes(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 24_i32), &
        context_bytes(tensor_offset + 25_i32), context_bytes(tensor_offset + 26_i32), &
        context_bytes(tensor_offset + 27_i32)), kind=i32)
      page_value_row_stride_bytes(page_index) = int(decode_context_u32le(context_bytes(tensor_offset + 28_i32), &
        context_bytes(tensor_offset + 29_i32), context_bytes(tensor_offset + 30_i32), &
        context_bytes(tensor_offset + 31_i32)), kind=i32)
    end do
    snapshot_valid = .true.
  end subroutine extract_cuda_context_page_tensor_snapshot

  pure subroutine extract_cuda_context_pack_usage_snapshot(context_bytes, context_byte_count, usage_hash, &
                                                           usage_bytes, first_pack_offset, last_pack_offset, &
                                                           last_pack_bytes, usage_count, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i64), intent(out) :: usage_hash
    integer(i64), intent(out) :: usage_bytes
    integer(i64), intent(out) :: first_pack_offset
    integer(i64), intent(out) :: last_pack_offset
    integer(i64), intent(out) :: last_pack_bytes
    integer(i32), intent(out) :: usage_count
    logical, intent(out)      :: snapshot_valid
    integer(i32), parameter   :: USAGE_PROFILE_OFFSET = 769_i32

    usage_hash = 0_i64
    usage_bytes = 0_i64
    first_pack_offset = 0_i64
    last_pack_offset = 0_i64
    last_pack_bytes = 0_i64
    usage_count = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < USAGE_PROFILE_OFFSET + 43_i32) return

    usage_hash = decode_context_u64at(context_bytes, USAGE_PROFILE_OFFSET)
    usage_bytes = decode_context_u64at(context_bytes, USAGE_PROFILE_OFFSET + 8_i32)
    first_pack_offset = decode_context_u64at(context_bytes, USAGE_PROFILE_OFFSET + 16_i32)
    last_pack_offset = decode_context_u64at(context_bytes, USAGE_PROFILE_OFFSET + 24_i32)
    last_pack_bytes = decode_context_u64at(context_bytes, USAGE_PROFILE_OFFSET + 32_i32)
    usage_count = decode_context_i32at(context_bytes, USAGE_PROFILE_OFFSET + 40_i32)
    snapshot_valid = .true.
  end subroutine extract_cuda_context_pack_usage_snapshot

  pure subroutine extract_cuda_context_pack_dispatch_snapshot(context_bytes, context_byte_count, entry_offsets, &
                                                              entry_bytes, role_codes, layout_codes, &
                                                              dispatch_count, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i64), intent(out) :: entry_offsets(:)
    integer(i64), intent(out) :: entry_bytes(:)
    integer(i32), intent(out) :: role_codes(:)
    integer(i32), intent(out) :: layout_codes(:)
    integer(i32), intent(out) :: dispatch_count
    logical, intent(out)      :: snapshot_valid
    integer(i32), parameter   :: DISPATCH_OFFSET = 817_i32
    integer(i32), parameter   :: DISPATCH_STRIDE = 24_i32
    integer(i32)              :: entry_index
    integer(i32)              :: entry_limit
    integer(i32)              :: offset_1based

    entry_offsets = 0_i64
    entry_bytes = 0_i64
    role_codes = 0_i32
    layout_codes = 0_i32
    dispatch_count = 0_i32
    snapshot_valid = .false.
    if (.not. cuda_context_bytes_are_valid(context_bytes, context_byte_count)) return
    if (context_byte_count < DISPATCH_OFFSET + (DISPATCH_STRIDE * MAX_CUDA_PACK_DISPATCH_ENTRIES) - 1_i32) return

    entry_limit = min(MAX_CUDA_PACK_DISPATCH_ENTRIES, int(size(entry_offsets), kind=i32))
    entry_limit = min(entry_limit, int(size(entry_bytes), kind=i32))
    entry_limit = min(entry_limit, int(size(role_codes), kind=i32))
    entry_limit = min(entry_limit, int(size(layout_codes), kind=i32))
    do entry_index = 1_i32, entry_limit
      offset_1based = DISPATCH_OFFSET + ((entry_index - 1_i32) * DISPATCH_STRIDE)
      entry_offsets(entry_index) = decode_context_u64at(context_bytes, offset_1based)
      entry_bytes(entry_index) = decode_context_u64at(context_bytes, offset_1based + 8_i32)
      role_codes(entry_index) = decode_context_i32at(context_bytes, offset_1based + 16_i32)
      layout_codes(entry_index) = decode_context_i32at(context_bytes, offset_1based + 20_i32)
      if (entry_bytes(entry_index) > 0_i64) dispatch_count = dispatch_count + 1_i32
    end do
    snapshot_valid = .true.
  end subroutine extract_cuda_context_pack_dispatch_snapshot

  pure integer(i32) function decode_context_u16le(byte_1, byte_2) result(value_u16)
    integer(i8), intent(in) :: byte_1
    integer(i8), intent(in) :: byte_2

    value_u16 = context_byte_to_u32(byte_1) + shiftl(context_byte_to_u32(byte_2), 8)
  end function decode_context_u16le

  pure integer(i64) function decode_context_u32le(byte_1, byte_2, byte_3, byte_4) result(value_u32)
    integer(i8), intent(in) :: byte_1
    integer(i8), intent(in) :: byte_2
    integer(i8), intent(in) :: byte_3
    integer(i8), intent(in) :: byte_4

    value_u32 = int(context_byte_to_u32(byte_1), kind=i64)
    value_u32 = value_u32 + shiftl(int(context_byte_to_u32(byte_2), kind=i64), 8)
    value_u32 = value_u32 + shiftl(int(context_byte_to_u32(byte_3), kind=i64), 16)
    value_u32 = value_u32 + shiftl(int(context_byte_to_u32(byte_4), kind=i64), 24)
  end function decode_context_u32le

  pure integer(i64) function decode_context_u64le(byte_1, byte_2, byte_3, byte_4, byte_5, byte_6, byte_7, &
                                                  byte_8) result(value_u64)
    integer(i8), intent(in) :: byte_1
    integer(i8), intent(in) :: byte_2
    integer(i8), intent(in) :: byte_3
    integer(i8), intent(in) :: byte_4
    integer(i8), intent(in) :: byte_5
    integer(i8), intent(in) :: byte_6
    integer(i8), intent(in) :: byte_7
    integer(i8), intent(in) :: byte_8

    value_u64 = int(context_byte_to_u32(byte_1), kind=i64)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_2), kind=i64), 8)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_3), kind=i64), 16)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_4), kind=i64), 24)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_5), kind=i64), 32)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_6), kind=i64), 40)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_7), kind=i64), 48)
    value_u64 = value_u64 + shiftl(int(context_byte_to_u32(byte_8), kind=i64), 56)
  end function decode_context_u64le

  pure integer(i64) function decode_context_u64at(context_bytes, offset_1based) result(value_u64)
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: offset_1based

    value_u64 = decode_context_u64le(context_bytes(offset_1based), context_bytes(offset_1based + 1_i32), &
      context_bytes(offset_1based + 2_i32), context_bytes(offset_1based + 3_i32), &
      context_bytes(offset_1based + 4_i32), context_bytes(offset_1based + 5_i32), &
      context_bytes(offset_1based + 6_i32), context_bytes(offset_1based + 7_i32))
  end function decode_context_u64at

  pure integer(i32) function decode_context_i32at(context_bytes, offset_1based) result(value_i32)
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: offset_1based

    value_i32 = int(decode_context_u32le(context_bytes(offset_1based), context_bytes(offset_1based + 1_i32), &
      context_bytes(offset_1based + 2_i32), context_bytes(offset_1based + 3_i32)), kind=i32)
  end function decode_context_i32at

  pure integer(i64) function compute_context_checksum32(context_bytes, stored_count) result(checksum_value)
    integer(i8), intent(in) :: context_bytes(:)
    integer(i32), intent(in) :: stored_count
    integer(i64), parameter :: OFFSET_BASIS = int(z'811C9DC5', kind=i64)
    integer(i64), parameter :: FNV_PRIME = int(z'01000193', kind=i64)
    integer(i64), parameter :: MASK_32 = int(z'FFFFFFFF', kind=i64)
    integer(i32)            :: byte_index

    checksum_value = OFFSET_BASIS
    if (stored_count <= 16_i32) then
      checksum_value = ieor(checksum_value, int(max(0_i32, stored_count), kind=i64))
      checksum_value = iand(checksum_value * FNV_PRIME, MASK_32)
      if (checksum_value == 0_i64) checksum_value = 1_i64
      return
    end if

    do byte_index = 17_i32, stored_count
      checksum_value = ieor(checksum_value, int(context_byte_to_u32(context_bytes(byte_index)), kind=i64))
      checksum_value = iand(checksum_value * FNV_PRIME, MASK_32)
    end do
    checksum_value = ieor(checksum_value, int(stored_count, kind=i64))
    checksum_value = iand(checksum_value * FNV_PRIME, MASK_32)
    if (checksum_value == 0_i64) checksum_value = 1_i64
  end function compute_context_checksum32

  pure integer(i32) function context_byte_to_u32(byte_value) result(unsigned_value)
    integer(i8), intent(in) :: byte_value

    unsigned_value = int(byte_value, kind=i32)
    if (unsigned_value < 0_i32) unsigned_value = unsigned_value + 256_i32
  end function context_byte_to_u32

end module mod_cuda_executor
