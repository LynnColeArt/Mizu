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
  public :: extract_cuda_context_slot_snapshot

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
    integer(i64)                 :: workspace_bytes_local
    logical                      :: loaded_ok
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

    payload_hash = combine_positive_hash64(positive_hash64(trim(payload_text)), modal_content_hash)
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
    integer(i64)                 :: workspace_bytes_local
    logical                      :: loaded_ok
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
    payload_hash = artifact_hash
    payload_hash = combine_positive_hash64(payload_hash, token_content_hash)
    payload_hash = combine_positive_hash64(payload_hash, modal_content_hash)
    workspace_buffer_local = c_null_ptr
    workspace_bytes_local = 0_i64
    if (present(workspace_buffer)) workspace_buffer_local = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_local = workspace_bytes
    call launch_cuda_prefill(payload_hash, artifact_hash, max(0_i64, staged_tokens), staged_modal_count, &
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
    integer(i64)                 :: workspace_bytes_local
    integer(i64)                 :: current_context_artifact_hash
    integer(i32)                 :: current_context_stage
    logical                      :: loaded_ok
    logical                      :: lineage_known
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
    call launch_cuda_decode(payload_hash, artifact_hash, kv_before, token_budget, emitted_token_count, token_value, &
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
