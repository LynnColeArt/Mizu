module mod_apple_executor
  use iso_c_binding,     only: c_ptr, c_null_ptr
  use mod_kinds,         only: i8, i32, i64, MAX_PATH_LEN
  use mod_status,        only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                               MIZU_STATUS_INVALID_STATE
  use mod_types,         only: MIZU_STOP_REASON_NONE, MIZU_STAGE_NONE, MIZU_STAGE_PROJECTOR, &
                               MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, MIZU_EXEC_ROUTE_NONE, &
                               MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_METAL
  use mod_apple_bridge,  only: launch_apple_projector, launch_apple_prefill, launch_apple_decode
  use mod_model_manifest, only: hash_text64

  implicit none

  private
  public :: execute_apple_projector, execute_apple_prefill, execute_apple_decode
  public :: apple_context_bytes_are_valid, extract_apple_context_lineage
  public :: extract_apple_context_snapshot

contains

  subroutine execute_apple_projector(cache_root, artifact_path, execution_route, modal_byte_count, &
                                     placeholder_count, modal_content_hash, embedding_count, status_code, &
                                     workspace_buffer, workspace_bytes)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i32), intent(in)     :: execution_route
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
    if (.not. apple_route_is_supported(execution_route)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if
    if (len_trim(artifact_path) == 0) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call load_apple_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    if (.not. loaded_ok) call synthesize_apple_artifact_payload(MIZU_STAGE_PROJECTOR, execution_route, artifact_path, payload_text)
    if (index(payload_text, "stage=2") <= 0) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    payload_hash = combine_positive_hash64(positive_hash64(trim(payload_text)), modal_content_hash)
    workspace_buffer_local = c_null_ptr
    workspace_bytes_local = 0_i64
    if (present(workspace_buffer)) workspace_buffer_local = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_local = workspace_bytes
    call launch_apple_projector(execution_route, payload_hash, max(0_i64, modal_byte_count), placeholder_count, &
      embedding_count, status_code, workspace_buffer_local, workspace_bytes_local)
  end subroutine execute_apple_projector

  subroutine execute_apple_prefill(cache_root, artifact_path, execution_route, staged_tokens, staged_modal_count, &
                                   token_content_hash, modal_content_hash, consumed_token_count, status_code, &
                                   workspace_buffer, workspace_bytes, token_values, modal_bytes, context_bytes, &
                                   context_byte_count, context_artifact_hash)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i32), intent(in)     :: execution_route
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
    if (.not. apple_route_is_supported(execution_route)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if
    if (len_trim(artifact_path) == 0) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call load_apple_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    if (.not. loaded_ok) call synthesize_apple_artifact_payload(MIZU_STAGE_PREFILL, execution_route, artifact_path, payload_text)
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
    call launch_apple_prefill(execution_route, payload_hash, artifact_hash, max(0_i64, staged_tokens), &
      staged_modal_count, consumed_token_count, status_code, workspace_buffer_local, workspace_bytes_local, &
      token_values, modal_bytes, context_bytes, context_byte_count)
    if (present(context_artifact_hash)) context_artifact_hash = merge(artifact_hash, 0_i64, status_code == MIZU_STATUS_OK)
  end subroutine execute_apple_prefill

  subroutine execute_apple_decode(cache_root, artifact_path, execution_route, kv_before, token_budget, &
                                  emitted_token_count, token_value, stop_reason, status_code, workspace_buffer, &
                                  workspace_bytes, context_bytes, context_byte_count, updated_context_bytes, &
                                  updated_context_byte_count, context_artifact_hash)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i32), intent(in)     :: execution_route
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
    integer(i32)                 :: current_context_route
    logical                      :: loaded_ok
    logical                      :: lineage_known
    type(c_ptr)                  :: workspace_buffer_local

    emitted_token_count = 0_i64
    token_value = 0_i32
    stop_reason = MIZU_STOP_REASON_NONE
    current_context_stage = MIZU_STAGE_NONE
    current_context_route = MIZU_EXEC_ROUTE_NONE
    current_context_artifact_hash = 0_i64
    lineage_known = .false.
    if (present(context_artifact_hash)) context_artifact_hash = 0_i64

    if (.not. apple_route_is_supported(execution_route)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if
    if (len_trim(artifact_path) == 0 .or. token_budget <= 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call load_apple_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
    if (.not. loaded_ok) call synthesize_apple_artifact_payload(MIZU_STAGE_DECODE, execution_route, artifact_path, payload_text)
    if (index(payload_text, "stage=4") <= 0) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if
    if (.not. apple_context_bytes_are_valid(context_bytes, context_byte_count)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    artifact_hash = positive_hash64(trim(payload_text))
    payload_hash = artifact_hash
    call extract_apple_context_lineage(context_bytes, context_byte_count, current_context_stage, &
      current_context_route, current_context_artifact_hash, lineage_known)
    if (lineage_known .and. current_context_stage == MIZU_STAGE_DECODE) then
      if (current_context_artifact_hash /= artifact_hash) then
        status_code = MIZU_STATUS_INVALID_STATE
        return
      end if
      if (current_context_route /= execution_route) then
        status_code = MIZU_STATUS_INVALID_STATE
        return
      end if
    end if

    workspace_buffer_local = c_null_ptr
    workspace_bytes_local = 0_i64
    if (present(workspace_buffer)) workspace_buffer_local = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_local = workspace_bytes
    call launch_apple_decode(execution_route, payload_hash, artifact_hash, kv_before, token_budget, &
      emitted_token_count, token_value, stop_reason, status_code, workspace_buffer_local, workspace_bytes_local, &
      context_bytes, context_byte_count, updated_context_bytes, updated_context_byte_count)
    if (present(context_artifact_hash)) context_artifact_hash = merge(artifact_hash, 0_i64, status_code == MIZU_STATUS_OK)
  end subroutine execute_apple_decode

  pure logical function apple_context_bytes_are_valid(context_bytes, context_byte_count) result(is_valid)
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: context_byte_count
    integer(i32), parameter  :: TOTAL_BYTES = 96_i32
    integer(i32), parameter  :: VERSION_1 = 1_i32
    integer(i32), parameter  :: KIND_PREFILL = 1_i32
    integer(i32), parameter  :: KIND_DECODE = 2_i32
    integer(i32)             :: stored_count
    integer(i64)             :: expected_checksum
    integer(i64)             :: actual_checksum

    is_valid = .false.
    if (context_byte_count < TOTAL_BYTES) return
    if (int(size(context_bytes), kind=i32) < context_byte_count) return
    if (int(context_bytes(1), kind=i32) /= iachar("M")) return
    if (int(context_bytes(2), kind=i32) /= iachar("Z")) return
    if (int(context_bytes(3), kind=i32) /= iachar("A")) return
    if (int(context_bytes(4), kind=i32) /= iachar("P")) return
    if (int(context_bytes(5), kind=i32) /= VERSION_1) return
    if (int(context_bytes(6), kind=i32) /= KIND_PREFILL .and. int(context_bytes(6), kind=i32) /= KIND_DECODE) return
    stored_count = int(decode_context_u32le(context_bytes(13), context_bytes(14), context_bytes(15), context_bytes(16)), kind=i32)
    if (stored_count /= context_byte_count) return
    expected_checksum = decode_context_u32le(context_bytes(9), context_bytes(10), context_bytes(11), context_bytes(12))
    actual_checksum = compute_context_checksum32(context_bytes, stored_count)
    if (actual_checksum /= expected_checksum) return
    is_valid = .true.
  end function apple_context_bytes_are_valid

  pure subroutine extract_apple_context_lineage(context_bytes, context_byte_count, producer_stage, execution_route, &
                                                artifact_hash, lineage_known)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: producer_stage
    integer(i32), intent(out) :: execution_route
    integer(i64), intent(out) :: artifact_hash
    logical, intent(out)      :: lineage_known
    integer(i32)              :: producer_kind

    producer_stage = MIZU_STAGE_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE
    artifact_hash = 0_i64
    lineage_known = .false.
    if (.not. apple_context_bytes_are_valid(context_bytes, context_byte_count)) return

    producer_kind = int(context_bytes(6), kind=i32)
    select case (producer_kind)
    case (1_i32)
      producer_stage = MIZU_STAGE_PREFILL
    case (2_i32)
      producer_stage = MIZU_STAGE_DECODE
    end select
    execution_route = int(context_bytes(7), kind=i32)
    artifact_hash = decode_context_u64le(context_bytes(17), context_bytes(18), context_bytes(19), context_bytes(20), &
      context_bytes(21), context_bytes(22), context_bytes(23), context_bytes(24))
    lineage_known = (producer_stage /= MIZU_STAGE_NONE .and. artifact_hash /= 0_i64 .and. &
      apple_route_is_supported(execution_route))
  end subroutine extract_apple_context_lineage

  pure subroutine extract_apple_context_snapshot(context_bytes, context_byte_count, producer_stage, execution_route, &
                                                 artifact_hash, token_digest, modal_digest, kv_token_count, &
                                                 decode_step_count, last_token, state_digest, snapshot_valid)
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: producer_stage
    integer(i32), intent(out) :: execution_route
    integer(i64), intent(out) :: artifact_hash
    integer(i64), intent(out) :: token_digest
    integer(i64), intent(out) :: modal_digest
    integer(i64), intent(out) :: kv_token_count
    integer(i64), intent(out) :: decode_step_count
    integer(i32), intent(out) :: last_token
    integer(i64), intent(out) :: state_digest
    logical, intent(out)      :: snapshot_valid
    integer(i64), parameter   :: MASK_32 = int(z'FFFFFFFF', kind=i64)
    integer(i64), parameter   :: MASK_16 = int(z'FFFF', kind=i64)
    integer(i64)              :: counter_word
    integer(i64)              :: summary_word

    producer_stage = MIZU_STAGE_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE
    artifact_hash = 0_i64
    token_digest = 0_i64
    modal_digest = 0_i64
    kv_token_count = 0_i64
    decode_step_count = 0_i64
    last_token = 0_i32
    state_digest = 0_i64
    snapshot_valid = .false.
    if (.not. apple_context_bytes_are_valid(context_bytes, context_byte_count)) return

    call extract_apple_context_lineage(context_bytes, context_byte_count, producer_stage, execution_route, &
      artifact_hash, snapshot_valid)
    if (.not. snapshot_valid) return

    token_digest = decode_context_u64le(context_bytes(25), context_bytes(26), context_bytes(27), context_bytes(28), &
      context_bytes(29), context_bytes(30), context_bytes(31), context_bytes(32))
    modal_digest = decode_context_u64le(context_bytes(33), context_bytes(34), context_bytes(35), context_bytes(36), &
      context_bytes(37), context_bytes(38), context_bytes(39), context_bytes(40))
    counter_word = decode_context_u64le(context_bytes(41), context_bytes(42), context_bytes(43), context_bytes(44), &
      context_bytes(45), context_bytes(46), context_bytes(47), context_bytes(48))
    summary_word = decode_context_u64le(context_bytes(49), context_bytes(50), context_bytes(51), context_bytes(52), &
      context_bytes(53), context_bytes(54), context_bytes(55), context_bytes(56))
    state_digest = decode_context_u64le(context_bytes(57), context_bytes(58), context_bytes(59), context_bytes(60), &
      context_bytes(61), context_bytes(62), context_bytes(63), context_bytes(64))
    kv_token_count = iand(counter_word, MASK_32)
    decode_step_count = iand(shiftr(counter_word, 32), MASK_32)
    last_token = int(iand(summary_word, MASK_16), kind=i32)
  end subroutine extract_apple_context_snapshot

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

  subroutine load_apple_artifact_payload(cache_root, artifact_path, payload_text, loaded_ok)
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
  end subroutine load_apple_artifact_payload

  subroutine synthesize_apple_artifact_payload(stage_kind, execution_route, artifact_path, payload_text)
    integer(i32), intent(in)      :: stage_kind
    integer(i32), intent(in)      :: execution_route
    character(len=*), intent(in)  :: artifact_path
    character(len=*), intent(out) :: payload_text

    payload_text = ""
    write(payload_text, '(A,";virtual=1;stage=",I0,";route=",I0)') trim(artifact_path), stage_kind, execution_route
  end subroutine synthesize_apple_artifact_payload

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

  pure logical function apple_route_is_supported(execution_route) result(is_supported)
    integer(i32), intent(in) :: execution_route

    is_supported = (execution_route == MIZU_EXEC_ROUTE_ANE .or. execution_route == MIZU_EXEC_ROUTE_METAL)
  end function apple_route_is_supported

  pure integer(i64) function compute_context_checksum32(context_bytes, stored_count) result(checksum)
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: stored_count
    integer(i32), parameter  :: OFFSET_BASIS = int(z'811C9DC5', kind=i32)
    integer(i32), parameter  :: FNV_PRIME = 16777619_i32
    integer(i32)             :: index
    integer(i32)             :: checksum32
    integer(i32)             :: unsigned_byte

    checksum32 = OFFSET_BASIS
    if (stored_count <= 16_i32) then
      checksum32 = ieor(checksum32, max(0_i32, stored_count))
      checksum32 = checksum32 * FNV_PRIME
    else
      do index = 17_i32, stored_count
        unsigned_byte = iand(int(context_bytes(index), kind=i32), int(z'FF', kind=i32))
        checksum32 = ieor(checksum32, unsigned_byte)
        checksum32 = checksum32 * FNV_PRIME
      end do
      checksum32 = ieor(checksum32, stored_count)
      checksum32 = checksum32 * FNV_PRIME
    end if
    checksum = iand(int(checksum32, kind=i64), int(z'FFFFFFFF', kind=i64))
    if (checksum == 0_i64) checksum = 1_i64
  end function compute_context_checksum32

  pure integer(i64) function decode_context_u32le(byte_1, byte_2, byte_3, byte_4) result(value)
    integer(i8), intent(in) :: byte_1, byte_2, byte_3, byte_4
    integer(i64)            :: b1, b2, b3, b4

    b1 = iand(int(byte_1, kind=i64), int(z'FF', kind=i64))
    b2 = iand(int(byte_2, kind=i64), int(z'FF', kind=i64))
    b3 = iand(int(byte_3, kind=i64), int(z'FF', kind=i64))
    b4 = iand(int(byte_4, kind=i64), int(z'FF', kind=i64))
    value = b1 + shiftl(b2, 8) + shiftl(b3, 16) + shiftl(b4, 24)
  end function decode_context_u32le

  pure integer(i64) function decode_context_u64le(byte_1, byte_2, byte_3, byte_4, byte_5, byte_6, byte_7, &
                                                  byte_8) result(value)
    integer(i8), intent(in) :: byte_1, byte_2, byte_3, byte_4, byte_5, byte_6, byte_7, byte_8
    integer(i64)            :: b1, b2, b3, b4, b5, b6, b7, b8

    b1 = iand(int(byte_1, kind=i64), int(z'FF', kind=i64))
    b2 = iand(int(byte_2, kind=i64), int(z'FF', kind=i64))
    b3 = iand(int(byte_3, kind=i64), int(z'FF', kind=i64))
    b4 = iand(int(byte_4, kind=i64), int(z'FF', kind=i64))
    b5 = iand(int(byte_5, kind=i64), int(z'FF', kind=i64))
    b6 = iand(int(byte_6, kind=i64), int(z'FF', kind=i64))
    b7 = iand(int(byte_7, kind=i64), int(z'FF', kind=i64))
    b8 = iand(int(byte_8, kind=i64), int(z'FF', kind=i64))
    value = b1 + shiftl(b2, 8) + shiftl(b3, 16) + shiftl(b4, 24) + shiftl(b5, 32) + shiftl(b6, 40) + &
      shiftl(b7, 48) + shiftl(b8, 56)
  end function decode_context_u64le

end module mod_apple_executor
