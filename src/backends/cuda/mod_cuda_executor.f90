module mod_cuda_executor
  use mod_kinds,          only: i32, i64, MAX_PATH_LEN
  use mod_status,         only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                                MIZU_STATUS_INVALID_STATE
  use mod_types,          only: MIZU_STOP_REASON_NONE
  use mod_model_manifest, only: hash_text64

  implicit none

  private
  public :: execute_cuda_prefill, execute_cuda_decode

contains

  subroutine execute_cuda_prefill(cache_root, artifact_path, staged_tokens, staged_modal_count, &
                                  consumed_token_count, status_code)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i64), intent(in)     :: staged_tokens
    integer(i32), intent(in)     :: staged_modal_count
    integer(i64), intent(out)    :: consumed_token_count
    integer(i32), intent(out)    :: status_code
    character(len=1024)          :: payload_text
    logical                      :: loaded_ok

    consumed_token_count = 0_i64
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

    consumed_token_count = max(0_i64, staged_tokens)
    if (consumed_token_count == 0_i64 .and. staged_modal_count > 0_i32) consumed_token_count = 1_i64
    status_code = MIZU_STATUS_OK
  end subroutine execute_cuda_prefill

  subroutine execute_cuda_decode(cache_root, artifact_path, kv_before, token_budget, emitted_token_count, &
                                 token_value, stop_reason, status_code)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: artifact_path
    integer(i64), intent(in)     :: kv_before
    integer(i64), intent(in)     :: token_budget
    integer(i64), intent(out)    :: emitted_token_count
    integer(i32), intent(out)    :: token_value
    integer(i32), intent(out)    :: stop_reason
    integer(i32), intent(out)    :: status_code
    character(len=1024)          :: payload_text
    character(len=128)           :: seed_text
    integer(i64)                 :: token_hash
    logical                      :: loaded_ok

    emitted_token_count = 0_i64
    token_value = 0_i32
    stop_reason = MIZU_STOP_REASON_NONE

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

    write(seed_text, '(I0,":",I0)') kv_before, token_budget
    token_hash = positive_hash64(trim(payload_text) // ":" // trim(seed_text))
    emitted_token_count = min(token_budget, 1_i64)
    token_value = int(1_i64 + mod(token_hash, 4095_i64), kind=i32)
    status_code = MIZU_STATUS_OK
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

end module mod_cuda_executor
