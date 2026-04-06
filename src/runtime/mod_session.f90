module mod_session
  use mod_kinds,  only: i32, i64
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                        MIZU_STATUS_INVALID_STATE, MIZU_STATUS_SESSION_EVICTED
  use mod_types,  only: MIZU_SESSION_STATE_NONE, MIZU_SESSION_STATE_PENDING_INPUTS, &
                        MIZU_SESSION_STATE_LIVE_CONTEXT, MIZU_SESSION_STATE_PARKED, &
                        MIZU_STOP_REASON_NONE, MIZU_MODALITY_KIND_UNKNOWN, &
                        MIZU_DTYPE_UNKNOWN, session_config, session_info, &
                        session_state, execution_report

  implicit none

  private
  public :: initialize_session_state, reset_session_state
  public :: validate_attach_tokens, validate_attach_modal_input
  public :: validate_clear_pending_inputs, validate_prefill
  public :: validate_decode, validate_park, validate_resume
  public :: validate_read_output
  public :: stage_tokens, stage_modal_input, clear_pending_inputs
  public :: complete_prefill, complete_decode
  public :: park_session_state, resume_session_state
  public :: evict_parked_session, build_session_info

contains

  subroutine initialize_session_state(session, config)
    type(session_state), intent(out) :: session
    type(session_config), intent(in) :: config

    session%config             = config
    session%last_report        = execution_report()
    session%kv_token_count     = 0_i64
    session%staged_token_count = 0_i64
    session%last_output_token_count = 0_i64
    session%last_output_tokens = 0_i32
    session%staged_modal_count = 0_i32
    session%staged_modal_byte_count = 0_i64
    session%staged_modal_kind  = MIZU_MODALITY_KIND_UNKNOWN
    session%staged_modal_dtype = MIZU_DTYPE_UNKNOWN
    session%staged_modal_slot_name = ""
    session%last_stop_reason   = MIZU_STOP_REASON_NONE
    session%is_open            = .true.
    session%has_pending_inputs = .false.
    session%has_live_context   = .false.
    session%is_parked          = .false.
    session%has_decode_result  = .false.
    session%is_evicted         = .false.
  end subroutine initialize_session_state

  subroutine reset_session_state(session)
    type(session_state), intent(inout) :: session

    session%config                = session_config()
    session%last_report           = execution_report()
    session%kv_token_count        = 0_i64
    session%staged_token_count    = 0_i64
    session%last_output_token_count = 0_i64
    session%last_output_tokens    = 0_i32
    session%staged_modal_count    = 0_i32
    session%staged_modal_byte_count = 0_i64
    session%staged_modal_kind     = MIZU_MODALITY_KIND_UNKNOWN
    session%staged_modal_dtype    = MIZU_DTYPE_UNKNOWN
    session%staged_modal_slot_name = ""
    session%last_stop_reason      = MIZU_STOP_REASON_NONE
    session%is_open               = .false.
    session%has_pending_inputs    = .false.
    session%has_live_context      = .false.
    session%is_parked             = .false.
    session%has_decode_result     = .false.
    session%is_evicted            = .false.
  end subroutine reset_session_state

  pure integer(i32) function validate_attach_tokens(session) result(status_code)
    type(session_state), intent(in) :: session

    status_code = validate_attach_common(session)
  end function validate_attach_tokens

  pure integer(i32) function validate_attach_modal_input(session) result(status_code)
    type(session_state), intent(in) :: session

    status_code = validate_attach_common(session)
  end function validate_attach_modal_input

  pure integer(i32) function validate_clear_pending_inputs(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_clear_pending_inputs

  pure integer(i32) function validate_prefill(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_pending_inputs) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_prefill

  pure integer(i32) function validate_decode(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_live_context) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%has_pending_inputs) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_decode

  pure integer(i32) function validate_park(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_live_context) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_park

  pure integer(i32) function validate_resume(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_evicted) then
      status_code = MIZU_STATUS_SESSION_EVICTED
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_resume

  pure integer(i32) function validate_read_output(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (.not. session%has_decode_result) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_read_output

  subroutine stage_tokens(session, token_count, status_code)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in)           :: token_count
    integer(i32), intent(out)          :: status_code

    status_code = validate_attach_tokens(session)
    if (status_code /= MIZU_STATUS_OK) return

    if (token_count <= 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    session%staged_token_count = session%staged_token_count + token_count
    session%has_pending_inputs = .true.
  end subroutine stage_tokens

  subroutine stage_modal_input(session, status_code, byte_count, modality_kind, dtype, slot_name)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code
    integer(i64), intent(in), optional :: byte_count
    integer(i32), intent(in), optional :: modality_kind
    integer(i32), intent(in), optional :: dtype
    character(len=*), intent(in), optional :: slot_name

    status_code = validate_attach_modal_input(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%staged_modal_count = session%staged_modal_count + 1_i32
    if (present(byte_count)) then
      session%staged_modal_byte_count = session%staged_modal_byte_count + max(0_i64, byte_count)
    end if
    if (present(modality_kind)) then
      session%staged_modal_kind = modality_kind
    end if
    if (present(dtype)) then
      session%staged_modal_dtype = dtype
    end if
    if (present(slot_name)) then
      if (len_trim(slot_name) > 0) then
        session%staged_modal_slot_name = slot_name
      end if
    end if
    session%has_pending_inputs = .true.
  end subroutine stage_modal_input

  subroutine clear_pending_inputs(session, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code

    status_code = validate_clear_pending_inputs(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%staged_token_count = 0_i64
    session%staged_modal_count = 0_i32
    session%staged_modal_byte_count = 0_i64
    session%staged_modal_kind  = MIZU_MODALITY_KIND_UNKNOWN
    session%staged_modal_dtype = MIZU_DTYPE_UNKNOWN
    session%staged_modal_slot_name = ""
    session%has_pending_inputs = .false.
  end subroutine clear_pending_inputs

  subroutine complete_prefill(session, consumed_token_count, status_code)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in), optional :: consumed_token_count
    integer(i32), intent(out)          :: status_code
    integer(i64)                       :: token_count

    status_code = validate_prefill(session)
    if (status_code /= MIZU_STATUS_OK) return

    token_count = session%staged_token_count
    if (present(consumed_token_count)) token_count = consumed_token_count

    if (token_count < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    session%kv_token_count     = session%kv_token_count + token_count
    session%staged_token_count = 0_i64
    session%staged_modal_count = 0_i32
    session%staged_modal_byte_count = 0_i64
    session%staged_modal_kind  = MIZU_MODALITY_KIND_UNKNOWN
    session%staged_modal_dtype = MIZU_DTYPE_UNKNOWN
    session%staged_modal_slot_name = ""
    session%has_pending_inputs = .false.
    session%has_live_context   = .true.
    session%is_parked          = .false.
    session%is_evicted         = .false.
    session%has_decode_result  = .false.
    session%last_output_token_count = 0_i64
    session%last_stop_reason   = MIZU_STOP_REASON_NONE
  end subroutine complete_prefill

  subroutine complete_decode(session, emitted_token_count, stop_reason, status_code)
    type(session_state), intent(inout) :: session
    integer(i64), intent(in)           :: emitted_token_count
    integer(i32), intent(in), optional :: stop_reason
    integer(i32), intent(out)          :: status_code

    status_code = validate_decode(session)
    if (status_code /= MIZU_STATUS_OK) return

    if (emitted_token_count < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    session%kv_token_count        = session%kv_token_count + emitted_token_count
    session%last_output_token_count = emitted_token_count
    session%has_decode_result     = .true.

    if (present(stop_reason)) then
      session%last_stop_reason = stop_reason
    else
      session%last_stop_reason = MIZU_STOP_REASON_NONE
    end if
  end subroutine complete_decode

  subroutine park_session_state(session, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code

    status_code = validate_park(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%is_parked = .true.
  end subroutine park_session_state

  subroutine resume_session_state(session, status_code)
    type(session_state), intent(inout) :: session
    integer(i32), intent(out)          :: status_code

    status_code = validate_resume(session)
    if (status_code /= MIZU_STATUS_OK) return

    session%is_parked = .false.
  end subroutine resume_session_state

  subroutine evict_parked_session(session)
    type(session_state), intent(inout) :: session

    session%is_evicted         = .true.
    session%is_parked          = .true.
    session%has_live_context   = .false.
    session%kv_token_count     = 0_i64
    session%staged_modal_byte_count = 0_i64
    session%staged_modal_kind  = MIZU_MODALITY_KIND_UNKNOWN
    session%staged_modal_dtype = MIZU_DTYPE_UNKNOWN
    session%staged_modal_slot_name = ""
    session%has_decode_result  = .false.
    session%last_output_token_count = 0_i64
  end subroutine evict_parked_session

  pure function build_session_info(session) result(info)
    type(session_state), intent(in) :: session
    type(session_info)              :: info

    info%session_state_flags = pack_session_state_flags(session)
    info%kv_token_count      = session%kv_token_count
    info%staged_token_count  = session%staged_token_count
    info%staged_modal_count  = session%staged_modal_count
  end function build_session_info

  pure integer(i32) function validate_attach_common(session) result(status_code)
    type(session_state), intent(in) :: session

    if (.not. session%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (session%is_parked) then
      status_code = MIZU_STATUS_INVALID_STATE
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_attach_common

  pure integer(i64) function pack_session_state_flags(session) result(state_flags)
    type(session_state), intent(in) :: session

    state_flags = MIZU_SESSION_STATE_NONE

    if (session%has_pending_inputs) then
      state_flags = ior(state_flags, MIZU_SESSION_STATE_PENDING_INPUTS)
    end if

    if (session%has_live_context) then
      state_flags = ior(state_flags, MIZU_SESSION_STATE_LIVE_CONTEXT)
    end if

    if (session%is_parked) then
      state_flags = ior(state_flags, MIZU_SESSION_STATE_PARKED)
    end if
  end function pack_session_state_flags

end module mod_session
