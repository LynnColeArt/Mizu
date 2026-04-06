module mod_request
  use mod_kinds, only: i32, i64
  use mod_types, only: MIZU_STAGE_NONE, session_handle

  implicit none

  private
  public :: request_state, initialize_request_state, reset_request_state

  type :: request_state
    integer(i64)        :: request_id  = 0_i64
    type(session_handle) :: session_owner
    integer(i32)        :: stage_kind  = MIZU_STAGE_NONE
    logical             :: is_active   = .false.
  end type request_state

contains

  subroutine initialize_request_state(request, request_id)
    type(request_state), intent(out) :: request
    integer(i64), intent(in)         :: request_id

    request%request_id = request_id
    request%stage_kind = MIZU_STAGE_NONE
    request%is_active  = .true.
  end subroutine initialize_request_state

  subroutine reset_request_state(request)
    type(request_state), intent(inout) :: request

    request%request_id = 0_i64
    request%stage_kind = MIZU_STAGE_NONE
    request%is_active  = .false.
  end subroutine reset_request_state

end module mod_request
