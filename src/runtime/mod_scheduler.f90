module mod_scheduler
  use mod_kinds,  only: i32
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_BUSY, MIZU_STATUS_INVALID_ARGUMENT

  implicit none

  private
  public :: scheduler_state, initialize_scheduler
  public :: can_accept_request, begin_request, complete_request
  public :: reset_scheduler

  type :: scheduler_state
    integer(i32) :: max_inflight_requests = 0_i32
    integer(i32) :: active_request_count  = 0_i32
    logical      :: is_initialized        = .false.
  end type scheduler_state

contains

  subroutine initialize_scheduler(scheduler, max_inflight_requests)
    type(scheduler_state), intent(out) :: scheduler
    integer(i32), intent(in)           :: max_inflight_requests

    scheduler%max_inflight_requests = max(0_i32, max_inflight_requests)
    scheduler%active_request_count  = 0_i32
    scheduler%is_initialized        = .true.
  end subroutine initialize_scheduler

  pure logical function can_accept_request(scheduler) result(can_accept)
    type(scheduler_state), intent(in) :: scheduler

    can_accept = scheduler%is_initialized .and. &
      (scheduler%active_request_count < scheduler%max_inflight_requests)
  end function can_accept_request

  subroutine begin_request(scheduler, status_code)
    type(scheduler_state), intent(inout) :: scheduler
    integer(i32), intent(out)            :: status_code

    if (.not. scheduler%is_initialized) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
    else if (.not. can_accept_request(scheduler)) then
      status_code = MIZU_STATUS_BUSY
    else
      scheduler%active_request_count = scheduler%active_request_count + 1_i32
      status_code = MIZU_STATUS_OK
    end if
  end subroutine begin_request

  subroutine complete_request(scheduler)
    type(scheduler_state), intent(inout) :: scheduler

    if (scheduler%active_request_count > 0_i32) then
      scheduler%active_request_count = scheduler%active_request_count - 1_i32
    end if
  end subroutine complete_request

  subroutine reset_scheduler(scheduler)
    type(scheduler_state), intent(inout) :: scheduler

    scheduler%max_inflight_requests = 0_i32
    scheduler%active_request_count  = 0_i32
    scheduler%is_initialized        = .false.
  end subroutine reset_scheduler

end module mod_scheduler
