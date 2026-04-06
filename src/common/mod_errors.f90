module mod_errors
  use mod_kinds,  only: i32, MAX_ERROR_MESSAGE_LEN
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_END_OF_SEQUENCE, &
                        MIZU_STATUS_BUSY, MIZU_STATUS_BACKEND_UNAVAILABLE, &
                        MIZU_STATUS_NO_VALID_PLAN, MIZU_STATUS_SESSION_EVICTED, &
                        MIZU_STATUS_IO_ERROR, MIZU_STATUS_CACHE_ERROR, &
                        MIZU_STATUS_EXECUTION_ERROR, MIZU_STATUS_INTERNAL_ERROR

  implicit none

  private
  public :: PLANNER_FAILURE_NONE, PLANNER_FAILURE_BACKEND_UNAVAILABLE
  public :: PLANNER_FAILURE_UNSUPPORTED_MODEL, PLANNER_FAILURE_UNSUPPORTED_MODALITY
  public :: PLANNER_FAILURE_UNSUPPORTED_OP, PLANNER_FAILURE_UNSUPPORTED_SHAPE
  public :: PLANNER_FAILURE_ROUTE_DISALLOWED, PLANNER_FAILURE_NO_CANDIDATE_PLAN
  public :: CACHE_FAILURE_NONE, CACHE_FAILURE_ROOT_UNAVAILABLE
  public :: CACHE_FAILURE_KEY_INVALID, CACHE_FAILURE_ENTRY_NOT_FOUND
  public :: CACHE_FAILURE_ENTRY_CORRUPT, CACHE_FAILURE_WRITE_FAILED
  public :: CACHE_FAILURE_VERSION_MISMATCH, CACHE_FAILURE_ENTRY_EVICTED
  public :: error_record, clear_error_record, set_error_record
  public :: status_is_recoverable

  integer(i32), parameter :: PLANNER_FAILURE_NONE                = 0_i32
  integer(i32), parameter :: PLANNER_FAILURE_BACKEND_UNAVAILABLE = 1_i32
  integer(i32), parameter :: PLANNER_FAILURE_UNSUPPORTED_MODEL   = 2_i32
  integer(i32), parameter :: PLANNER_FAILURE_UNSUPPORTED_MODALITY = 3_i32
  integer(i32), parameter :: PLANNER_FAILURE_UNSUPPORTED_OP      = 4_i32
  integer(i32), parameter :: PLANNER_FAILURE_UNSUPPORTED_SHAPE   = 5_i32
  integer(i32), parameter :: PLANNER_FAILURE_ROUTE_DISALLOWED    = 6_i32
  integer(i32), parameter :: PLANNER_FAILURE_NO_CANDIDATE_PLAN   = 7_i32

  integer(i32), parameter :: CACHE_FAILURE_NONE             = 0_i32
  integer(i32), parameter :: CACHE_FAILURE_ROOT_UNAVAILABLE = 1_i32
  integer(i32), parameter :: CACHE_FAILURE_KEY_INVALID      = 2_i32
  integer(i32), parameter :: CACHE_FAILURE_ENTRY_NOT_FOUND  = 3_i32
  integer(i32), parameter :: CACHE_FAILURE_ENTRY_CORRUPT    = 4_i32
  integer(i32), parameter :: CACHE_FAILURE_WRITE_FAILED     = 5_i32
  integer(i32), parameter :: CACHE_FAILURE_VERSION_MISMATCH = 6_i32
  integer(i32), parameter :: CACHE_FAILURE_ENTRY_EVICTED    = 7_i32

  type :: error_record
    integer(i32)                          :: status_code             = MIZU_STATUS_OK
    integer(i32)                          :: planner_failure_reason  = PLANNER_FAILURE_NONE
    integer(i32)                          :: cache_failure_reason    = CACHE_FAILURE_NONE
    logical                               :: is_recoverable          = .true.
    character(len=MAX_ERROR_MESSAGE_LEN)  :: message                 = ""
  end type error_record

contains

  pure logical function status_is_recoverable(status_code) result(is_recoverable)
    integer(i32), intent(in) :: status_code

    select case (status_code)
    case (MIZU_STATUS_OK, MIZU_STATUS_END_OF_SEQUENCE)
      is_recoverable = .true.
    case (MIZU_STATUS_BUSY, MIZU_STATUS_BACKEND_UNAVAILABLE, &
          MIZU_STATUS_NO_VALID_PLAN, MIZU_STATUS_SESSION_EVICTED, &
          MIZU_STATUS_IO_ERROR, MIZU_STATUS_CACHE_ERROR, &
          MIZU_STATUS_EXECUTION_ERROR)
      is_recoverable = .true.
    case (MIZU_STATUS_INTERNAL_ERROR)
      is_recoverable = .false.
    case default
      is_recoverable = .true.
    end select
  end function status_is_recoverable

  subroutine clear_error_record(error)
    type(error_record), intent(inout) :: error

    error%status_code            = MIZU_STATUS_OK
    error%planner_failure_reason = PLANNER_FAILURE_NONE
    error%cache_failure_reason   = CACHE_FAILURE_NONE
    error%is_recoverable         = .true.
    error%message                = ""
  end subroutine clear_error_record

  subroutine set_error_record(error, status_code, message, planner_failure_reason, &
                              cache_failure_reason)
    type(error_record), intent(inout)        :: error
    integer(i32), intent(in)                 :: status_code
    character(len=*), intent(in), optional   :: message
    integer(i32), intent(in), optional       :: planner_failure_reason
    integer(i32), intent(in), optional       :: cache_failure_reason
    integer                                  :: copy_len

    call clear_error_record(error)

    error%status_code    = status_code
    error%is_recoverable = status_is_recoverable(status_code)

    if (present(planner_failure_reason)) then
      error%planner_failure_reason = planner_failure_reason
    end if

    if (present(cache_failure_reason)) then
      error%cache_failure_reason = cache_failure_reason
    end if

    if (present(message)) then
      copy_len = min(len_trim(message), len(error%message))
      if (copy_len > 0) then
        error%message(1:copy_len) = message(1:copy_len)
      end if
    end if
  end subroutine set_error_record

end module mod_errors
