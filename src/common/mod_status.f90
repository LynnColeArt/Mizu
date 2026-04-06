module mod_status
  use mod_kinds, only: i32

  implicit none

  private
  public :: MIZU_STATUS_OK, MIZU_STATUS_END_OF_SEQUENCE
  public :: MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_INVALID_STATE
  public :: MIZU_STATUS_BUFFER_TOO_SMALL, MIZU_STATUS_ABI_MISMATCH
  public :: MIZU_STATUS_BUSY
  public :: MIZU_STATUS_UNSUPPORTED_MODEL, MIZU_STATUS_UNSUPPORTED_MODALITY
  public :: MIZU_STATUS_BACKEND_UNAVAILABLE, MIZU_STATUS_NO_VALID_PLAN
  public :: MIZU_STATUS_SESSION_EVICTED
  public :: MIZU_STATUS_IO_ERROR, MIZU_STATUS_CACHE_ERROR
  public :: MIZU_STATUS_EXECUTION_ERROR, MIZU_STATUS_INTERNAL_ERROR
  public :: STATUS_BAND_SUCCESS, STATUS_BAND_CALLER
  public :: STATUS_BAND_SUPPORT, STATUS_BAND_RUNTIME
  public :: get_status_band, status_is_success, status_is_failure

  integer(i32), parameter :: MIZU_STATUS_OK               = 0_i32
  integer(i32), parameter :: MIZU_STATUS_END_OF_SEQUENCE  = 1_i32

  integer(i32), parameter :: MIZU_STATUS_INVALID_ARGUMENT = 1000_i32
  integer(i32), parameter :: MIZU_STATUS_INVALID_STATE    = 1001_i32
  integer(i32), parameter :: MIZU_STATUS_BUFFER_TOO_SMALL = 1002_i32
  integer(i32), parameter :: MIZU_STATUS_ABI_MISMATCH     = 1003_i32
  integer(i32), parameter :: MIZU_STATUS_BUSY             = 1004_i32

  integer(i32), parameter :: MIZU_STATUS_UNSUPPORTED_MODEL      = 2000_i32
  integer(i32), parameter :: MIZU_STATUS_UNSUPPORTED_MODALITY   = 2001_i32
  integer(i32), parameter :: MIZU_STATUS_BACKEND_UNAVAILABLE    = 2002_i32
  integer(i32), parameter :: MIZU_STATUS_NO_VALID_PLAN          = 2003_i32
  integer(i32), parameter :: MIZU_STATUS_SESSION_EVICTED        = 2004_i32

  integer(i32), parameter :: MIZU_STATUS_IO_ERROR        = 3000_i32
  integer(i32), parameter :: MIZU_STATUS_CACHE_ERROR     = 3001_i32
  integer(i32), parameter :: MIZU_STATUS_EXECUTION_ERROR = 3002_i32
  integer(i32), parameter :: MIZU_STATUS_INTERNAL_ERROR  = 3003_i32

  integer(i32), parameter :: STATUS_BAND_SUCCESS = 0_i32
  integer(i32), parameter :: STATUS_BAND_CALLER  = 1_i32
  integer(i32), parameter :: STATUS_BAND_SUPPORT = 2_i32
  integer(i32), parameter :: STATUS_BAND_RUNTIME = 3_i32

contains

  pure integer(i32) function get_status_band(status_code) result(status_band)
    integer(i32), intent(in) :: status_code

    select case (status_code)
    case (MIZU_STATUS_OK, MIZU_STATUS_END_OF_SEQUENCE)
      status_band = STATUS_BAND_SUCCESS
    case (MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_INVALID_STATE, &
          MIZU_STATUS_BUFFER_TOO_SMALL, MIZU_STATUS_ABI_MISMATCH, &
          MIZU_STATUS_BUSY)
      status_band = STATUS_BAND_CALLER
    case (MIZU_STATUS_UNSUPPORTED_MODEL, MIZU_STATUS_UNSUPPORTED_MODALITY, &
          MIZU_STATUS_BACKEND_UNAVAILABLE, MIZU_STATUS_NO_VALID_PLAN, &
          MIZU_STATUS_SESSION_EVICTED)
      status_band = STATUS_BAND_SUPPORT
    case default
      status_band = STATUS_BAND_RUNTIME
    end select
  end function get_status_band

  pure logical function status_is_success(status_code) result(is_success)
    integer(i32), intent(in) :: status_code

    is_success = (get_status_band(status_code) == STATUS_BAND_SUCCESS)
  end function status_is_success

  pure logical function status_is_failure(status_code) result(is_failure)
    integer(i32), intent(in) :: status_code

    is_failure = .not. status_is_success(status_code)
  end function status_is_failure

end module mod_status
