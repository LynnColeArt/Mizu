module mod_workspace
  use mod_kinds, only: i32, i64
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT
  use mod_types,  only: workspace_state

  implicit none

  private
  public :: initialize_workspace, reserve_workspace_bytes
  public :: release_workspace_bytes, reset_workspace

contains

  subroutine initialize_workspace(workspace, bytes_reserved)
    type(workspace_state), intent(out) :: workspace
    integer(i64), intent(in)           :: bytes_reserved

    workspace%bytes_reserved = max(0_i64, bytes_reserved)
    workspace%bytes_in_use   = 0_i64
    workspace%is_ready       = .true.
  end subroutine initialize_workspace

  subroutine reserve_workspace_bytes(workspace, bytes_requested, status_code)
    type(workspace_state), intent(inout) :: workspace
    integer(i64), intent(in)             :: bytes_requested
    integer(i32), intent(out)            :: status_code

    if (bytes_requested < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    workspace%bytes_in_use = bytes_requested
    if (workspace%bytes_reserved < bytes_requested) then
      workspace%bytes_reserved = bytes_requested
    end if

    workspace%is_ready = .true.
    status_code = MIZU_STATUS_OK
  end subroutine reserve_workspace_bytes

  subroutine release_workspace_bytes(workspace)
    type(workspace_state), intent(inout) :: workspace

    workspace%bytes_in_use = 0_i64
  end subroutine release_workspace_bytes

  subroutine reset_workspace(workspace)
    type(workspace_state), intent(inout) :: workspace

    workspace%bytes_reserved = 0_i64
    workspace%bytes_in_use   = 0_i64
    workspace%is_ready       = .false.
  end subroutine reset_workspace

end module mod_workspace
