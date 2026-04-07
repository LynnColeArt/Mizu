module mod_workspace
  use iso_c_binding, only: c_ptr, c_null_ptr, c_size_t, c_associated
  use mod_kinds, only: i32, i64
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_INTERNAL_ERROR
  use mod_types,  only: workspace_state

  implicit none

  private
  public :: initialize_workspace, reserve_workspace_bytes
  public :: release_workspace_bytes, reset_workspace

  interface
    function c_malloc(size) bind(c, name="malloc") result(buffer_ptr)
      import c_ptr, c_size_t
      integer(c_size_t), value :: size
      type(c_ptr)              :: buffer_ptr
    end function c_malloc

    function c_realloc(buffer_ptr, size) bind(c, name="realloc") result(resized_ptr)
      import c_ptr, c_size_t
      type(c_ptr), value       :: buffer_ptr
      integer(c_size_t), value :: size
      type(c_ptr)              :: resized_ptr
    end function c_realloc

    subroutine c_free(buffer_ptr) bind(c, name="free")
      import c_ptr
      type(c_ptr), value :: buffer_ptr
    end subroutine c_free
  end interface

contains

  subroutine initialize_workspace(workspace, bytes_reserved)
    type(workspace_state), intent(out) :: workspace
    integer(i64), intent(in)           :: bytes_reserved
    integer(i32)                       :: status_code

    workspace = workspace_state()
    workspace%is_ready = .true.
    if (bytes_reserved <= 0_i64) return

    call reserve_workspace_bytes(workspace, bytes_reserved, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      call reset_workspace(workspace)
      return
    end if
    call release_workspace_bytes(workspace)
  end subroutine initialize_workspace

  subroutine reserve_workspace_bytes(workspace, bytes_requested, status_code)
    type(workspace_state), intent(inout) :: workspace
    integer(i64), intent(in)             :: bytes_requested
    integer(i32), intent(out)            :: status_code
    type(c_ptr)                          :: resized_ptr

    if (bytes_requested < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (bytes_requested > workspace%bytes_reserved) then
      if (c_associated(workspace%host_buffer)) then
        resized_ptr = c_realloc(workspace%host_buffer, int(bytes_requested, kind=c_size_t))
      else
        resized_ptr = c_malloc(int(bytes_requested, kind=c_size_t))
      end if

      if (.not. c_associated(resized_ptr) .and. bytes_requested > 0_i64) then
        status_code = MIZU_STATUS_INTERNAL_ERROR
        return
      end if

      workspace%host_buffer = resized_ptr
      workspace%bytes_reserved = bytes_requested
    end if

    workspace%bytes_in_use = bytes_requested
    workspace%is_ready = .true.
    status_code = MIZU_STATUS_OK
  end subroutine reserve_workspace_bytes

  subroutine release_workspace_bytes(workspace)
    type(workspace_state), intent(inout) :: workspace

    workspace%bytes_in_use = 0_i64
  end subroutine release_workspace_bytes

  subroutine reset_workspace(workspace)
    type(workspace_state), intent(inout) :: workspace

    if (c_associated(workspace%host_buffer)) then
      call c_free(workspace%host_buffer)
      workspace%host_buffer = c_null_ptr
    end if
    workspace%bytes_reserved = 0_i64
    workspace%bytes_in_use   = 0_i64
    workspace%is_ready       = .false.
  end subroutine reset_workspace

end module mod_workspace
