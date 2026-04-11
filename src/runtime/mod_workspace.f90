module mod_workspace
  use iso_c_binding, only: c_associated, c_f_pointer, c_null_ptr, c_ptr
  use mod_kinds,     only: c_i8, i32, i64
  use mod_memory,    only: DEFAULT_HOST_ALIGNMENT_BYTES, allocate_aligned_host_buffer, free_host_buffer
  use mod_status,    only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT
  use mod_types,     only: workspace_state

  implicit none

  private
  public :: initialize_workspace, reserve_workspace_bytes
  public :: release_workspace_bytes, reset_workspace

contains

  subroutine initialize_workspace(workspace, bytes_reserved)
    type(workspace_state), intent(out) :: workspace
    integer(i64), intent(in)           :: bytes_reserved
    integer(i32)                       :: status_code

    workspace = workspace_state()
    workspace%host_alignment_bytes = DEFAULT_HOST_ALIGNMENT_BYTES
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
    type(c_ptr)                          :: new_buffer
    integer(i64)                         :: alignment_bytes

    if (bytes_requested < 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    alignment_bytes = workspace%host_alignment_bytes
    if (alignment_bytes <= 0_i64) alignment_bytes = DEFAULT_HOST_ALIGNMENT_BYTES

    if (bytes_requested > workspace%bytes_reserved) then
      call allocate_aligned_host_buffer(bytes_requested, alignment_bytes, new_buffer, status_code)
      if (status_code /= MIZU_STATUS_OK) then
        return
      end if

      call copy_workspace_bytes(workspace%host_buffer, new_buffer, &
        min(workspace%bytes_reserved, bytes_requested))
      call free_host_buffer(workspace%host_buffer)
      workspace%host_buffer = new_buffer
      workspace%bytes_reserved = bytes_requested
      workspace%host_alignment_bytes = alignment_bytes
      workspace%allocation_count = workspace%allocation_count + 1_i64
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

    call free_host_buffer(workspace%host_buffer)
    workspace%bytes_reserved       = 0_i64
    workspace%bytes_in_use         = 0_i64
    workspace%host_alignment_bytes = 0_i64
    workspace%allocation_count     = 0_i64
    workspace%is_ready             = .false.
  end subroutine reset_workspace

  subroutine copy_workspace_bytes(source_buffer, target_buffer, byte_count)
    type(c_ptr), intent(in)  :: source_buffer
    type(c_ptr), intent(in)  :: target_buffer
    integer(i64), intent(in) :: byte_count
    integer(c_i8), pointer   :: source_view(:)
    integer(c_i8), pointer   :: target_view(:)

    if (byte_count <= 0_i64) return
    if (.not. c_associated(source_buffer) .or. .not. c_associated(target_buffer)) return

    call c_f_pointer(source_buffer, source_view, [int(byte_count)])
    call c_f_pointer(target_buffer, target_view, [int(byte_count)])
    target_view = source_view
  end subroutine copy_workspace_bytes

end module mod_workspace
