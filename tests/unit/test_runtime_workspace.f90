program test_runtime_workspace
  use iso_c_binding, only: c_associated, c_f_pointer
  use mod_kinds,     only: c_i8, i32, i64, MEGABYTE
  use mod_status,    only: MIZU_STATUS_OK
  use mod_types,     only: runtime_state, runtime_config
  use mod_runtime,   only: initialize_runtime_state, reset_runtime_state
  use mod_workspace, only: reserve_workspace_bytes, release_workspace_bytes

  implicit none

  type(runtime_state) :: runtime
  integer(c_i8), pointer :: workspace_view(:)
  integer(i32)        :: status_code

  call initialize_runtime_state(runtime, runtime_config())
  call expect_true("runtime workspace should start ready", runtime%workspace%is_ready)
  call expect_equal_i64("runtime workspace should start with zero reserved bytes", &
    runtime%workspace%bytes_reserved, 0_i64)
  call expect_equal_i64("runtime workspace should start with zero bytes in use", &
    runtime%workspace%bytes_in_use, 0_i64)
  call expect_true("runtime workspace should start without a host buffer", &
    .not. c_associated(runtime%workspace%host_buffer))

  call reserve_workspace_bytes(runtime%workspace, 2_i64 * MEGABYTE, status_code)
  call expect_equal_i32("first workspace reservation should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("first reservation should set reserved bytes", runtime%workspace%bytes_reserved, &
    2_i64 * MEGABYTE)
  call expect_equal_i64("first reservation should set bytes in use", runtime%workspace%bytes_in_use, &
    2_i64 * MEGABYTE)
  call expect_true("first reservation should allocate a host buffer", &
    c_associated(runtime%workspace%host_buffer))
  call c_f_pointer(runtime%workspace%host_buffer, workspace_view, [int(runtime%workspace%bytes_reserved)])
  workspace_view(1) = 17_c_i8

  call reserve_workspace_bytes(runtime%workspace, 1_i64 * MEGABYTE, status_code)
  call expect_equal_i32("smaller workspace reservation should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("smaller reservation should preserve high-water mark", &
    runtime%workspace%bytes_reserved, 2_i64 * MEGABYTE)
  call expect_equal_i64("smaller reservation should update in-use bytes", &
    runtime%workspace%bytes_in_use, 1_i64 * MEGABYTE)
  call c_f_pointer(runtime%workspace%host_buffer, workspace_view, [int(runtime%workspace%bytes_reserved)])
  call expect_equal_i32("smaller reservation should preserve the allocated host buffer", &
    int(workspace_view(1), kind=i32), 17_i32)

  call reserve_workspace_bytes(runtime%workspace, 6_i64 * MEGABYTE, status_code)
  call expect_equal_i32("larger workspace reservation should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("larger reservation should grow high-water mark", &
    runtime%workspace%bytes_reserved, 6_i64 * MEGABYTE)
  call expect_equal_i64("larger reservation should update in-use bytes", &
    runtime%workspace%bytes_in_use, 6_i64 * MEGABYTE)
  call expect_true("larger reservation should keep a host buffer", c_associated(runtime%workspace%host_buffer))

  call release_workspace_bytes(runtime%workspace)
  call expect_equal_i64("workspace release should clear in-use bytes", runtime%workspace%bytes_in_use, 0_i64)
  call expect_equal_i64("workspace release should preserve reserved bytes", runtime%workspace%bytes_reserved, &
    6_i64 * MEGABYTE)

  call reset_runtime_state(runtime)
  call expect_true("runtime workspace should reset to not-ready", .not. runtime%workspace%is_ready)
  call expect_equal_i64("runtime workspace reset should clear reserved bytes", &
    runtime%workspace%bytes_reserved, 0_i64)
  call expect_equal_i64("runtime workspace reset should clear in-use bytes", &
    runtime%workspace%bytes_in_use, 0_i64)
  call expect_true("runtime workspace reset should free the host buffer", &
    .not. c_associated(runtime%workspace%host_buffer))

  write(*, "(A)") "test_runtime_workspace: PASS"

contains

  subroutine expect_true(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (.not. condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_true

  subroutine expect_equal_i32(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i32), intent(in)     :: actual
    integer(i32), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_i64

end program test_runtime_workspace
