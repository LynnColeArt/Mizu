module mod_memory
  use iso_c_binding, only: c_associated, c_int, c_intptr_t, c_null_ptr, c_ptr, c_size_t
  use mod_kinds,     only: i32, i64
  use mod_status,    only: MIZU_STATUS_INTERNAL_ERROR, MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_OK

  implicit none

  private
  public :: DEFAULT_HOST_ALIGNMENT_BYTES
  public :: allocate_aligned_host_buffer, free_host_buffer, pointer_is_aligned

  integer(i64), parameter :: HOST_POINTER_BYTES = int(storage_size(0_c_intptr_t) / 8, kind=i64)
  integer(i64), parameter :: DEFAULT_HOST_ALIGNMENT_BYTES = 64_i64

  interface
    function c_posix_memalign(memptr, alignment, size) bind(c, name="posix_memalign") result(error_code)
      import c_int, c_ptr, c_size_t
      type(c_ptr), intent(out)  :: memptr
      integer(c_size_t), value  :: alignment
      integer(c_size_t), value  :: size
      integer(c_int)            :: error_code
    end function c_posix_memalign

    subroutine c_free(buffer_ptr) bind(c, name="free")
      import c_ptr
      type(c_ptr), value :: buffer_ptr
    end subroutine c_free
  end interface

contains

  subroutine allocate_aligned_host_buffer(bytes_requested, alignment_bytes, buffer_ptr, status_code)
    integer(i64), intent(in)  :: bytes_requested
    integer(i64), intent(in)  :: alignment_bytes
    type(c_ptr), intent(out)  :: buffer_ptr
    integer(i32), intent(out) :: status_code
    integer(c_int)            :: error_code

    buffer_ptr = c_null_ptr
    if (bytes_requested < 0_i64 .or. .not. is_valid_alignment(alignment_bytes)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (bytes_requested == 0_i64) then
      status_code = MIZU_STATUS_OK
      return
    end if

    error_code = c_posix_memalign(buffer_ptr, int(alignment_bytes, kind=c_size_t), &
      int(bytes_requested, kind=c_size_t))
    if (error_code /= 0_c_int .or. .not. c_associated(buffer_ptr)) then
      buffer_ptr = c_null_ptr
      status_code = MIZU_STATUS_INTERNAL_ERROR
      return
    end if

    status_code = MIZU_STATUS_OK
  end subroutine allocate_aligned_host_buffer

  subroutine free_host_buffer(buffer_ptr)
    type(c_ptr), intent(inout) :: buffer_ptr

    if (c_associated(buffer_ptr)) call c_free(buffer_ptr)
    buffer_ptr = c_null_ptr
  end subroutine free_host_buffer

  logical function pointer_is_aligned(buffer_ptr, alignment_bytes) result(is_aligned)
    type(c_ptr), intent(in)  :: buffer_ptr
    integer(i64), intent(in) :: alignment_bytes
    integer(c_intptr_t)      :: address_value

    if (.not. c_associated(buffer_ptr) .or. .not. is_valid_alignment(alignment_bytes)) then
      is_aligned = .false.
      return
    end if

    address_value = transfer(buffer_ptr, address_value)
    is_aligned = iand(address_value, int(alignment_bytes - 1_i64, kind=c_intptr_t)) == 0_c_intptr_t
  end function pointer_is_aligned

  pure logical function is_valid_alignment(alignment_bytes) result(is_valid)
    integer(i64), intent(in) :: alignment_bytes

    is_valid = alignment_bytes >= HOST_POINTER_BYTES .and. &
      mod(alignment_bytes, HOST_POINTER_BYTES) == 0_i64 .and. &
      is_power_of_two(alignment_bytes)
  end function is_valid_alignment

  pure logical function is_power_of_two(value) result(is_power)
    integer(i64), intent(in) :: value

    if (value <= 0_i64) then
      is_power = .false.
      return
    end if

    is_power = iand(value, value - 1_i64) == 0_i64
  end function is_power_of_two

end module mod_memory
