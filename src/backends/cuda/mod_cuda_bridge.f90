module mod_cuda_bridge
  use iso_c_binding, only: c_char, c_int32_t, c_int64_t, c_size_t, c_null_char
  use mod_kinds,     only: i32, i64, MAX_NAME_LEN
  use mod_status,    only: MIZU_STATUS_OK

  implicit none

  private
  public :: cuda_device_info
  public :: query_cuda_device_info, launch_cuda_prefill, launch_cuda_decode

  type :: cuda_device_info
    logical                    :: is_available = .false.
    integer(i32)               :: device_count = 0_i32
    integer(i64)               :: total_memory_bytes = 0_i64
    integer(i32)               :: compute_major = 0_i32
    integer(i32)               :: compute_minor = 0_i32
    integer(i32)               :: multiprocessor_count = 0_i32
    character(len=MAX_NAME_LEN) :: device_name = ""
  end type cuda_device_info

  interface
    subroutine c_mizu_cuda_bridge_get_device_info(device_count, total_memory_bytes, compute_major, &
                                                  compute_minor, multiprocessor_count, device_name, &
                                                  device_name_capacity, status_code) &
        bind(c, name="mizu_cuda_bridge_get_device_info")
      import c_char, c_int32_t, c_int64_t, c_size_t
      integer(c_int32_t), intent(out) :: device_count
      integer(c_int64_t), intent(out) :: total_memory_bytes
      integer(c_int32_t), intent(out) :: compute_major
      integer(c_int32_t), intent(out) :: compute_minor
      integer(c_int32_t), intent(out) :: multiprocessor_count
      character(kind=c_char), intent(out) :: device_name(*)
      integer(c_size_t), value       :: device_name_capacity
      integer(c_int32_t), intent(out) :: status_code
    end subroutine c_mizu_cuda_bridge_get_device_info

    subroutine c_mizu_cuda_bridge_prefill(payload_hash, staged_tokens, staged_modal_count, &
                                          consumed_token_count, status_code) &
        bind(c, name="mizu_cuda_bridge_prefill")
      import c_int32_t, c_int64_t
      integer(c_int64_t), value      :: payload_hash
      integer(c_int64_t), value      :: staged_tokens
      integer(c_int32_t), value      :: staged_modal_count
      integer(c_int64_t), intent(out) :: consumed_token_count
      integer(c_int32_t), intent(out) :: status_code
    end subroutine c_mizu_cuda_bridge_prefill

    subroutine c_mizu_cuda_bridge_decode(payload_hash, kv_before, token_budget, emitted_token_count, &
                                         token_value, stop_reason, status_code) &
        bind(c, name="mizu_cuda_bridge_decode")
      import c_int32_t, c_int64_t
      integer(c_int64_t), value      :: payload_hash
      integer(c_int64_t), value      :: kv_before
      integer(c_int64_t), value      :: token_budget
      integer(c_int64_t), intent(out) :: emitted_token_count
      integer(c_int32_t), intent(out) :: token_value
      integer(c_int32_t), intent(out) :: stop_reason
      integer(c_int32_t), intent(out) :: status_code
    end subroutine c_mizu_cuda_bridge_decode
  end interface

contains

  subroutine query_cuda_device_info(info, status_code)
    type(cuda_device_info), intent(out) :: info
    integer(i32), intent(out)           :: status_code
    character(kind=c_char)              :: device_name_buffer(MAX_NAME_LEN + 1)
    integer(c_int32_t)                  :: device_count_c
    integer(c_int64_t)                  :: total_memory_bytes_c
    integer(c_int32_t)                  :: compute_major_c
    integer(c_int32_t)                  :: compute_minor_c
    integer(c_int32_t)                  :: multiprocessor_count_c
    integer(c_int32_t)                  :: status_code_c

    info = cuda_device_info()
    device_name_buffer = c_null_char
    device_count_c = 0_c_int32_t
    total_memory_bytes_c = 0_c_int64_t
    compute_major_c = 0_c_int32_t
    compute_minor_c = 0_c_int32_t
    multiprocessor_count_c = 0_c_int32_t
    status_code_c = 0_c_int32_t

    call c_mizu_cuda_bridge_get_device_info(device_count_c, total_memory_bytes_c, compute_major_c, &
      compute_minor_c, multiprocessor_count_c, device_name_buffer, int(size(device_name_buffer), kind=c_size_t), &
      status_code_c)

    status_code = int(status_code_c, kind=i32)
    if (status_code /= MIZU_STATUS_OK) return

    info%device_count = int(device_count_c, kind=i32)
    info%total_memory_bytes = int(total_memory_bytes_c, kind=i64)
    info%compute_major = int(compute_major_c, kind=i32)
    info%compute_minor = int(compute_minor_c, kind=i32)
    info%multiprocessor_count = int(multiprocessor_count_c, kind=i32)
    info%is_available = (info%device_count > 0_i32)
    call copy_c_string_to_fortran(device_name_buffer, info%device_name)
  end subroutine query_cuda_device_info

  subroutine launch_cuda_prefill(payload_hash, staged_tokens, staged_modal_count, consumed_token_count, &
                                 status_code)
    integer(i64), intent(in)  :: payload_hash
    integer(i64), intent(in)  :: staged_tokens
    integer(i32), intent(in)  :: staged_modal_count
    integer(i64), intent(out) :: consumed_token_count
    integer(i32), intent(out) :: status_code
    integer(c_int64_t)        :: consumed_token_count_c
    integer(c_int32_t)        :: status_code_c

    call c_mizu_cuda_bridge_prefill(int(payload_hash, kind=c_int64_t), int(staged_tokens, kind=c_int64_t), &
      int(staged_modal_count, kind=c_int32_t), consumed_token_count_c, status_code_c)

    consumed_token_count = int(consumed_token_count_c, kind=i64)
    status_code = int(status_code_c, kind=i32)
  end subroutine launch_cuda_prefill

  subroutine launch_cuda_decode(payload_hash, kv_before, token_budget, emitted_token_count, token_value, &
                                stop_reason, status_code)
    integer(i64), intent(in)  :: payload_hash
    integer(i64), intent(in)  :: kv_before
    integer(i64), intent(in)  :: token_budget
    integer(i64), intent(out) :: emitted_token_count
    integer(i32), intent(out) :: token_value
    integer(i32), intent(out) :: stop_reason
    integer(i32), intent(out) :: status_code
    integer(c_int64_t)        :: emitted_token_count_c
    integer(c_int32_t)        :: token_value_c
    integer(c_int32_t)        :: stop_reason_c
    integer(c_int32_t)        :: status_code_c

    call c_mizu_cuda_bridge_decode(int(payload_hash, kind=c_int64_t), int(kv_before, kind=c_int64_t), &
      int(token_budget, kind=c_int64_t), emitted_token_count_c, token_value_c, stop_reason_c, status_code_c)

    emitted_token_count = int(emitted_token_count_c, kind=i64)
    token_value = int(token_value_c, kind=i32)
    stop_reason = int(stop_reason_c, kind=i32)
    status_code = int(status_code_c, kind=i32)
  end subroutine launch_cuda_decode

  subroutine copy_c_string_to_fortran(c_buffer, output_text)
    character(kind=c_char), intent(in) :: c_buffer(:)
    character(len=*), intent(out)      :: output_text
    integer(i32)                       :: index_char
    character(len=1)                   :: plain_char

    output_text = ""
    do index_char = 1_i32, min(int(size(c_buffer), kind=i32), len(output_text))
      if (c_buffer(index_char) == c_null_char) exit
      plain_char = transfer(c_buffer(index_char), plain_char)
      output_text(index_char:index_char) = plain_char
    end do
  end subroutine copy_c_string_to_fortran

end module mod_cuda_bridge
