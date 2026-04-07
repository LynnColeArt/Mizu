module mod_cuda_bridge
  use iso_c_binding, only: c_char, c_int32_t, c_int64_t, c_size_t, c_null_char, c_ptr, c_null_ptr, c_loc
  use mod_kinds,     only: c_i8, c_i32, i8, i32, i64, MAX_NAME_LEN
  use mod_types,     only: MAX_LIVE_CONTEXT_BYTES
  use mod_status,    only: MIZU_STATUS_OK

  implicit none

  private
  public :: cuda_device_info
  public :: query_cuda_device_info, launch_cuda_projector, launch_cuda_prefill, launch_cuda_decode

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

    subroutine c_mizu_cuda_bridge_prefill(payload_hash, token_values, token_count, modal_bytes, &
                                          modal_byte_count, staged_modal_count, workspace_buffer, &
                                          workspace_bytes, context_bytes, context_capacity, &
                                          context_byte_count, consumed_token_count, status_code) &
        bind(c, name="mizu_cuda_bridge_prefill")
      import c_int32_t, c_int64_t, c_ptr
      integer(c_int64_t), value      :: payload_hash
      type(c_ptr), value             :: token_values
      integer(c_int64_t), value      :: token_count
      type(c_ptr), value             :: modal_bytes
      integer(c_int64_t), value      :: modal_byte_count
      integer(c_int32_t), value      :: staged_modal_count
      type(c_ptr), value             :: workspace_buffer
      integer(c_int64_t), value      :: workspace_bytes
      type(c_ptr), value             :: context_bytes
      integer(c_int32_t), value      :: context_capacity
      integer(c_int32_t), intent(out) :: context_byte_count
      integer(c_int64_t), intent(out) :: consumed_token_count
      integer(c_int32_t), intent(out) :: status_code
    end subroutine c_mizu_cuda_bridge_prefill

    subroutine c_mizu_cuda_bridge_projector(payload_hash, modal_byte_count, placeholder_count, &
                                            workspace_buffer, workspace_bytes, embedding_count, &
                                            status_code) &
        bind(c, name="mizu_cuda_bridge_projector")
      import c_int32_t, c_int64_t, c_ptr
      integer(c_int64_t), value       :: payload_hash
      integer(c_int64_t), value       :: modal_byte_count
      integer(c_int32_t), value       :: placeholder_count
      type(c_ptr), value              :: workspace_buffer
      integer(c_int64_t), value       :: workspace_bytes
      integer(c_int64_t), intent(out) :: embedding_count
      integer(c_int32_t), intent(out) :: status_code
    end subroutine c_mizu_cuda_bridge_projector

    subroutine c_mizu_cuda_bridge_decode(payload_hash, kv_before, token_budget, context_bytes, &
                                         context_byte_count, workspace_buffer, workspace_bytes, &
                                         updated_context_bytes, updated_context_capacity, &
                                         updated_context_byte_count, emitted_token_count, token_value, &
                                         stop_reason, status_code) &
        bind(c, name="mizu_cuda_bridge_decode")
      import c_int32_t, c_int64_t, c_ptr
      integer(c_int64_t), value      :: payload_hash
      integer(c_int64_t), value      :: kv_before
      integer(c_int64_t), value      :: token_budget
      type(c_ptr), value             :: context_bytes
      integer(c_int32_t), value      :: context_byte_count
      type(c_ptr), value             :: workspace_buffer
      integer(c_int64_t), value      :: workspace_bytes
      type(c_ptr), value             :: updated_context_bytes
      integer(c_int32_t), value      :: updated_context_capacity
      integer(c_int32_t), intent(out) :: updated_context_byte_count
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
                                 status_code, workspace_buffer, workspace_bytes, token_values, modal_bytes, &
                                 context_bytes, context_byte_count)
    integer(i64), intent(in)  :: payload_hash
    integer(i64), intent(in)  :: staged_tokens
    integer(i32), intent(in)  :: staged_modal_count
    integer(i64), intent(out) :: consumed_token_count
    integer(i32), intent(out) :: status_code
    type(c_ptr), intent(in), optional :: workspace_buffer
    integer(i64), intent(in), optional :: workspace_bytes
    integer(i32), intent(in), optional :: token_values(:)
    integer(i8), intent(in), optional  :: modal_bytes(:)
    integer(i8), intent(out), optional  :: context_bytes(:)
    integer(i32), intent(out), optional :: context_byte_count
    integer(c_int64_t)        :: consumed_token_count_c
    integer(c_int32_t)        :: context_byte_count_c
    integer(c_int32_t)        :: status_code_c
    type(c_ptr)               :: workspace_buffer_c
    type(c_ptr)               :: token_values_c
    type(c_ptr)               :: modal_bytes_c
    type(c_ptr)               :: context_bytes_c
    integer(c_int64_t)        :: workspace_bytes_c
    integer(c_int64_t)        :: token_count_c
    integer(c_int64_t)        :: modal_byte_count_c
    integer(c_i32), allocatable, target :: token_values_copy(:)
    integer(c_i8), allocatable, target  :: modal_bytes_copy(:)
    integer(c_i8), target               :: context_bytes_copy(MAX_LIVE_CONTEXT_BYTES)

    workspace_buffer_c = c_null_ptr
    token_values_c = c_null_ptr
    modal_bytes_c = c_null_ptr
    context_bytes_c = c_null_ptr
    workspace_bytes_c = 0_c_int64_t
    token_count_c = int(max(0_i64, staged_tokens), kind=c_int64_t)
    modal_byte_count_c = 0_c_int64_t
    context_byte_count_c = 0_c_int32_t
    context_bytes_copy = 0_c_i8
    if (present(workspace_buffer)) workspace_buffer_c = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_c = int(max(0_i64, workspace_bytes), kind=c_int64_t)
    if (present(token_values)) then
      if (size(token_values) > 0) then
        allocate(token_values_copy(size(token_values)))
        token_values_copy = int(token_values, kind=c_i32)
        token_values_c = c_loc(token_values_copy(1))
        token_count_c = int(size(token_values_copy), kind=c_int64_t)
      end if
    end if
    if (present(modal_bytes)) then
      if (size(modal_bytes) > 0) then
        allocate(modal_bytes_copy(size(modal_bytes)))
        modal_bytes_copy = int(modal_bytes, kind=c_i8)
        modal_bytes_c = c_loc(modal_bytes_copy(1))
        modal_byte_count_c = int(size(modal_bytes_copy), kind=c_int64_t)
      end if
    end if
    if (present(context_bytes)) then
      context_bytes_c = c_loc(context_bytes_copy(1))
    end if

    call c_mizu_cuda_bridge_prefill(int(payload_hash, kind=c_int64_t), token_values_c, token_count_c, &
      modal_bytes_c, modal_byte_count_c, int(staged_modal_count, kind=c_int32_t), workspace_buffer_c, &
      workspace_bytes_c, context_bytes_c, int(MAX_LIVE_CONTEXT_BYTES, kind=c_int32_t), context_byte_count_c, &
      consumed_token_count_c, status_code_c)

    consumed_token_count = int(consumed_token_count_c, kind=i64)
    status_code = int(status_code_c, kind=i32)
    if (present(context_byte_count)) context_byte_count = int(context_byte_count_c, kind=i32)
    if (present(context_bytes)) then
      context_bytes = 0_i8
      if (context_byte_count_c > 0_c_int32_t) then
        context_bytes(1:min(size(context_bytes), int(context_byte_count_c, kind=i32))) = int( &
          context_bytes_copy(1:min(size(context_bytes), int(context_byte_count_c, kind=i32))), kind=i8)
      end if
    end if
  end subroutine launch_cuda_prefill

  subroutine launch_cuda_projector(payload_hash, modal_byte_count, placeholder_count, embedding_count, &
                                   status_code, workspace_buffer, workspace_bytes)
    integer(i64), intent(in)  :: payload_hash
    integer(i64), intent(in)  :: modal_byte_count
    integer(i32), intent(in)  :: placeholder_count
    integer(i64), intent(out) :: embedding_count
    integer(i32), intent(out) :: status_code
    type(c_ptr), intent(in), optional :: workspace_buffer
    integer(i64), intent(in), optional :: workspace_bytes
    integer(c_int64_t)        :: embedding_count_c
    integer(c_int32_t)        :: status_code_c
    type(c_ptr)               :: workspace_buffer_c
    integer(c_int64_t)        :: workspace_bytes_c

    workspace_buffer_c = c_null_ptr
    workspace_bytes_c = 0_c_int64_t
    if (present(workspace_buffer)) workspace_buffer_c = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_c = int(max(0_i64, workspace_bytes), kind=c_int64_t)

    call c_mizu_cuda_bridge_projector(int(payload_hash, kind=c_int64_t), int(modal_byte_count, kind=c_int64_t), &
      int(placeholder_count, kind=c_int32_t), workspace_buffer_c, workspace_bytes_c, &
      embedding_count_c, status_code_c)

    embedding_count = int(embedding_count_c, kind=i64)
    status_code = int(status_code_c, kind=i32)
  end subroutine launch_cuda_projector

  subroutine launch_cuda_decode(payload_hash, kv_before, token_budget, emitted_token_count, token_value, &
                                stop_reason, status_code, workspace_buffer, workspace_bytes, context_bytes, &
                                context_byte_count, updated_context_bytes, updated_context_byte_count)
    integer(i64), intent(in)  :: payload_hash
    integer(i64), intent(in)  :: kv_before
    integer(i64), intent(in)  :: token_budget
    integer(i64), intent(out) :: emitted_token_count
    integer(i32), intent(out) :: token_value
    integer(i32), intent(out) :: stop_reason
    integer(i32), intent(out) :: status_code
    type(c_ptr), intent(in), optional :: workspace_buffer
    integer(i64), intent(in), optional :: workspace_bytes
    integer(i8), intent(in), optional  :: context_bytes(:)
    integer(i32), intent(in), optional :: context_byte_count
    integer(i8), intent(out), optional :: updated_context_bytes(:)
    integer(i32), intent(out), optional :: updated_context_byte_count
    integer(c_int64_t)        :: emitted_token_count_c
    integer(c_int32_t)        :: token_value_c
    integer(c_int32_t)        :: stop_reason_c
    integer(c_int32_t)        :: status_code_c
    integer(c_int32_t)        :: context_byte_count_c
    integer(c_int32_t)        :: updated_context_byte_count_c
    type(c_ptr)               :: workspace_buffer_c
    type(c_ptr)               :: context_bytes_c
    type(c_ptr)               :: updated_context_bytes_c
    integer(c_int64_t)        :: workspace_bytes_c
    integer(c_i8), allocatable, target  :: context_bytes_copy(:)
    integer(c_i8), target               :: updated_context_bytes_copy(MAX_LIVE_CONTEXT_BYTES)

    workspace_buffer_c = c_null_ptr
    context_bytes_c = c_null_ptr
    updated_context_bytes_c = c_null_ptr
    workspace_bytes_c = 0_c_int64_t
    context_byte_count_c = 0_c_int32_t
    updated_context_byte_count_c = 0_c_int32_t
    updated_context_bytes_copy = 0_c_i8
    if (present(workspace_buffer)) workspace_buffer_c = workspace_buffer
    if (present(workspace_bytes)) workspace_bytes_c = int(max(0_i64, workspace_bytes), kind=c_int64_t)
    if (present(context_bytes)) then
      if (size(context_bytes) > 0) then
        allocate(context_bytes_copy(size(context_bytes)))
        context_bytes_copy = int(context_bytes, kind=c_i8)
        context_bytes_c = c_loc(context_bytes_copy(1))
        context_byte_count_c = int(size(context_bytes_copy), kind=c_int32_t)
      end if
    end if
    if (present(context_byte_count)) then
      context_byte_count_c = int(max(0_i32, min(context_byte_count, int(MAX_LIVE_CONTEXT_BYTES, kind=i32))), &
        kind=c_int32_t)
    end if
    if (present(updated_context_bytes)) updated_context_bytes_c = c_loc(updated_context_bytes_copy(1))

    call c_mizu_cuda_bridge_decode(int(payload_hash, kind=c_int64_t), int(kv_before, kind=c_int64_t), &
      int(token_budget, kind=c_int64_t), context_bytes_c, context_byte_count_c, workspace_buffer_c, &
      workspace_bytes_c, updated_context_bytes_c, int(MAX_LIVE_CONTEXT_BYTES, kind=c_int32_t), &
      updated_context_byte_count_c, emitted_token_count_c, token_value_c, stop_reason_c, status_code_c)

    emitted_token_count = int(emitted_token_count_c, kind=i64)
    token_value = int(token_value_c, kind=i32)
    stop_reason = int(stop_reason_c, kind=i32)
    status_code = int(status_code_c, kind=i32)
    if (present(updated_context_byte_count)) updated_context_byte_count = int(updated_context_byte_count_c, kind=i32)
    if (present(updated_context_bytes)) then
      updated_context_bytes = 0_i8
      if (updated_context_byte_count_c > 0_c_int32_t) then
        updated_context_bytes(1:min(size(updated_context_bytes), int(updated_context_byte_count_c, kind=i32))) = int( &
          updated_context_bytes_copy(1:min(size(updated_context_bytes), int(updated_context_byte_count_c, kind=i32))), &
          kind=i8)
      end if
    end if
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
