module mod_cuda_capability
  use mod_kinds,                 only: i32, i64, GIGABYTE
  use mod_status,                only: MIZU_STATUS_OK
  use mod_types,                 only: MIZU_BACKEND_FAMILY_CUDA, MIZU_BACKEND_MASK_NONE, &
                                       MIZU_BACKEND_MASK_CUDA, MIZU_DTYPE_U8, MIZU_DTYPE_BF16
  use mod_backend_contract,      only: capability_probe_request, capability_probe_result
  use mod_backend_probe_support, only: read_boolean_env_override, command_succeeds
  use mod_cuda_bridge,           only: cuda_device_info, query_cuda_device_info

  implicit none

  private
  public :: probe_cuda_backend

contains

  subroutine probe_cuda_backend(request, result, status_code)
    type(capability_probe_request), intent(in)  :: request
    type(capability_probe_result), intent(out)  :: result
    integer(i32), intent(out)                   :: status_code
    logical                                     :: cuda_available
    logical                                     :: has_override
    type(cuda_device_info)                      :: device_info
    integer(i32)                                :: bridge_status_code

    result = capability_probe_result()
    result%descriptor%family = MIZU_BACKEND_FAMILY_CUDA
    result%descriptor%backend_name = "cuda"
    result%descriptor%device_name = "nvidia_cuda_device"

    call read_boolean_env_override("MIZU_FORCE_CUDA_AVAILABLE", has_override, cuda_available)
    device_info = cuda_device_info()
    bridge_status_code = MIZU_STATUS_OK

    if (.not. has_override) then
      call query_cuda_device_info(device_info, bridge_status_code)
      cuda_available = (bridge_status_code == MIZU_STATUS_OK .and. device_info%is_available)
      if (.not. cuda_available) cuda_available = command_succeeds("nvidia-smi -L >/dev/null 2>&1")
    else if (cuda_available) then
      call query_cuda_device_info(device_info, bridge_status_code)
    end if

    result%descriptor%route_mask = MIZU_BACKEND_MASK_NONE
    if (cuda_available) result%descriptor%route_mask = MIZU_BACKEND_MASK_CUDA
    result%descriptor%is_available = (result%descriptor%route_mask /= MIZU_BACKEND_MASK_NONE)
    if (device_info%is_available .and. len_trim(device_info%device_name) > 0) then
      result%descriptor%device_name = trim(device_info%device_name)
    end if

    if (iand(request%allowed_backend_mask, MIZU_BACKEND_MASK_CUDA) == 0_i64) then
      result%descriptor%is_available = .false.
      result%descriptor%route_mask = MIZU_BACKEND_MASK_NONE
    end if

    if (result%descriptor%is_available) then
      result%constraints%supported_dtype_mask = ior(dtype_mask(MIZU_DTYPE_U8), dtype_mask(MIZU_DTYPE_BF16))
      result%constraints%supported_op_mask = supported_op_mask()
      result%constraints%max_workspace_bytes = estimate_workspace_bytes(device_info%total_memory_bytes)
      result%constraints%max_sequence_tokens = estimate_sequence_limit(device_info%total_memory_bytes)
      result%constraints%planner_version = estimate_planner_version(device_info)
    end if

    status_code = MIZU_STATUS_OK
    result%status_code = status_code
  end subroutine probe_cuda_backend

  pure integer(i64) function dtype_mask(dtype_value) result(mask_value)
    integer(i32), intent(in) :: dtype_value

    mask_value = shiftl(1_i64, max(0_i32, dtype_value))
  end function dtype_mask

  pure integer(i64) function supported_op_mask() result(mask_value)
    mask_value = ior(shiftl(1_i64, 0), shiftl(1_i64, 1))
    mask_value = ior(mask_value, shiftl(1_i64, 2))
  end function supported_op_mask

  pure integer(i64) function estimate_workspace_bytes(total_memory_bytes) result(workspace_bytes)
    integer(i64), intent(in) :: total_memory_bytes
    integer(i64)             :: reserved_bytes

    if (total_memory_bytes <= 0_i64) then
      workspace_bytes = 4_i64 * GIGABYTE
      return
    end if

    reserved_bytes = max(1_i64 * GIGABYTE, total_memory_bytes / 2_i64)
    workspace_bytes = min(reserved_bytes, 12_i64 * GIGABYTE)
  end function estimate_workspace_bytes

  pure integer(i64) function estimate_sequence_limit(total_memory_bytes) result(sequence_limit)
    integer(i64), intent(in) :: total_memory_bytes

    if (total_memory_bytes >= 16_i64 * GIGABYTE) then
      sequence_limit = 262144_i64
    else if (total_memory_bytes >= 8_i64 * GIGABYTE) then
      sequence_limit = 131072_i64
    else
      sequence_limit = 65536_i64
    end if
  end function estimate_sequence_limit

  pure integer(i64) function estimate_planner_version(device_info) result(planner_version)
    type(cuda_device_info), intent(in) :: device_info

    planner_version = max(1_i64, int(device_info%compute_major, kind=i64) * 10_i64 + &
      int(device_info%compute_minor, kind=i64))
  end function estimate_planner_version

end module mod_cuda_capability
