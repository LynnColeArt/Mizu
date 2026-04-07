module mod_apple_capability
  use mod_kinds,                 only: i32, i64, GIGABYTE
  use mod_status,                only: MIZU_STATUS_OK
  use mod_types,                 only: MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_MASK_NONE, &
                                       MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL, &
                                       MIZU_DTYPE_U8, MIZU_DTYPE_BF16
  use mod_backend_contract,      only: capability_probe_request, capability_probe_result
  use mod_backend_probe_support, only: read_boolean_env_override

  implicit none

  private
  public :: probe_apple_backend

contains

  subroutine probe_apple_backend(request, result, status_code)
    type(capability_probe_request), intent(in)  :: request
    type(capability_probe_result), intent(out)  :: result
    integer(i32), intent(out)                   :: status_code
    logical                                     :: ane_available
    logical                                     :: metal_available
    logical                                     :: has_override

    result = capability_probe_result()
    result%descriptor%family = MIZU_BACKEND_FAMILY_APPLE
    result%descriptor%backend_name = "apple"
    result%descriptor%device_name = "apple_platform"

    call read_boolean_env_override("MIZU_FORCE_APPLE_ANE_AVAILABLE", has_override, ane_available)
    if (.not. has_override) ane_available = .false.

    call read_boolean_env_override("MIZU_FORCE_APPLE_METAL_AVAILABLE", has_override, metal_available)
    if (.not. has_override) then
      inquire(file="/System/Library/Frameworks/Metal.framework", exist=metal_available)
    end if

    result%descriptor%route_mask = MIZU_BACKEND_MASK_NONE
    if (ane_available) then
      result%descriptor%route_mask = ior(result%descriptor%route_mask, MIZU_BACKEND_MASK_APPLE_ANE)
      result%descriptor%device_name = "apple_neural_engine"
    end if
    if (metal_available) then
      result%descriptor%route_mask = ior(result%descriptor%route_mask, MIZU_BACKEND_MASK_APPLE_METAL)
      if (len_trim(result%descriptor%device_name) == 0 .or. &
          trim(result%descriptor%device_name) == "apple_platform") then
        result%descriptor%device_name = "apple_metal_device"
      end if
    end if

    result%descriptor%is_available = (result%descriptor%route_mask /= MIZU_BACKEND_MASK_NONE)
    if (iand(request%allowed_backend_mask, ior(MIZU_BACKEND_MASK_APPLE_ANE, MIZU_BACKEND_MASK_APPLE_METAL)) == 0_i64) then
      result%descriptor%is_available = .false.
      result%descriptor%route_mask = MIZU_BACKEND_MASK_NONE
    end if

    if (result%descriptor%is_available) then
      result%constraints%supported_dtype_mask = ior(dtype_mask(MIZU_DTYPE_U8), dtype_mask(MIZU_DTYPE_BF16))
      result%constraints%supported_op_mask = supported_op_mask()
      result%constraints%max_workspace_bytes = 1_i64 * GIGABYTE
      result%constraints%max_sequence_tokens = 65536_i64
      result%constraints%planner_version = 1_i64
    end if

    status_code = MIZU_STATUS_OK
    result%status_code = status_code
  end subroutine probe_apple_backend

  pure integer(i64) function dtype_mask(dtype_value) result(mask_value)
    integer(i32), intent(in) :: dtype_value

    mask_value = shiftl(1_i64, max(0_i32, dtype_value))
  end function dtype_mask

  pure integer(i64) function supported_op_mask() result(mask_value)
    mask_value = ior(shiftl(1_i64, 0), shiftl(1_i64, 1))
    mask_value = ior(mask_value, shiftl(1_i64, 2))
  end function supported_op_mask

end module mod_apple_capability
