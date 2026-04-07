module mod_cuda_planner
  use mod_kinds,            only: i32, i64, KILOBYTE, MEGABYTE
  use mod_status,           only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT
  use mod_types,            only: MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR, &
                                  MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, &
                                  MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                                  MIZU_BACKEND_FAMILY_CUDA, MIZU_EXEC_ROUTE_CUDA
  use mod_backend_contract, only: plan_request, plan_candidate, planner_result, &
                                  planner_result_is_success

  implicit none

  private
  public :: CUDA_ARTIFACT_PAYLOAD_LEN
  public :: plan_cuda_stage, build_cuda_artifact_payload_text

  integer(i32), parameter :: CUDA_ARTIFACT_PAYLOAD_LEN = 1024_i32

contains

  subroutine plan_cuda_stage(request, result, status_code)
    type(plan_request), intent(in)    :: request
    type(planner_result), intent(out) :: result
    integer(i32), intent(out)         :: status_code

    result = planner_result()
    if (request%stage_kind /= MIZU_STAGE_MODEL_LOAD .and. &
        request%stage_kind /= MIZU_STAGE_PROJECTOR .and. &
        request%stage_kind /= MIZU_STAGE_PREFILL .and. &
        request%stage_kind /= MIZU_STAGE_DECODE) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      result%status_code = status_code
      return
    end if

    result%status_code = MIZU_STATUS_OK
    result%candidate_count = 1_i32
    result%chosen_plan = plan_candidate()
    result%chosen_plan%backend_family = MIZU_BACKEND_FAMILY_CUDA
    result%chosen_plan%execution_route = MIZU_EXEC_ROUTE_CUDA
    result%chosen_plan%workspace_bytes = estimate_cuda_workspace_bytes(request)
    result%chosen_plan%pack_format = cuda_pack_format_label(request%stage_kind)

    status_code = MIZU_STATUS_OK
  end subroutine plan_cuda_stage

  subroutine build_cuda_artifact_payload_text(request, candidate, candidate_key_text, payload_text, &
                                              payload_bytes)
    type(plan_request), intent(in)    :: request
    type(plan_candidate), intent(in)  :: candidate
    character(len=*), intent(in)      :: candidate_key_text
    character(len=*), intent(out)     :: payload_text
    integer(i64), intent(out)         :: payload_bytes

    payload_text = ""
    write(payload_text, '(A,";stage=",I0,";model=",I0,";shape0=",I0,";shape1=",I0,";shape2=",I0, &
      &";tokens=",I0,";workspace=",I0,";format=",A)') &
      trim(candidate_key_text), request%stage_kind, request%model_family, &
      request%shape_signature(1), request%shape_signature(2), request%shape_signature(3), &
      request%token_count, candidate%workspace_bytes, trim(candidate%pack_format)
    payload_bytes = int(len_trim(payload_text) + 1, kind=i64)
  end subroutine build_cuda_artifact_payload_text

  integer(i64) function estimate_cuda_workspace_bytes(request) result(workspace_bytes)
    type(plan_request), intent(in) :: request
    integer(i64)                   :: base_bytes

    select case (request%stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      base_bytes = cuda_weight_pack_bytes(request%model_family)
    case (MIZU_STAGE_PROJECTOR)
      base_bytes = (8_i64 * MEGABYTE) + max(0_i64, request%shape_signature(1)) * 256_i64
    case (MIZU_STAGE_PREFILL)
      base_bytes = (16_i64 * MEGABYTE) + &
        (max(0_i64, request%shape_signature(1)) + max(0_i64, request%shape_signature(2))) * 2048_i64 + &
        max(0_i64, request%shape_signature(3)) * (8_i64 * MEGABYTE)
    case (MIZU_STAGE_DECODE)
      base_bytes = (8_i64 * MEGABYTE) + max(0_i64, request%shape_signature(1)) * 1024_i64 + &
        max(1_i64, request%token_count) * (512_i64 * KILOBYTE)
    case default
      base_bytes = 4_i64 * MEGABYTE
    end select

    workspace_bytes = align_bytes(max(1_i64 * MEGABYTE, base_bytes), 256_i64)
  end function estimate_cuda_workspace_bytes

  integer(i64) function cuda_weight_pack_bytes(model_family) result(pack_bytes)
    integer(i32), intent(in) :: model_family

    select case (model_family)
    case (MIZU_MODEL_FAMILY_QWEN3_5)
      pack_bytes = 256_i64 * MEGABYTE
    case (MIZU_MODEL_FAMILY_GEMMA4)
      pack_bytes = 512_i64 * MEGABYTE
    case default
      pack_bytes = 128_i64 * MEGABYTE
    end select
  end function cuda_weight_pack_bytes

  function cuda_pack_format_label(stage_kind) result(pack_format)
    integer(i32), intent(in)  :: stage_kind
    character(len=128)        :: pack_format

    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      pack_format = "cuda_bf16_weight_pack_v1"
    case (MIZU_STAGE_PROJECTOR)
      pack_format = "cuda_u8_bf16_projector_plan_v1"
    case (MIZU_STAGE_PREFILL)
      pack_format = "cuda_bf16_prefill_plan_v1"
    case (MIZU_STAGE_DECODE)
      pack_format = "cuda_bf16_decode_plan_v1"
    case default
      pack_format = "cuda_generic_plan_v1"
    end select
  end function cuda_pack_format_label

  integer(i64) function align_bytes(byte_count, alignment) result(aligned_bytes)
    integer(i64), intent(in) :: byte_count
    integer(i64), intent(in) :: alignment
    integer(i64)             :: rounded_bytes

    if (alignment <= 0_i64) then
      aligned_bytes = byte_count
      return
    end if

    rounded_bytes = ((max(0_i64, byte_count) + alignment - 1_i64) / alignment) * alignment
    aligned_bytes = max(alignment, rounded_bytes)
  end function align_bytes

end module mod_cuda_planner
