program test_cuda_planner
  use mod_kinds,            only: i32, i64
  use mod_status,           only: MIZU_STATUS_OK
  use mod_types,            only: MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR, &
                                  MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, &
                                  MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                                  MIZU_BACKEND_MASK_CUDA
  use mod_backend_contract, only: plan_request, planner_result, initialize_plan_request, &
                                  planner_result_is_success, OP_FAMILY_NONE, OP_FAMILY_PROJECTOR, &
                                  OP_FAMILY_PREFILL, OP_FAMILY_DECODE
  use mod_cuda_planner,     only: CUDA_ARTIFACT_PAYLOAD_LEN, plan_cuda_stage, &
                                  build_cuda_artifact_payload_text

  implicit none

  type(plan_request)                 :: request
  type(planner_result)               :: result
  character(len=CUDA_ARTIFACT_PAYLOAD_LEN) :: payload_text
  integer(i64)                       :: payload_bytes
  integer(i32)                       :: status_code

  call initialize_plan_request(request, MIZU_STAGE_MODEL_LOAD, OP_FAMILY_NONE, &
    MIZU_MODEL_FAMILY_QWEN3_5, MIZU_BACKEND_MASK_CUDA)
  request%shape_signature(1) = 123_i64
  call plan_cuda_stage(request, result, status_code)
  call expect_equal_i32("model load planner should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("model load planner result should be successful", planner_result_is_success(result))
  call expect_equal_string("model load format", trim(result%chosen_plan%pack_format), &
    "cuda_bf16_weight_pack_v1")
  call expect_true("model load workspace should be positive", result%chosen_plan%workspace_bytes > 0_i64)

  call initialize_plan_request(request, MIZU_STAGE_PROJECTOR, OP_FAMILY_PROJECTOR, &
    MIZU_MODEL_FAMILY_QWEN3_5, MIZU_BACKEND_MASK_CUDA)
  request%shape_signature(1) = 4096_i64
  call plan_cuda_stage(request, result, status_code)
  call expect_equal_string("projector format", trim(result%chosen_plan%pack_format), &
    "cuda_u8_bf16_projector_plan_v1")

  call initialize_plan_request(request, MIZU_STAGE_PREFILL, OP_FAMILY_PREFILL, &
    MIZU_MODEL_FAMILY_GEMMA4, MIZU_BACKEND_MASK_CUDA)
  request%shape_signature(1:3) = [128_i64, 64_i64, 1_i64]
  request%token_count = 64_i64
  call plan_cuda_stage(request, result, status_code)
  call expect_equal_string("prefill format", trim(result%chosen_plan%pack_format), &
    "cuda_bf16_prefill_plan_v1")
  call build_cuda_artifact_payload_text(request, result%chosen_plan, "candidate:prefill:cuda", &
    payload_text, payload_bytes)
  call expect_true("prefill payload should mention workspace", index(payload_text, "workspace=") > 0)
  call expect_true("prefill payload bytes should be positive", payload_bytes > 0_i64)

  call initialize_plan_request(request, MIZU_STAGE_DECODE, OP_FAMILY_DECODE, &
    MIZU_MODEL_FAMILY_GEMMA4, MIZU_BACKEND_MASK_CUDA)
  request%shape_signature(1:3) = [512_i64, 1_i64, 1_i64]
  request%token_count = 1_i64
  call plan_cuda_stage(request, result, status_code)
  call expect_equal_string("decode format", trim(result%chosen_plan%pack_format), &
    "cuda_bf16_decode_plan_v1")

  write(*, "(A)") "test_cuda_planner: PASS"

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

  subroutine expect_equal_string(label, actual, expected)
    character(len=*), intent(in) :: label
    character(len=*), intent(in) :: actual
    character(len=*), intent(in) :: expected

    if (trim(actual) /= trim(expected)) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_equal_string

end program test_cuda_planner
