program test_cuda_executor
  use iso_c_binding,     only: c_associated, c_f_pointer
  use mod_kinds,         only: c_i8, i8, i32, i64
  use mod_status,        only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_STATE
  use mod_types,         only: MIZU_STOP_REASON_NONE, workspace_state, MAX_LIVE_CONTEXT_BYTES
  use mod_cuda_executor, only: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode
  use mod_workspace,     only: initialize_workspace, reserve_workspace_bytes, release_workspace_bytes, &
                               reset_workspace

  implicit none

  type(workspace_state) :: workspace
  integer(c_i8), pointer :: workspace_view(:)
  integer(c_i8) :: prefill_scratch_a(16)
  integer(c_i8) :: prefill_scratch_b(16)
  integer(i8)  :: context_bytes_a(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: context_bytes_b(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: decode_context_bytes(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: updated_context_bytes(MAX_LIVE_CONTEXT_BYTES)
  integer(i64) :: embedding_count
  integer(i64) :: consumed_token_count
  integer(i64) :: emitted_token_count
  integer(i32) :: token_value
  integer(i32) :: token_value_with_other_context
  integer(i32) :: context_byte_count_a
  integer(i32) :: context_byte_count_b
  integer(i32) :: decode_context_byte_count
  integer(i32) :: updated_context_byte_count
  integer(i32) :: stop_reason
  integer(i32) :: status_code
  integer      :: shell_status
  integer(i32) :: token_values_a(7)
  integer(i32) :: token_values_b(7)
  integer(i8)  :: modal_bytes_a(6)
  integer(i8)  :: modal_bytes_b(6)
  character(len=*), parameter :: cache_root = "/tmp/mizu_test_cuda_executor"
  character(len=*), parameter :: projector_path = "artifacts/cuda/cuda/projector/test.mm"
  character(len=*), parameter :: prefill_path = "artifacts/cuda/cuda/plans/prefill/test.plan"
  character(len=*), parameter :: decode_path = "artifacts/cuda/cuda/plans/decode/test.plan"

  token_values_a = [3_i32, 5_i32, 7_i32, 11_i32, 13_i32, 17_i32, 19_i32]
  token_values_b = [2_i32, 4_i32, 6_i32, 8_i32, 10_i32, 12_i32, 14_i32]
  modal_bytes_a = [1_i8, 3_i8, 5_i8, 7_i8, 9_i8, 11_i8]
  modal_bytes_b = [2_i8, 4_i8, 6_i8, 8_i8, 10_i8, 12_i8]
  context_bytes_a = 0_i8
  context_bytes_b = 0_i8
  decode_context_bytes = 0_i8
  updated_context_bytes = 0_i8
  context_byte_count_a = 0_i32
  context_byte_count_b = 0_i32
  decode_context_byte_count = 0_i32
  updated_context_byte_count = 0_i32

  shell_status = 0
  call execute_command_line("rm -rf " // cache_root // " && mkdir -p " // cache_root // &
    "/artifacts/cuda/cuda/projector " // cache_root // "/artifacts/cuda/cuda/plans/prefill " // &
    cache_root // "/artifacts/cuda/cuda/plans/decode", &
    exitstat=shell_status)
  call expect_equal_i32("cuda executor fixture dirs should be created", int(shell_status, kind=i32), 0_i32)

  open(unit=9, file=trim(cache_root) // "/" // trim(projector_path), status="replace", action="write")
  write(9, "(A)") "candidate=projector;stage=2;workspace=8388608;format=cuda_u8_bf16_projector_plan_v1"
  close(9)

  open(unit=10, file=trim(cache_root) // "/" // trim(prefill_path), status="replace", action="write")
  write(10, "(A)") "candidate=prefill;stage=3;format=cuda_bf16_prefill_plan_v1"
  close(10)

  open(unit=11, file=trim(cache_root) // "/" // trim(decode_path), status="replace", action="write")
  write(11, "(A)") "candidate=decode;stage=4;format=cuda_bf16_decode_plan_v1"
  close(11)

  call initialize_workspace(workspace, 0_i64)
  call reserve_workspace_bytes(workspace, 64_i64, status_code)
  call expect_equal_i32("workspace reservation should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("executor workspace should allocate a host buffer", c_associated(workspace%host_buffer))
  call c_f_pointer(workspace%host_buffer, workspace_view, [int(workspace%bytes_reserved)])

  workspace_view = 0_c_i8
  call execute_cuda_projector(cache_root, projector_path, 8192_i64, 1_i32, 12345_i64, embedding_count, &
    status_code, workspace%host_buffer, workspace%bytes_in_use)
  call expect_equal_i32("cuda projector should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("cuda projector should emit at least one embedding slot", embedding_count >= 1_i64)
  call expect_true("cuda projector should stamp workspace scratch bytes", any(workspace_view(1:16) /= 0_c_i8))

  workspace_view = 0_c_i8
  call execute_cuda_prefill(cache_root, prefill_path, 7_i64, 1_i32, 0_i64, 0_i64, consumed_token_count, &
    status_code, workspace%host_buffer, workspace%bytes_in_use, token_values_a, modal_bytes_a, context_bytes_a, &
    context_byte_count_a)
  call expect_equal_i32("cuda prefill should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda prefill should consume staged tokens", consumed_token_count, 7_i64)
  call expect_true("cuda prefill should stamp workspace scratch bytes", any(workspace_view(1:16) /= 0_c_i8))
  call expect_true("cuda prefill should emit a live context buffer", context_byte_count_a > 32_i32)
  call expect_equal_i32("cuda prefill context should start with magic M", int(context_bytes_a(1), kind=i32), iachar("M"))
  call expect_equal_i32("cuda prefill context should start with magic Z", int(context_bytes_a(2), kind=i32), iachar("Z"))
  call expect_equal_i32("cuda prefill context should declare version 1", int(context_bytes_a(5), kind=i32), 1_i32)
  call expect_equal_i32("cuda prefill context should declare prefill kind", int(context_bytes_a(6), kind=i32), 1_i32)
  prefill_scratch_a = workspace_view(1:16)

  workspace_view = 0_c_i8
  call execute_cuda_prefill(cache_root, prefill_path, 7_i64, 1_i32, 0_i64, 0_i64, consumed_token_count, &
    status_code, workspace%host_buffer, workspace%bytes_in_use, token_values_b, modal_bytes_b, context_bytes_b, &
    context_byte_count_b)
  call expect_equal_i32("cuda prefill with different tensors should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda prefill should still consume staged tokens with different tensors", &
    consumed_token_count, 7_i64)
  call expect_true("cuda prefill should emit a second live context buffer", context_byte_count_b > 32_i32)
  prefill_scratch_b = workspace_view(1:16)
  call expect_true("cuda prefill should reflect tensor content in workspace scratch", &
    any(prefill_scratch_a /= prefill_scratch_b))
  call expect_true("cuda prefill should produce different context buffers for different tensors", &
    any(context_bytes_a /= context_bytes_b))

  workspace_view = 0_c_i8
  call execute_cuda_decode(cache_root, decode_path, 42_i64, 1_i64, emitted_token_count, token_value, stop_reason, &
    status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_a, context_byte_count_a, &
    updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda decode should emit one token", emitted_token_count, 1_i64)
  call expect_true("cuda decode should generate a positive token id", token_value > 0_i32)
  call expect_equal_i32("cuda decode stop reason should stay none", stop_reason, MIZU_STOP_REASON_NONE)
  call expect_true("cuda decode should stamp workspace scratch bytes", any(workspace_view(1:16) /= 0_c_i8))
  call expect_true("cuda decode should emit an updated context buffer", updated_context_byte_count > 32_i32)
  call expect_equal_i32("cuda decode context should keep magic M", int(updated_context_bytes(1), kind=i32), iachar("M"))
  call expect_equal_i32("cuda decode context should declare decode kind", int(updated_context_bytes(6), kind=i32), 2_i32)
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 42_i64, 1_i64, emitted_token_count, token_value_with_other_context, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_b, &
    context_byte_count_b, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("cuda decode with another context should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("cuda decode should reflect direct context buffer identity", &
    token_value_with_other_context /= token_value)

  open(unit=12, file=trim(cache_root) // "/" // trim(decode_path), status="replace", action="write")
  write(12, "(A)") "candidate=decode;stage=4;format=cuda_bf16_decode_plan_v2"
  close(12)

  call execute_cuda_decode(cache_root, decode_path, 43_i64, 1_i64, emitted_token_count, token_value_with_other_context, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, decode_context_bytes, &
    decode_context_byte_count, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("cuda decode should reject a context from another decode artifact", &
    status_code, MIZU_STATUS_INVALID_STATE)

  context_bytes_b(20) = context_bytes_b(20) + 1_i8
  call execute_cuda_decode(cache_root, decode_path, 42_i64, 1_i64, emitted_token_count, token_value_with_other_context, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_b, &
    context_byte_count_b, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("cuda decode should reject a corrupted context payload", status_code, MIZU_STATUS_INVALID_STATE)

  call release_workspace_bytes(workspace)
  call reset_workspace(workspace)
  call execute_command_line("rm -rf " // cache_root)
  write(*, "(A)") "test_cuda_executor: PASS"

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

end program test_cuda_executor
