program test_cuda_executor
  use mod_kinds,         only: i32, i64
  use mod_status,        only: MIZU_STATUS_OK
  use mod_types,         only: MIZU_STOP_REASON_NONE
  use mod_cuda_executor, only: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode

  implicit none

  integer(i64) :: embedding_count
  integer(i64) :: consumed_token_count
  integer(i64) :: emitted_token_count
  integer(i32) :: token_value
  integer(i32) :: stop_reason
  integer(i32) :: status_code
  integer      :: shell_status
  character(len=*), parameter :: cache_root = "/tmp/mizu_test_cuda_executor"
  character(len=*), parameter :: projector_path = "artifacts/cuda/cuda/projector/test.mm"
  character(len=*), parameter :: prefill_path = "artifacts/cuda/cuda/plans/prefill/test.plan"
  character(len=*), parameter :: decode_path = "artifacts/cuda/cuda/plans/decode/test.plan"

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

  call execute_cuda_projector(cache_root, projector_path, 8192_i64, 1_i32, embedding_count, status_code)
  call expect_equal_i32("cuda projector should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("cuda projector should emit at least one embedding slot", embedding_count >= 1_i64)

  call execute_cuda_prefill(cache_root, prefill_path, 7_i64, 1_i32, consumed_token_count, status_code)
  call expect_equal_i32("cuda prefill should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda prefill should consume staged tokens", consumed_token_count, 7_i64)

  call execute_cuda_decode(cache_root, decode_path, 42_i64, 1_i64, emitted_token_count, token_value, &
    stop_reason, status_code)
  call expect_equal_i32("cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda decode should emit one token", emitted_token_count, 1_i64)
  call expect_true("cuda decode should generate a positive token id", token_value > 0_i32)
  call expect_equal_i32("cuda decode stop reason should stay none", stop_reason, MIZU_STOP_REASON_NONE)

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
