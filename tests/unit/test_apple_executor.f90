program test_apple_executor
  use iso_c_binding,      only: c_associated, c_f_pointer
  use mod_kinds,          only: c_i8, i8, i32, i64
  use mod_status,         only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_STATE
  use mod_types,          only: MIZU_STOP_REASON_NONE, MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, &
                                MIZU_EXEC_ROUTE_ANE, MIZU_EXEC_ROUTE_METAL, workspace_state, &
                                MAX_LIVE_CONTEXT_BYTES
  use mod_apple_executor, only: execute_apple_projector, execute_apple_prefill, execute_apple_decode, &
                                apple_context_bytes_are_valid, extract_apple_context_lineage, &
                                extract_apple_context_snapshot
  use mod_workspace,      only: initialize_workspace, reserve_workspace_bytes, reset_workspace

  implicit none

  type(workspace_state) :: workspace
  integer(c_i8), pointer :: workspace_view(:)
  integer(i32) :: status_code
  integer(i32) :: stop_reason
  integer(i32) :: token_value_ane
  integer(i32) :: token_value_metal
  integer(i32) :: producer_stage
  integer(i32) :: execution_route
  integer(i32) :: last_token
  integer(i32) :: context_byte_count_ane
  integer(i32) :: context_byte_count_metal
  integer(i32) :: updated_context_byte_count_ane
  integer(i32) :: updated_context_byte_count_metal
  integer(i64) :: projector_embeddings_ane
  integer(i64) :: projector_embeddings_metal
  integer(i64) :: consumed_token_count
  integer(i64) :: emitted_token_count
  integer(i64) :: artifact_hash
  integer(i64) :: prefill_artifact_hash_ane
  integer(i64) :: prefill_artifact_hash_metal
  integer(i64) :: decode_artifact_hash_ane
  integer(i64) :: decode_artifact_hash_metal
  integer(i64) :: token_digest
  integer(i64) :: modal_digest
  integer(i64) :: kv_token_count
  integer(i64) :: decode_step_count
  integer(i64) :: state_digest
  integer      :: shell_status
  integer(i32) :: token_values(4)
  integer(i8)  :: modal_bytes(6)
  integer(i8)  :: context_bytes_ane(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: context_bytes_metal(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: updated_context_bytes_ane(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: updated_context_bytes_metal(MAX_LIVE_CONTEXT_BYTES)
  logical      :: lineage_known
  logical      :: snapshot_valid
  character(len=*), parameter :: cache_root = "/tmp/mizu_test_apple_executor"
  character(len=*), parameter :: ane_projector_path = "artifacts/apple/ane/projector/test.mm"
  character(len=*), parameter :: ane_prefill_path = "artifacts/apple/ane/plans/prefill/test.plan"
  character(len=*), parameter :: ane_decode_path = "artifacts/apple/ane/plans/decode/test.plan"
  character(len=*), parameter :: metal_projector_path = "artifacts/apple/metal/projector/test.mm"
  character(len=*), parameter :: metal_prefill_path = "artifacts/apple/metal/plans/prefill/test.plan"
  character(len=*), parameter :: metal_decode_path = "artifacts/apple/metal/plans/decode/test.plan"

  token_values = [4_i32, 8_i32, 12_i32, 16_i32]
  modal_bytes = [1_i8, 2_i8, 3_i8, 4_i8, 5_i8, 6_i8]
  context_bytes_ane = 0_i8
  context_bytes_metal = 0_i8
  updated_context_bytes_ane = 0_i8
  updated_context_bytes_metal = 0_i8
  context_byte_count_ane = 0_i32
  context_byte_count_metal = 0_i32
  updated_context_byte_count_ane = 0_i32
  updated_context_byte_count_metal = 0_i32

  shell_status = 0
  call execute_command_line("rm -rf " // cache_root // " && mkdir -p " // cache_root // &
    "/artifacts/apple/ane/projector " // cache_root // "/artifacts/apple/ane/plans/prefill " // &
    cache_root // "/artifacts/apple/ane/plans/decode " // cache_root // "/artifacts/apple/metal/projector " // &
    cache_root // "/artifacts/apple/metal/plans/prefill " // cache_root // "/artifacts/apple/metal/plans/decode", &
    exitstat=shell_status)
  call expect_equal_i32("apple executor fixture dirs should be created", int(shell_status, kind=i32), 0_i32)

  open(unit=9, file=trim(cache_root) // "/" // trim(ane_projector_path), status="replace", action="write")
  write(9, "(A)") "candidate=projector;stage=2;route=apple_ane;format=apple_ane_u8_bf16_projector_plan_v1"
  close(9)

  open(unit=10, file=trim(cache_root) // "/" // trim(ane_prefill_path), status="replace", action="write")
  write(10, "(A)") "candidate=prefill;stage=3;route=apple_ane;format=apple_ane_bf16_prefill_plan_v1"
  close(10)

  open(unit=11, file=trim(cache_root) // "/" // trim(ane_decode_path), status="replace", action="write")
  write(11, "(A)") "candidate=decode;stage=4;route=apple_ane;format=apple_ane_bf16_decode_plan_v1"
  close(11)

  open(unit=12, file=trim(cache_root) // "/" // trim(metal_projector_path), status="replace", action="write")
  write(12, "(A)") "candidate=projector;stage=2;route=apple_metal;format=apple_metal_u8_bf16_projector_plan_v1"
  close(12)

  open(unit=13, file=trim(cache_root) // "/" // trim(metal_prefill_path), status="replace", action="write")
  write(13, "(A)") "candidate=prefill;stage=3;route=apple_metal;format=apple_metal_bf16_prefill_plan_v1"
  close(13)

  open(unit=14, file=trim(cache_root) // "/" // trim(metal_decode_path), status="replace", action="write")
  write(14, "(A)") "candidate=decode;stage=4;route=apple_metal;format=apple_metal_bf16_decode_plan_v1"
  close(14)

  call initialize_workspace(workspace, 0_i64)
  call reserve_workspace_bytes(workspace, 64_i64, status_code)
  call expect_equal_i32("workspace reservation should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("apple executor workspace should allocate a host buffer", c_associated(workspace%host_buffer))
  call c_f_pointer(workspace%host_buffer, workspace_view, [int(workspace%bytes_reserved)])

  workspace_view = 0_c_i8
  call execute_apple_projector(cache_root, ane_projector_path, MIZU_EXEC_ROUTE_ANE, 4096_i64, 1_i32, 77_i64, &
    projector_embeddings_ane, status_code, workspace%host_buffer, workspace%bytes_in_use)
  call expect_equal_i32("apple ANE projector should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("apple ANE projector should emit embeddings", projector_embeddings_ane > 0_i64)
  call expect_true("apple ANE projector should stamp workspace", any(workspace_view(1:16) /= 0_c_i8))

  workspace_view = 0_c_i8
  call execute_apple_projector(cache_root, metal_projector_path, MIZU_EXEC_ROUTE_METAL, 4096_i64, 1_i32, 77_i64, &
    projector_embeddings_metal, status_code, workspace%host_buffer, workspace%bytes_in_use)
  call expect_equal_i32("apple Metal projector should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("apple Metal projector should emit embeddings", projector_embeddings_metal > 0_i64)
  call expect_true("apple Metal projector should stamp workspace", any(workspace_view(1:16) /= 0_c_i8))

  workspace_view = 0_c_i8
  call execute_apple_prefill(cache_root, ane_prefill_path, MIZU_EXEC_ROUTE_ANE, 4_i64, 1_i32, 0_i64, 0_i64, &
    consumed_token_count, status_code, workspace%host_buffer, workspace%bytes_in_use, token_values, modal_bytes, &
    context_bytes_ane, context_byte_count_ane, context_artifact_hash=prefill_artifact_hash_ane)
  call expect_equal_i32("apple ANE prefill should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("apple ANE prefill should consume staged tokens", consumed_token_count, 4_i64)
  call expect_true("apple ANE prefill should emit a valid context", &
    apple_context_bytes_are_valid(context_bytes_ane, context_byte_count_ane))
  call expect_true("apple ANE prefill should stamp workspace", any(workspace_view(1:16) /= 0_c_i8))
  call extract_apple_context_snapshot(context_bytes_ane, context_byte_count_ane, producer_stage, execution_route, &
    artifact_hash, token_digest, modal_digest, kv_token_count, decode_step_count, last_token, state_digest, &
    snapshot_valid)
  call expect_true("apple ANE prefill snapshot should be readable", snapshot_valid)
  call expect_equal_i32("apple ANE prefill should report prefill stage", producer_stage, MIZU_STAGE_PREFILL)
  call expect_equal_i32("apple ANE prefill should retain route lineage", execution_route, MIZU_EXEC_ROUTE_ANE)
  call expect_equal_i64("apple ANE prefill should seed kv tokens from staged tokens", kv_token_count, 4_i64)
  call expect_equal_i64("apple ANE prefill should start decode step count at zero", decode_step_count, 0_i64)
  call expect_equal_i32("apple ANE prefill should retain the last staged token", last_token, token_values(4))
  call expect_true("apple ANE prefill should retain a nonzero artifact hash", artifact_hash /= 0_i64)
  call expect_equal_i64("apple ANE prefill should surface its artifact hash", artifact_hash, prefill_artifact_hash_ane)

  call execute_apple_prefill(cache_root, metal_prefill_path, MIZU_EXEC_ROUTE_METAL, 4_i64, 1_i32, 0_i64, 0_i64, &
    consumed_token_count, status_code, workspace%host_buffer, workspace%bytes_in_use, token_values, modal_bytes, &
    context_bytes_metal, context_byte_count_metal, context_artifact_hash=prefill_artifact_hash_metal)
  call expect_equal_i32("apple Metal prefill should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("apple Metal prefill should emit a valid context", &
    apple_context_bytes_are_valid(context_bytes_metal, context_byte_count_metal))
  call extract_apple_context_snapshot(context_bytes_metal, context_byte_count_metal, producer_stage, execution_route, &
    artifact_hash, token_digest, modal_digest, kv_token_count, decode_step_count, last_token, state_digest, &
    snapshot_valid)
  call expect_true("apple Metal prefill snapshot should be readable", snapshot_valid)
  call expect_equal_i32("apple Metal prefill should retain route lineage", execution_route, MIZU_EXEC_ROUTE_METAL)
  call expect_true("apple Metal prefill should diverge from the ANE context image", &
    any(context_bytes_metal(1:context_byte_count_metal) /= context_bytes_ane(1:context_byte_count_ane)))

  workspace_view = 0_c_i8
  call execute_apple_decode(cache_root, ane_decode_path, MIZU_EXEC_ROUTE_ANE, 4_i64, 1_i64, emitted_token_count, &
    token_value_ane, stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_ane, &
    context_byte_count_ane, updated_context_bytes_ane, updated_context_byte_count_ane, &
    context_artifact_hash=decode_artifact_hash_ane)
  call expect_equal_i32("apple ANE decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("apple ANE decode should emit one token", emitted_token_count, 1_i64)
  call expect_equal_i32("apple ANE decode should not request stop", stop_reason, MIZU_STOP_REASON_NONE)
  call expect_true("apple ANE decode should emit a valid updated context", &
    apple_context_bytes_are_valid(updated_context_bytes_ane, updated_context_byte_count_ane))
  call expect_true("apple ANE decode should stamp workspace", any(workspace_view(1:16) /= 0_c_i8))
  call extract_apple_context_lineage(updated_context_bytes_ane, updated_context_byte_count_ane, producer_stage, &
    execution_route, artifact_hash, lineage_known)
  call expect_true("apple ANE decode lineage should be readable", lineage_known)
  call expect_equal_i32("apple ANE decode should switch producer stage to decode", producer_stage, MIZU_STAGE_DECODE)
  call expect_equal_i32("apple ANE decode should keep route lineage", execution_route, MIZU_EXEC_ROUTE_ANE)
  call expect_equal_i64("apple ANE decode should retain decode artifact lineage", artifact_hash, decode_artifact_hash_ane)
  call extract_apple_context_snapshot(updated_context_bytes_ane, updated_context_byte_count_ane, producer_stage, &
    execution_route, artifact_hash, token_digest, modal_digest, kv_token_count, decode_step_count, last_token, &
    state_digest, snapshot_valid)
  call expect_true("apple ANE decode snapshot should be readable", snapshot_valid)
  call expect_equal_i64("apple ANE decode should advance kv tokens", kv_token_count, 5_i64)
  call expect_equal_i64("apple ANE decode should advance decode steps", decode_step_count, 1_i64)
  call expect_equal_i32("apple ANE decode should retain the emitted token in the snapshot", last_token, token_value_ane)

  call execute_apple_decode(cache_root, metal_decode_path, MIZU_EXEC_ROUTE_METAL, 4_i64, 1_i64, emitted_token_count, &
    token_value_metal, stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_metal, &
    context_byte_count_metal, updated_context_bytes_metal, updated_context_byte_count_metal, &
    context_artifact_hash=decode_artifact_hash_metal)
  call expect_equal_i32("apple Metal decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("apple Metal decode should emit a valid updated context", &
    apple_context_bytes_are_valid(updated_context_bytes_metal, updated_context_byte_count_metal))
  call expect_true("apple Metal decode should diverge from the ANE decode context image", &
    any(updated_context_bytes_metal(1:updated_context_byte_count_metal) /= &
        updated_context_bytes_ane(1:updated_context_byte_count_ane)))

  call execute_apple_decode(cache_root, metal_decode_path, MIZU_EXEC_ROUTE_METAL, 5_i64, 1_i64, emitted_token_count, &
    token_value_metal, stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, &
    updated_context_bytes_ane, updated_context_byte_count_ane, updated_context_bytes_metal, &
    updated_context_byte_count_metal)
  call expect_equal_i32("apple decode should reject cross-route reuse after decode state exists", &
    status_code, MIZU_STATUS_INVALID_STATE)

  updated_context_bytes_ane(10) = updated_context_bytes_ane(10) + 1_i8
  call expect_true("corrupted Apple context bytes should fail validation", &
    .not. apple_context_bytes_are_valid(updated_context_bytes_ane, updated_context_byte_count_ane))

  call reset_workspace(workspace)
  write(*, "(A)") "test_apple_executor: PASS"

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

end program test_apple_executor
