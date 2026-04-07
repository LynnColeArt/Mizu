program test_cuda_executor
  use iso_c_binding,     only: c_associated, c_f_pointer
  use mod_kinds,         only: c_i8, i8, i32, i64
  use mod_status,        only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_STATE
  use mod_types,         only: MIZU_STOP_REASON_NONE, MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, &
                               workspace_state, MAX_LIVE_CONTEXT_BYTES
  use mod_cuda_executor, only: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode, &
                               extract_cuda_context_state_snapshot, extract_cuda_context_window_snapshot, &
                               extract_cuda_context_kv_lane_snapshot
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
  integer(i32) :: token_value_step_2
  integer(i32) :: token_value_with_other_context
  integer(i32) :: context_byte_count_a
  integer(i32) :: context_byte_count_b
  integer(i32) :: decode_context_byte_count
  integer(i32) :: updated_context_byte_count
  integer(i32) :: stop_reason
  integer(i32) :: status_code
  integer(i32) :: producer_stage
  integer(i32) :: summary_control_a
  integer(i32) :: summary_control_b
  integer      :: shell_status
  integer(i64) :: artifact_hash
  integer(i64) :: token_digest
  integer(i64) :: modal_digest
  integer(i64) :: kv_token_count
  integer(i64) :: decode_step_count
  integer(i64) :: rolling_state_digest
  integer(i64) :: summary_primary_count
  integer(i64) :: summary_secondary_count
  integer(i64) :: prefill_token_digest_a
  integer(i64) :: prefill_modal_digest_a
  integer(i64) :: prefill_rolling_state_a
  integer(i64) :: decode_rolling_state_1
  integer(i64) :: state_image_digest
  integer(i64) :: prefill_state_image_digest_a
  integer(i64) :: prefill_page_digest_a
  integer(i64) :: decode_page_digest_1
  integer(i64) :: page_anchors(4)
  integer(i64) :: page_token_counts(4)
  integer(i64) :: page_lane_digests(4)
  integer(i32) :: page_kinds(4)
  integer(i32) :: page_key_lanes(8, 4)
  integer(i32) :: page_value_lanes(8, 4)
  integer(i32) :: recent_tokens(4)
  integer(i32) :: current_page_index
  integer(i32) :: valid_page_count
  integer(i32) :: recent_token_count
  integer(i32) :: token_values_a(7)
  integer(i32) :: token_values_b(7)
  integer(i8)  :: modal_bytes_a(6)
  integer(i8)  :: modal_bytes_b(6)
  logical      :: snapshot_valid
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
  call expect_equal_i32("cuda prefill should fully populate the fixed context payload", context_byte_count_a, &
    MAX_LIVE_CONTEXT_BYTES)
  call expect_equal_i32("cuda prefill context should start with magic M", int(context_bytes_a(1), kind=i32), iachar("M"))
  call expect_equal_i32("cuda prefill context should start with magic Z", int(context_bytes_a(2), kind=i32), iachar("Z"))
  call expect_equal_i32("cuda prefill context should declare version 1", int(context_bytes_a(5), kind=i32), 1_i32)
  call expect_equal_i32("cuda prefill context should declare prefill kind", int(context_bytes_a(6), kind=i32), 1_i32)
  call extract_cuda_context_state_snapshot(context_bytes_a, context_byte_count_a, producer_stage, artifact_hash, &
    token_digest, modal_digest, kv_token_count, decode_step_count, rolling_state_digest, summary_primary_count, &
    summary_secondary_count, summary_control_a, summary_control_b, snapshot_valid)
  call expect_true("cuda prefill context snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill snapshot should report prefill stage", producer_stage, MIZU_STAGE_PREFILL)
  call expect_true("cuda prefill snapshot should retain artifact lineage", artifact_hash /= 0_i64)
  call expect_equal_i64("cuda prefill snapshot should seed kv tokens from consumed tokens", kv_token_count, 7_i64)
  call expect_equal_i64("cuda prefill snapshot should start decode step count at zero", decode_step_count, 0_i64)
  call expect_equal_i64("cuda prefill summary should report kv tokens", summary_primary_count, 7_i64)
  call expect_equal_i64("cuda prefill summary should report modal byte count", summary_secondary_count, 6_i64)
  call expect_equal_i32("cuda prefill summary should report staged modal count", summary_control_a, 1_i32)
  call expect_equal_i32("cuda prefill summary should clear the trailing control slot", summary_control_b, 0_i32)
  call extract_cuda_context_window_snapshot(context_bytes_a, context_byte_count_a, page_anchors, page_token_counts, &
    page_kinds, current_page_index, valid_page_count, recent_tokens, recent_token_count, state_image_digest, &
    snapshot_valid)
  call expect_true("cuda prefill window snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill window should report one populated kv page", valid_page_count, 1_i32)
  call expect_equal_i32("cuda prefill window should point at the first kv page", current_page_index, 0_i32)
  call expect_equal_i64("cuda prefill window should start the first kv page at token zero", page_anchors(1), 0_i64)
  call expect_equal_i64("cuda prefill window should seed the first kv page with staged tokens", page_token_counts(1), &
    7_i64)
  call expect_equal_i32("cuda prefill window should mark the first kv page as prefill-owned", page_kinds(1), 1_i32)
  call expect_equal_i32("cuda prefill window should retain four recent staged tokens", recent_token_count, 4_i32)
  call expect_equal_i32("cuda prefill window should retain the oldest recent token", recent_tokens(1), 11_i32)
  call expect_equal_i32("cuda prefill window should retain the newest recent token", recent_tokens(4), 19_i32)
  call expect_true("cuda prefill window should retain a nonzero state image digest", state_image_digest /= 0_i64)
  call extract_cuda_context_kv_lane_snapshot(context_bytes_a, context_byte_count_a, page_key_lanes, page_value_lanes, &
    page_lane_digests, snapshot_valid)
  call expect_true("cuda prefill kv lane snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill key lane image should retain the first staged token", page_key_lanes(1, 1), 3_i32)
  call expect_equal_i32("cuda prefill key lane image should retain the seventh staged token", page_key_lanes(7, 1), &
    19_i32)
  call expect_equal_i32("cuda prefill key lane image should leave the trailing slot empty", page_key_lanes(8, 1), 0_i32)
  call expect_true("cuda prefill kv lane image should seed a nonzero value lane", page_value_lanes(1, 1) /= 0_i32)
  call expect_equal_i32("cuda prefill kv lane image should leave the trailing value lane empty", &
    page_value_lanes(8, 1), 0_i32)
  call expect_true("cuda prefill kv lane image should seed a nonzero page digest", page_lane_digests(1) /= 0_i64)
  prefill_token_digest_a = token_digest
  prefill_modal_digest_a = modal_digest
  prefill_rolling_state_a = rolling_state_digest
  prefill_state_image_digest_a = state_image_digest
  prefill_page_digest_a = page_lane_digests(1)
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
  call extract_cuda_context_state_snapshot(context_bytes_b, context_byte_count_b, producer_stage, artifact_hash, &
    token_digest, modal_digest, kv_token_count, decode_step_count, rolling_state_digest, summary_primary_count, &
    summary_secondary_count, summary_control_a, summary_control_b, snapshot_valid)
  call expect_true("cuda prefill snapshot for the second tensor set should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill snapshot for the second tensor set should report prefill stage", &
    producer_stage, MIZU_STAGE_PREFILL)
  call expect_equal_i64("cuda prefill snapshot for the second tensor set should seed kv tokens", kv_token_count, 7_i64)
  call expect_equal_i64("cuda prefill snapshot for the second tensor set should start decode steps at zero", &
    decode_step_count, 0_i64)
  call expect_true("cuda prefill token digest should change when staged tokens change", &
    token_digest /= prefill_token_digest_a)
  call expect_true("cuda prefill modal digest should change when modal bytes change", &
    modal_digest /= prefill_modal_digest_a)
  call expect_true("cuda prefill rolling state should change when staged tensors change", &
    rolling_state_digest /= prefill_rolling_state_a)
  call extract_cuda_context_window_snapshot(context_bytes_b, context_byte_count_b, page_anchors, page_token_counts, &
    page_kinds, current_page_index, valid_page_count, recent_tokens, recent_token_count, state_image_digest, &
    snapshot_valid)
  call expect_true("cuda prefill window snapshot for the second tensor set should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill window for the second tensor set should still report one kv page", &
    valid_page_count, 1_i32)
  call expect_equal_i32("cuda prefill window for the second tensor set should retain four recent staged tokens", &
    recent_token_count, 4_i32)
  call expect_equal_i32("cuda prefill window for the second tensor set should retain the oldest recent token", &
    recent_tokens(1), 8_i32)
  call expect_equal_i32("cuda prefill window for the second tensor set should retain the newest recent token", &
    recent_tokens(4), 14_i32)
  call expect_true("cuda prefill window state digest should change with tensor content", &
    state_image_digest /= prefill_state_image_digest_a)
  call extract_cuda_context_kv_lane_snapshot(context_bytes_b, context_byte_count_b, page_key_lanes, page_value_lanes, &
    page_lane_digests, snapshot_valid)
  call expect_true("cuda prefill kv lane snapshot for the second tensor set should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill key lane image for the second tensor set should retain the first staged token", &
    page_key_lanes(1, 1), 2_i32)
  call expect_equal_i32("cuda prefill key lane image for the second tensor set should retain the seventh staged token", &
    page_key_lanes(7, 1), 14_i32)
  call expect_true("cuda prefill value lane image should change with tensor content", page_value_lanes(1, 1) /= 0_i32)
  call expect_true("cuda prefill page digest should change with tensor content", page_lane_digests(1) /= &
    prefill_page_digest_a)

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
  call expect_equal_i32("cuda decode should fully populate the fixed context payload", updated_context_byte_count, &
    MAX_LIVE_CONTEXT_BYTES)
  call expect_equal_i32("cuda decode context should keep magic M", int(updated_context_bytes(1), kind=i32), iachar("M"))
  call expect_equal_i32("cuda decode context should declare decode kind", int(updated_context_bytes(6), kind=i32), 2_i32)
  call extract_cuda_context_state_snapshot(updated_context_bytes, updated_context_byte_count, producer_stage, &
    artifact_hash, token_digest, modal_digest, kv_token_count, decode_step_count, rolling_state_digest, &
    summary_primary_count, summary_secondary_count, summary_control_a, summary_control_b, snapshot_valid)
  call expect_true("cuda decode context snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode snapshot should report decode stage", producer_stage, MIZU_STAGE_DECODE)
  call expect_equal_i64("cuda decode snapshot should advance kv count from the decode input", kv_token_count, 43_i64)
  call expect_equal_i64("cuda decode snapshot should advance decode steps", decode_step_count, 1_i64)
  call expect_equal_i64("cuda decode summary should report kv tokens after decode", summary_primary_count, 43_i64)
  call expect_equal_i64("cuda decode summary should report decode step count", summary_secondary_count, 1_i64)
  call expect_equal_i32("cuda decode summary should retain the emitted token id", summary_control_a, token_value)
  call expect_equal_i32("cuda decode summary should retain the stop reason", summary_control_b, stop_reason)
  call expect_equal_i64("cuda decode should retain the prefill modal digest", modal_digest, prefill_modal_digest_a)
  call expect_true("cuda decode should advance the token digest beyond prefill state", token_digest /= prefill_token_digest_a)
  call extract_cuda_context_window_snapshot(updated_context_bytes, updated_context_byte_count, page_anchors, &
    page_token_counts, page_kinds, current_page_index, valid_page_count, recent_tokens, recent_token_count, &
    state_image_digest, snapshot_valid)
  call expect_true("cuda decode window snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode window should retain two kv pages after a far jump", valid_page_count, 2_i32)
  call expect_equal_i32("cuda decode window should move the page cursor to the decode-owned page", current_page_index, &
    1_i32)
  call expect_equal_i64("cuda decode window should preserve the prefill kv page", page_token_counts(1), 7_i64)
  call expect_equal_i64("cuda decode window should anchor the decode page at the incoming kv position", &
    page_anchors(2), 42_i64)
  call expect_equal_i64("cuda decode window should seed the decode page with one emitted token", &
    page_token_counts(2), 1_i64)
  call expect_equal_i32("cuda decode window should mark the decode page as decode-owned", page_kinds(2), 2_i32)
  call expect_equal_i32("cuda decode window should keep a full recent-token ring", recent_token_count, 4_i32)
  call expect_equal_i32("cuda decode window should roll forward the recent-token ring", recent_tokens(1), 13_i32)
  call expect_equal_i32("cuda decode window should append the emitted token to the ring", recent_tokens(4), &
    token_value)
  call expect_true("cuda decode window should advance the state image digest", &
    state_image_digest /= prefill_state_image_digest_a)
  call extract_cuda_context_kv_lane_snapshot(updated_context_bytes, updated_context_byte_count, page_key_lanes, &
    page_value_lanes, page_lane_digests, snapshot_valid)
  call expect_true("cuda decode kv lane snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode key lane image should preserve the prefill page payload", page_key_lanes(7, 1), &
    19_i32)
  call expect_equal_i32("cuda decode key lane image should seed the decode page payload with the emitted token", &
    page_key_lanes(1, 2), token_value)
  call expect_equal_i32("cuda decode key lane image should leave the next decode slot empty", page_key_lanes(2, 2), 0_i32)
  call expect_true("cuda decode value lane image should seed a nonzero decode lane", page_value_lanes(1, 2) /= 0_i32)
  call expect_equal_i64("cuda decode should preserve the prefill page digest for the untouched page", &
    page_lane_digests(1), prefill_page_digest_a)
  call expect_true("cuda decode should seed a nonzero digest for the decode-owned page", page_lane_digests(2) /= 0_i64)
  decode_rolling_state_1 = rolling_state_digest
  decode_page_digest_1 = page_lane_digests(2)
  prefill_state_image_digest_a = state_image_digest
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 43_i64, 1_i64, emitted_token_count, token_value_step_2, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, decode_context_bytes, &
    decode_context_byte_count, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("second cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("second cuda decode should emit one token", emitted_token_count, 1_i64)
  call expect_true("second cuda decode should generate a positive token id", token_value_step_2 > 0_i32)
  call extract_cuda_context_state_snapshot(updated_context_bytes, updated_context_byte_count, producer_stage, &
    artifact_hash, token_digest, modal_digest, kv_token_count, decode_step_count, rolling_state_digest, &
    summary_primary_count, summary_secondary_count, summary_control_a, summary_control_b, snapshot_valid)
  call expect_true("second cuda decode context snapshot should be readable", snapshot_valid)
  call expect_equal_i64("second cuda decode snapshot should keep advancing kv count", kv_token_count, 44_i64)
  call expect_equal_i64("second cuda decode snapshot should increment decode steps", decode_step_count, 2_i64)
  call expect_equal_i64("second cuda decode summary should report kv tokens after decode", summary_primary_count, &
    44_i64)
  call expect_equal_i64("second cuda decode summary should report decode step count", summary_secondary_count, 2_i64)
  call expect_equal_i32("second cuda decode summary should retain the emitted token id", summary_control_a, &
    token_value_step_2)
  call expect_true("second cuda decode should advance rolling state", rolling_state_digest /= decode_rolling_state_1)
  call extract_cuda_context_window_snapshot(updated_context_bytes, updated_context_byte_count, page_anchors, &
    page_token_counts, page_kinds, current_page_index, valid_page_count, recent_tokens, recent_token_count, &
    state_image_digest, snapshot_valid)
  call expect_true("second cuda decode window snapshot should be readable", snapshot_valid)
  call expect_equal_i32("second cuda decode window should stay on the same decode page", current_page_index, 1_i32)
  call expect_equal_i64("second cuda decode window should keep the decode page anchor stable", page_anchors(2), 42_i64)
  call expect_equal_i64("second cuda decode window should grow the decode page fill", page_token_counts(2), 2_i64)
  call expect_equal_i32("second cuda decode window should keep a full recent-token ring", recent_token_count, 4_i32)
  call expect_equal_i32("second cuda decode window should keep the earlier emitted token in the ring", &
    recent_tokens(3), token_value)
  call expect_equal_i32("second cuda decode window should append the latest emitted token", recent_tokens(4), &
    token_value_step_2)
  call expect_true("second cuda decode window should advance the state image digest", &
    state_image_digest /= prefill_state_image_digest_a)
  call extract_cuda_context_kv_lane_snapshot(updated_context_bytes, updated_context_byte_count, page_key_lanes, &
    page_value_lanes, page_lane_digests, snapshot_valid)
  call expect_true("second cuda decode kv lane snapshot should be readable", snapshot_valid)
  call expect_equal_i32("second cuda decode key lane image should keep the earlier emitted token", page_key_lanes(1, 2), &
    token_value)
  call expect_equal_i32("second cuda decode key lane image should append the latest emitted token", page_key_lanes(2, 2), &
    token_value_step_2)
  call expect_true("second cuda decode should retain the earlier decode value lane", page_value_lanes(1, 2) /= 0_i32)
  call expect_true("second cuda decode should seed a second decode value lane", page_value_lanes(2, 2) /= 0_i32)
  call expect_true("second cuda decode should advance the decode page digest", page_lane_digests(2) /= &
    decode_page_digest_1)
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
