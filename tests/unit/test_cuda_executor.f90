program test_cuda_executor
  use iso_c_binding,     only: c_associated, c_f_pointer
  use mod_kinds,         only: c_i8, i8, i32, i64
  use mod_status,        only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_STATE
  use mod_types,         only: MIZU_STOP_REASON_NONE, MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, &
                               workspace_state, MAX_LIVE_CONTEXT_BYTES
  use mod_cuda_bridge,   only: cuda_device_info, query_cuda_device_info
  use mod_cuda_executor, only: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode, &
                               extract_cuda_context_state_snapshot, extract_cuda_context_window_snapshot, &
                               extract_cuda_context_kv_lane_snapshot, extract_cuda_context_kv_layout_snapshot, &
                               extract_cuda_context_page_control_snapshot, &
                               extract_cuda_context_page_tensor_snapshot, &
                               extract_cuda_context_pack_usage_snapshot, &
                               extract_cuda_context_pack_dispatch_snapshot
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
  integer(i8)  :: usage_context_bytes(MAX_LIVE_CONTEXT_BYTES)
  integer(i8)  :: usage_decode_context_bytes(MAX_LIVE_CONTEXT_BYTES)
  integer(i64) :: embedding_count
  integer(i64) :: consumed_token_count
  integer(i64) :: emitted_token_count
  integer(i32) :: token_value
  integer(i32) :: token_value_step_2
  integer(i32) :: token_value_page_3
  integer(i32) :: token_value_page_4
  integer(i32) :: token_value_page_5
  integer(i32) :: token_value_with_other_context
  integer(i32) :: token_value_with_pack_cache
  integer(i32) :: token_value_without_pack_cache
  integer(i32) :: context_byte_count_a
  integer(i32) :: context_byte_count_b
  integer(i32) :: decode_context_byte_count
  integer(i32) :: updated_context_byte_count
  integer(i32) :: usage_context_byte_count
  integer(i32) :: usage_decode_context_byte_count
  integer(i32) :: stop_reason
  integer(i32) :: status_code
  integer(i32) :: bridge_status_code
  integer(i32) :: producer_stage
  integer(i32) :: summary_control_a
  integer(i32) :: summary_control_b
  integer(i32) :: pack_usage_count
  integer(i32) :: pack_dispatch_count
  integer      :: shell_status
  integer(i64) :: artifact_hash
  integer(i64) :: pack_usage_hash
  integer(i64) :: pack_usage_bytes
  integer(i64) :: first_pack_offset
  integer(i64) :: last_pack_offset
  integer(i64) :: last_pack_bytes
  integer(i64) :: pack_dispatch_offsets(4)
  integer(i64) :: pack_dispatch_bytes(4)
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
  integer(i32) :: page_key_rows(4)
  integer(i32) :: page_key_lane_counts(4)
  integer(i32) :: page_value_rows(4)
  integer(i32) :: page_value_lane_counts(4)
  integer(i32) :: page_head_blocks(4)
  integer(i32) :: page_generations(4)
  integer(i32) :: page_owner_kinds(4)
  integer(i32) :: page_usable_capacities(4)
  integer(i32) :: page_committed_tokens(4)
  integer(i32) :: page_free_slots(4)
  integer(i32) :: page_epochs(4)
  integer(i32) :: page_recycle_epochs(4)
  integer(i32) :: page_logical_ids(4)
  integer(i32) :: page_flags(4)
  integer(i32) :: page_key_storage_offsets(4)
  integer(i32) :: page_key_committed_bytes(4)
  integer(i32) :: page_key_capacity_bytes(4)
  integer(i32) :: page_key_row_stride_bytes(4)
  integer(i32) :: page_value_storage_offsets(4)
  integer(i32) :: page_value_committed_bytes(4)
  integer(i32) :: page_value_capacity_bytes(4)
  integer(i32) :: page_value_row_stride_bytes(4)
  integer(i32) :: pack_dispatch_role_codes(4)
  integer(i32) :: pack_dispatch_layout_codes(4)
  integer(i32) :: recent_tokens(4)
  integer(i32) :: current_page_index
  integer(i32) :: valid_page_count
  integer(i32) :: recent_token_count
  integer(i32) :: token_values_a(7)
  integer(i32) :: token_values_b(7)
  integer(i32) :: expected_token_value
  integer(i32) :: expected_token_value_step_2
  integer(i32) :: expected_token_value_page_3
  integer(i32) :: expected_token_value_page_4
  integer(i32) :: expected_token_value_page_5
  integer(i32) :: expected_token_value_with_other_context
  integer(i8)  :: modal_bytes_a(6)
  integer(i8)  :: modal_bytes_b(6)
  logical      :: snapshot_valid
  logical      :: using_stub_bridge
  type(cuda_device_info) :: cuda_info
  integer(i32), parameter :: PAGE_FLAG_RESIDENT = 1_i32
  integer(i32), parameter :: PAGE_FLAG_FULL = 2_i32
  integer(i32), parameter :: PAGE_FLAG_DECODE_OWNED = 4_i32
  integer(i32), parameter :: PAGE_FLAG_RECYCLED = 8_i32
  character(len=*), parameter :: cache_root = "/tmp/mizu_test_cuda_executor"
  character(len=*), parameter :: projector_path = "artifacts/cuda/cuda/projector/test.mm"
  character(len=*), parameter :: prefill_path = "artifacts/cuda/cuda/plans/prefill/test.plan"
  character(len=*), parameter :: decode_path = "artifacts/cuda/cuda/plans/decode/test.plan"
  character(len=*), parameter :: prefill_usage_path = "artifacts/cuda/cuda/plans/prefill/usage.plan"
  character(len=*), parameter :: decode_usage_path = "artifacts/cuda/cuda/plans/decode/usage.plan"
  character(len=*), parameter :: pack_tile_cache_path = "artifacts/cuda/cuda/weights/usage.pack.packtiles"
  character(len=*), parameter :: pack_tile_payload_path = "artifacts/cuda/cuda/weights/usage.pack.packpayload"
  character(len=*), parameter :: pack_tile_buffer_path = "artifacts/cuda/cuda/weights/usage.pack.packbuffer"
  character(len=*), parameter :: import_bundle_root = "tests/fixtures/models/fixture_import_bundle_tiny/mizu_import"

  token_values_a = [3_i32, 5_i32, 7_i32, 11_i32, 13_i32, 17_i32, 19_i32]
  token_values_b = [2_i32, 4_i32, 6_i32, 8_i32, 10_i32, 12_i32, 14_i32]
  modal_bytes_a = [1_i8, 3_i8, 5_i8, 7_i8, 9_i8, 11_i8]
  modal_bytes_b = [2_i8, 4_i8, 6_i8, 8_i8, 10_i8, 12_i8]
  context_bytes_a = 0_i8
  context_bytes_b = 0_i8
  decode_context_bytes = 0_i8
  updated_context_bytes = 0_i8
  usage_context_bytes = 0_i8
  usage_decode_context_bytes = 0_i8
  context_byte_count_a = 0_i32
  context_byte_count_b = 0_i32
  decode_context_byte_count = 0_i32
  updated_context_byte_count = 0_i32
  usage_context_byte_count = 0_i32
  usage_decode_context_byte_count = 0_i32

  call query_cuda_device_info(cuda_info, bridge_status_code)
  call expect_equal_i32("cuda bridge device probe should succeed", bridge_status_code, MIZU_STATUS_OK)
  using_stub_bridge = (trim(cuda_info%device_name) == "cuda_stub")
  expected_token_value = merge(1241_i32, 3479_i32, using_stub_bridge)
  expected_token_value_step_2 = merge(3752_i32, 3781_i32, using_stub_bridge)
  expected_token_value_page_3 = merge(2885_i32, 3107_i32, using_stub_bridge)
  expected_token_value_page_4 = merge(1900_i32, 1557_i32, using_stub_bridge)
  expected_token_value_page_5 = merge(2246_i32, 531_i32, using_stub_bridge)
  expected_token_value_with_other_context = merge(2182_i32, 3022_i32, using_stub_bridge)

  shell_status = 0
  call execute_command_line("rm -rf " // cache_root // " && mkdir -p " // cache_root // &
    "/artifacts/cuda/cuda/projector " // cache_root // "/artifacts/cuda/cuda/plans/prefill " // &
    cache_root // "/artifacts/cuda/cuda/plans/decode " // cache_root // "/artifacts/cuda/cuda/weights", &
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

  open(unit=14, file=trim(cache_root) // "/" // trim(pack_tile_cache_path), status="replace", action="write")
  write(14, "(A)") "kind=cuda_weight_pack_tile_cache_v3;pack_payload=" // pack_tile_payload_path // ";" // &
    "pack_buffer=" // pack_tile_buffer_path // ";" // &
    "pack1_offset=0;pack1_bytes=1089994752;pack1_materialized_hash=9010000000000001;" // &
    "pack1_page_hash=9100000000000001;pack1_page_words=8;pack1_page_data_offset=0;pack1_page_data_bytes=32;" // &
    "pack1_tile_hash=9200000000000001;pack1_tile_bytes=32;pack1_tile_data_offset=32;pack1_tile_data_bytes=32;" // &
    "pack2_offset=1089994752;pack2_bytes=25690112;pack2_materialized_hash=9010000000000002;" // &
    "pack2_page_hash=9100000000000002;pack2_page_words=8;pack2_page_data_offset=64;pack2_page_data_bytes=32;" // &
    "pack2_tile_hash=9200000000000002;pack2_tile_bytes=32;pack2_tile_data_offset=96;pack2_tile_data_bytes=32;" // &
    "pack3_offset=1115684864;pack3_bytes=14336;pack3_materialized_hash=9010000000000003;" // &
    "pack3_page_hash=9100000000000003;pack3_page_words=8;pack3_page_data_offset=128;pack3_page_data_bytes=32;" // &
    "pack3_tile_hash=9200000000000003;pack3_tile_bytes=32;pack3_tile_data_offset=160;pack3_tile_data_bytes=32;" // &
    "pack4_offset=1115699200;pack4_bytes=1089994752;pack4_materialized_hash=9010000000000004;" // &
    "pack4_page_hash=9100000000000004;pack4_page_words=8;pack4_page_data_offset=192;pack4_page_data_bytes=32;" // &
    "pack4_tile_hash=9200000000000004;pack4_tile_bytes=32;pack4_tile_data_offset=224;pack4_tile_data_bytes=32;" // &
    "pack_count=4"
  close(14)

  open(unit=15, file=trim(cache_root) // "/" // trim(pack_tile_payload_path), status="replace", action="write")
  write(15, "(A)") "kind=cuda_weight_pack_payload_v1;" // &
    "pack1_page_hex=00112233445566778899AABBCCDDEEFF0123456789ABCDEFFEDCBA9876543210;" // &
    "pack1_tile_hex=102132435465768798A9BACBDCEDFE0F1E2D3C4B5A69788796A5B4C3D2E1F001;" // &
    "pack2_page_hex=112233445566778899AABBCCDDEEFF00123456789ABCDEFF0FEDCBA987654321;" // &
    "pack2_tile_hex=2132435465768798A9BACBDCEDFE0F102F3E4D5C6B7A8998A7B6C5D4E3F20110;" // &
    "pack3_page_hex=2233445566778899AABBCCDDEEFF001223456789ABCDEFF01FEDCBA987654322;" // &
    "pack3_tile_hex=32435465768798A9BACBDCEDFE0F1021404F5E6D7C8B9AA9B8C7D6E5F4031221;" // &
    "pack4_page_hex=33445566778899AABBCCDDEEFF0011223456789ABCDEFF012FEDCBA987654323;" // &
    "pack4_tile_hex=435465768798A9BACBDCEDFE0F10213251606F7E8D9CABBAC9D8E7F605142332;" // &
    "pack_count=4"
  close(15)
  call write_pack_tile_buffer_fixture(trim(cache_root) // "/" // trim(pack_tile_buffer_path), .false.)

  open(unit=12, file=trim(cache_root) // "/" // trim(prefill_usage_path), status="replace", action="write")
  write(12, "(A)") "candidate=prefill_usage;stage=3;format=cuda_bf16_prefill_plan_v1;" // &
    "pack_use_kind=cuda_prefill_pack_usage_v1;" // &
    "pack_dispatch_kind=cuda_pack_dispatch_v1;" // &
    "pack_ref_tile_cache=" // pack_tile_cache_path // ";" // &
    "pack_span_root=" // import_bundle_root // ";" // &
    "pack_use1=token_embeddings|embedding_table|offset=0|" // &
    "bytes=1089994752|layout=row_major;" // &
    "pack_dispatch1=offset=0|bytes=1089994752|role=1|layout=1;" // &
    "pack_span1=weights/token_embeddings.bin|sample_bytes=64;" // &
    "pack_use2=decoder_blocks|decoder_stack|offset=1089994752|" // &
    "bytes=25690112|layout=packed;" // &
    "pack_dispatch2=offset=1089994752|bytes=25690112|role=2|layout=2;" // &
    "pack_span2=weights/decoder_blocks.bin|sample_bytes=64;" // &
    "pack_use3=final_norm|normalization|offset=1115684864|" // &
    "bytes=14336|layout=vector;" // &
    "pack_dispatch3=offset=1115684864|bytes=14336|role=3|layout=3;" // &
    "pack_span3=weights/final_norm.bin|sample_bytes=64;" // &
    "pack_dispatch_count=3;pack_use_count=3;pack_use_bytes=1115699200;" // &
    "pack_use_first_offset=0;pack_use_last_offset=1115684864;" // &
    "pack_use_last_bytes=14336;" // &
    "pack_use_hash=1111111111111111"
  close(12)

  open(unit=13, file=trim(cache_root) // "/" // trim(decode_usage_path), status="replace", action="write")
  write(13, "(A)") "candidate=decode_usage;stage=4;format=cuda_bf16_decode_plan_v1;" // &
    "pack_use_kind=cuda_decode_pack_usage_v1;" // &
    "pack_dispatch_kind=cuda_pack_dispatch_v1;" // &
    "pack_ref_tile_cache=" // pack_tile_cache_path // ";" // &
    "pack_span_root=" // import_bundle_root // ";" // &
    "pack_use1=token_embeddings|embedding_table|offset=0|" // &
    "bytes=1089994752|layout=row_major;" // &
    "pack_dispatch1=offset=0|bytes=1089994752|role=1|layout=1;" // &
    "pack_span1=weights/token_embeddings.bin|sample_bytes=64;" // &
    "pack_use2=decoder_blocks|decoder_stack|offset=1089994752|" // &
    "bytes=25690112|layout=packed;" // &
    "pack_dispatch2=offset=1089994752|bytes=25690112|role=2|layout=2;" // &
    "pack_span2=weights/decoder_blocks.bin|sample_bytes=64;" // &
    "pack_use3=final_norm|normalization|offset=1115684864|" // &
    "bytes=14336|layout=vector;" // &
    "pack_dispatch3=offset=1115684864|bytes=14336|role=3|layout=3;" // &
    "pack_span3=weights/final_norm.bin|sample_bytes=64;" // &
    "pack_use4=lm_head|token_projection|offset=1115699200|" // &
    "bytes=1089994752|layout=row_major;" // &
    "pack_dispatch4=offset=1115699200|bytes=1089994752|role=4|layout=1;" // &
    "pack_span4=weights/lm_head.bin|sample_bytes=64;" // &
    "pack_dispatch_count=4;pack_use_count=4;pack_use_bytes=2205693952;" // &
    "pack_use_first_offset=0;pack_use_last_offset=1115699200;" // &
    "pack_use_last_bytes=1089994752;" // &
    "pack_use_hash=2222222222222222"
  close(13)

  call initialize_workspace(workspace, 0_i64)
  call reserve_workspace_bytes(workspace, 64_i64, status_code)
  call expect_equal_i32("workspace reservation should succeed", status_code, MIZU_STATUS_OK)
  call expect_true("executor workspace should allocate a host buffer", c_associated(workspace%host_buffer))
  call c_f_pointer(workspace%host_buffer, workspace_view, [int(workspace%bytes_reserved)])

  workspace_view = 0_c_i8
  call execute_cuda_projector(cache_root, projector_path, 8192_i64, 1_i32, 12345_i64, embedding_count, &
    status_code, workspace%host_buffer, workspace%bytes_in_use)
  call expect_equal_i32("cuda projector should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda projector should deterministically emit two embedding slots", embedding_count, 2_i64)
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
  call extract_cuda_context_kv_layout_snapshot(context_bytes_a, context_byte_count_a, page_key_rows, &
    page_key_lane_counts, page_value_rows, page_value_lane_counts, page_head_blocks, page_generations, &
    snapshot_valid)
  call expect_true("cuda prefill kv layout snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill layout should seed the first page key row count", page_key_rows(1), 7_i32)
  call expect_equal_i32("cuda prefill layout should seed the first page value row count", page_value_rows(1), 7_i32)
  call expect_equal_i32("cuda prefill layout should keep a single key lane per row", page_key_lane_counts(1), 1_i32)
  call expect_equal_i32("cuda prefill layout should keep a single value lane per row", page_value_lane_counts(1), 1_i32)
  call expect_equal_i32("cuda prefill layout should seed the first page head block", page_head_blocks(1), 0_i32)
  call expect_equal_i32("cuda prefill layout should start the first page generation at zero", page_generations(1), 0_i32)
  call extract_cuda_context_page_control_snapshot(context_bytes_a, context_byte_count_a, page_owner_kinds, &
    page_usable_capacities, page_committed_tokens, page_free_slots, page_epochs, page_recycle_epochs, &
    page_logical_ids, page_flags, snapshot_valid)
  call expect_true("cuda prefill page control snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill page control should mark the first page as prefill-owned", &
    page_owner_kinds(1), 1_i32)
  call expect_equal_i32("cuda prefill page control should record page capacity", page_usable_capacities(1), 8_i32)
  call expect_equal_i32("cuda prefill page control should record committed tokens", page_committed_tokens(1), 7_i32)
  call expect_equal_i32("cuda prefill page control should record one free slot", page_free_slots(1), 1_i32)
  call expect_equal_i32("cuda prefill page control should seed the first page epoch", page_epochs(1), 1_i32)
  call expect_equal_i32("cuda prefill page control should start recycle epoch at zero", page_recycle_epochs(1), 0_i32)
  call expect_equal_i32("cuda prefill page control should seed the first logical page id", page_logical_ids(1), 1_i32)
  call expect_equal_i32("cuda prefill page control should mark the page as resident", page_flags(1), PAGE_FLAG_RESIDENT)
  call expect_equal_i32("cuda prefill page control should leave the full flag clear", &
    iand(page_flags(1), PAGE_FLAG_FULL), 0_i32)
  call extract_cuda_context_page_tensor_snapshot(context_bytes_a, context_byte_count_a, page_key_storage_offsets, &
    page_key_committed_bytes, page_key_capacity_bytes, page_key_row_stride_bytes, page_value_storage_offsets, &
    page_value_committed_bytes, page_value_capacity_bytes, page_value_row_stride_bytes, snapshot_valid)
  call expect_true("cuda prefill page tensor snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill page tensor should place key rows at the key payload origin", &
    page_key_storage_offsets(1), 128_i32)
  call expect_equal_i32("cuda prefill page tensor should place value rows at the value payload origin", &
    page_value_storage_offsets(1), 256_i32)
  call expect_equal_i32("cuda prefill page tensor should commit twenty-eight key bytes", &
    page_key_committed_bytes(1), 28_i32)
  call expect_equal_i32("cuda prefill page tensor should commit twenty-eight value bytes", &
    page_value_committed_bytes(1), 28_i32)
  call expect_equal_i32("cuda prefill page tensor should reserve thirty-two key bytes", &
    page_key_capacity_bytes(1), 32_i32)
  call expect_equal_i32("cuda prefill page tensor should reserve thirty-two value bytes", &
    page_value_capacity_bytes(1), 32_i32)
  call expect_equal_i32("cuda prefill page tensor should use four-byte key rows", page_key_row_stride_bytes(1), 4_i32)
  call expect_equal_i32("cuda prefill page tensor should use four-byte value rows", &
    page_value_row_stride_bytes(1), 4_i32)
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
  call extract_cuda_context_kv_layout_snapshot(context_bytes_b, context_byte_count_b, page_key_rows, &
    page_key_lane_counts, page_value_rows, page_value_lane_counts, page_head_blocks, page_generations, &
    snapshot_valid)
  call expect_true("cuda prefill kv layout snapshot for the second tensor set should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill layout for the second tensor set should keep the first page key row count", &
    page_key_rows(1), 7_i32)
  call expect_equal_i32("cuda prefill layout for the second tensor set should keep the first page generation at zero", &
    page_generations(1), 0_i32)
  call extract_cuda_context_page_control_snapshot(context_bytes_b, context_byte_count_b, page_owner_kinds, &
    page_usable_capacities, page_committed_tokens, page_free_slots, page_epochs, page_recycle_epochs, &
    page_logical_ids, page_flags, snapshot_valid)
  call expect_true("cuda prefill page control snapshot for the second tensor set should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill page control for the second tensor set should preserve owner kind", &
    page_owner_kinds(1), 1_i32)
  call expect_equal_i32("cuda prefill page control for the second tensor set should preserve logical page id", &
    page_logical_ids(1), 1_i32)
  call extract_cuda_context_page_tensor_snapshot(context_bytes_b, context_byte_count_b, page_key_storage_offsets, &
    page_key_committed_bytes, page_key_capacity_bytes, page_key_row_stride_bytes, page_value_storage_offsets, &
    page_value_committed_bytes, page_value_capacity_bytes, page_value_row_stride_bytes, snapshot_valid)
  call expect_true("cuda prefill page tensor snapshot for the second tensor set should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill page tensor for the second tensor set should preserve the key payload origin", &
    page_key_storage_offsets(1), 128_i32)
  call expect_equal_i32("cuda prefill page tensor for the second tensor set should preserve key capacity bytes", &
    page_key_capacity_bytes(1), 32_i32)

  call execute_cuda_prefill(cache_root, prefill_usage_path, 7_i64, 1_i32, 0_i64, 0_i64, consumed_token_count, &
    status_code, workspace%host_buffer, workspace%bytes_in_use, token_values_a, modal_bytes_a, usage_context_bytes, &
    usage_context_byte_count)
  call expect_equal_i32("cuda prefill with explicit pack usage should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda prefill with explicit pack usage should consume staged tokens", &
    consumed_token_count, 7_i64)
  call extract_cuda_context_pack_usage_snapshot(usage_context_bytes, usage_context_byte_count, pack_usage_hash, &
    pack_usage_bytes, first_pack_offset, last_pack_offset, last_pack_bytes, pack_usage_count, snapshot_valid)
  call expect_true("cuda prefill pack-usage snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill pack-usage snapshot should record three selected tensors", &
    pack_usage_count, 3_i32)
  call expect_equal_i64("cuda prefill pack-usage snapshot should record prefill usage bytes", &
    pack_usage_bytes, 1115699200_i64)
  call expect_equal_i64("cuda prefill pack-usage snapshot should start at the first packed offset", &
    first_pack_offset, 0_i64)
  call expect_equal_i64("cuda prefill pack-usage snapshot should end at the normalization tensor offset", &
    last_pack_offset, 1115684864_i64)
  call expect_equal_i64("cuda prefill pack-usage snapshot should record the normalization tensor bytes", &
    last_pack_bytes, 14336_i64)
  call extract_cuda_context_pack_dispatch_snapshot(usage_context_bytes, usage_context_byte_count, &
    pack_dispatch_offsets, pack_dispatch_bytes, pack_dispatch_role_codes, pack_dispatch_layout_codes, &
    pack_dispatch_count, snapshot_valid)
  call expect_true("cuda prefill pack-dispatch snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should record three live entries", &
    pack_dispatch_count, 3_i32)
  call expect_equal_i64("cuda prefill pack-dispatch snapshot should keep the embedding tensor offset", &
    pack_dispatch_offsets(1), 0_i64)
  call expect_equal_i64("cuda prefill pack-dispatch snapshot should keep the decoder tensor offset", &
    pack_dispatch_offsets(2), 1089994752_i64)
  call expect_equal_i64("cuda prefill pack-dispatch snapshot should keep the normalization tensor offset", &
    pack_dispatch_offsets(3), 1115684864_i64)
  call expect_equal_i64("cuda prefill pack-dispatch snapshot should keep the decoder tensor bytes", &
    pack_dispatch_bytes(2), 25690112_i64)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should label the embedding tensor role", &
    pack_dispatch_role_codes(1), 1_i32)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should label the decoder tensor role", &
    pack_dispatch_role_codes(2), 2_i32)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should label the normalization tensor role", &
    pack_dispatch_role_codes(3), 3_i32)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should label the embedding layout", &
    pack_dispatch_layout_codes(1), 1_i32)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should label the decoder layout", &
    pack_dispatch_layout_codes(2), 2_i32)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should label the normalization layout", &
    pack_dispatch_layout_codes(3), 3_i32)
  call expect_equal_i64("cuda prefill pack-dispatch snapshot should clear the trailing offset slot", &
    pack_dispatch_offsets(4), 0_i64)
  call expect_equal_i32("cuda prefill pack-dispatch snapshot should clear the trailing role slot", &
    pack_dispatch_role_codes(4), 0_i32)

  workspace_view = 0_c_i8
  call execute_cuda_decode(cache_root, decode_path, 42_i64, 1_i64, emitted_token_count, token_value, stop_reason, &
    status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_a, context_byte_count_a, &
    updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("cuda decode should deterministically emit the first build-specific reference token", &
    token_value, expected_token_value)
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
  call extract_cuda_context_kv_layout_snapshot(updated_context_bytes, updated_context_byte_count, page_key_rows, &
    page_key_lane_counts, page_value_rows, page_value_lane_counts, page_head_blocks, page_generations, &
    snapshot_valid)
  call expect_true("cuda decode kv layout snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode layout should preserve the prefill page row count", page_key_rows(1), 7_i32)
  call expect_equal_i32("cuda decode layout should preserve the prefill page generation", page_generations(1), 0_i32)
  call expect_equal_i32("cuda decode layout should seed one key row on the decode page", page_key_rows(2), 1_i32)
  call expect_equal_i32("cuda decode layout should seed one value row on the decode page", page_value_rows(2), 1_i32)
  call expect_equal_i32("cuda decode layout should seed one key lane on the decode page", page_key_lane_counts(2), 1_i32)
  call expect_equal_i32("cuda decode layout should seed one value lane on the decode page", page_value_lane_counts(2), 1_i32)
  call expect_equal_i32("cuda decode layout should derive the decode page head block from kv anchor", &
    page_head_blocks(2), 5_i32)
  call expect_equal_i32("cuda decode layout should start the decode page generation at one", page_generations(2), 1_i32)
  call extract_cuda_context_page_control_snapshot(updated_context_bytes, updated_context_byte_count, page_owner_kinds, &
    page_usable_capacities, page_committed_tokens, page_free_slots, page_epochs, page_recycle_epochs, &
    page_logical_ids, page_flags, snapshot_valid)
  call expect_true("cuda decode page control snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode page control should preserve the prefill owner kind", page_owner_kinds(1), 1_i32)
  call expect_equal_i32("cuda decode page control should mark the new page as decode-owned", page_owner_kinds(2), 2_i32)
  call expect_equal_i32("cuda decode page control should record full page capacity on the decode page", &
    page_usable_capacities(2), 8_i32)
  call expect_equal_i32("cuda decode page control should record one committed token on the decode page", &
    page_committed_tokens(2), 1_i32)
  call expect_equal_i32("cuda decode page control should leave seven free slots on the decode page", &
    page_free_slots(2), 7_i32)
  call expect_equal_i32("cuda decode page control should assign a second page epoch", page_epochs(2), 2_i32)
  call expect_equal_i32("cuda decode page control should keep the decode page recycle epoch cold", &
    page_recycle_epochs(2), 0_i32)
  call expect_equal_i32("cuda decode page control should assign a second logical page id", page_logical_ids(2), 2_i32)
  call expect_equal_i32("cuda decode page control should mark the decode page as resident and decode-owned", &
    page_flags(2), PAGE_FLAG_RESIDENT + PAGE_FLAG_DECODE_OWNED)
  call expect_equal_i32("cuda decode page control should keep the recycle flag clear", &
    iand(page_flags(2), PAGE_FLAG_RECYCLED), 0_i32)
  call extract_cuda_context_page_tensor_snapshot(updated_context_bytes, updated_context_byte_count, &
    page_key_storage_offsets, page_key_committed_bytes, page_key_capacity_bytes, page_key_row_stride_bytes, &
    page_value_storage_offsets, page_value_committed_bytes, page_value_capacity_bytes, &
    page_value_row_stride_bytes, snapshot_valid)
  call expect_true("cuda decode page tensor snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode page tensor should keep the prefill page at the first key offset", &
    page_key_storage_offsets(1), 128_i32)
  call expect_equal_i32("cuda decode page tensor should place the new decode page at the second key offset", &
    page_key_storage_offsets(2), 160_i32)
  call expect_equal_i32("cuda decode page tensor should place the new decode value page at the second value offset", &
    page_value_storage_offsets(2), 288_i32)
  call expect_equal_i32("cuda decode page tensor should commit four key bytes on the decode page", &
    page_key_committed_bytes(2), 4_i32)
  call expect_equal_i32("cuda decode page tensor should commit four value bytes on the decode page", &
    page_value_committed_bytes(2), 4_i32)
  call expect_equal_i32("cuda decode page tensor should reserve thirty-two key bytes on the decode page", &
    page_key_capacity_bytes(2), 32_i32)
  call expect_equal_i32("cuda decode page tensor should keep a four-byte key row stride", &
    page_key_row_stride_bytes(2), 4_i32)
  decode_rolling_state_1 = rolling_state_digest
  decode_page_digest_1 = page_lane_digests(2)
  prefill_state_image_digest_a = state_image_digest
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 43_i64, 1_i64, emitted_token_count, token_value_step_2, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, decode_context_bytes, &
    decode_context_byte_count, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("second cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("second cuda decode should deterministically emit the second build-specific reference token", &
    token_value_step_2, expected_token_value_step_2)
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
  call extract_cuda_context_kv_layout_snapshot(updated_context_bytes, updated_context_byte_count, page_key_rows, &
    page_key_lane_counts, page_value_rows, page_value_lane_counts, page_head_blocks, page_generations, &
    snapshot_valid)
  call expect_true("second cuda decode kv layout snapshot should be readable", snapshot_valid)
  call expect_equal_i32("second cuda decode layout should grow the decode page key row count", page_key_rows(2), 2_i32)
  call expect_equal_i32("second cuda decode layout should grow the decode page value row count", page_value_rows(2), 2_i32)
  call expect_equal_i32("second cuda decode layout should keep the decode page head block stable", &
    page_head_blocks(2), 5_i32)
  call expect_equal_i32("second cuda decode layout should advance the decode page generation", page_generations(2), 2_i32)
  call extract_cuda_context_page_control_snapshot(updated_context_bytes, updated_context_byte_count, page_owner_kinds, &
    page_usable_capacities, page_committed_tokens, page_free_slots, page_epochs, page_recycle_epochs, &
    page_logical_ids, page_flags, snapshot_valid)
  call expect_true("second cuda decode page control snapshot should be readable", snapshot_valid)
  call expect_equal_i32("second cuda decode page control should keep the decode page owner kind", &
    page_owner_kinds(2), 2_i32)
  call expect_equal_i32("second cuda decode page control should grow committed tokens on the decode page", &
    page_committed_tokens(2), 2_i32)
  call expect_equal_i32("second cuda decode page control should shrink free slots on the decode page", &
    page_free_slots(2), 6_i32)
  call expect_equal_i32("second cuda decode page control should keep the decode page epoch stable", &
    page_epochs(2), 2_i32)
  call expect_equal_i32("second cuda decode page control should keep the logical page id stable", &
    page_logical_ids(2), 2_i32)
  call expect_equal_i32("second cuda decode page control should keep the decode flags stable", &
    page_flags(2), PAGE_FLAG_RESIDENT + PAGE_FLAG_DECODE_OWNED)
  call extract_cuda_context_page_tensor_snapshot(updated_context_bytes, updated_context_byte_count, &
    page_key_storage_offsets, page_key_committed_bytes, page_key_capacity_bytes, page_key_row_stride_bytes, &
    page_value_storage_offsets, page_value_committed_bytes, page_value_capacity_bytes, &
    page_value_row_stride_bytes, snapshot_valid)
  call expect_true("second cuda decode page tensor snapshot should be readable", snapshot_valid)
  call expect_equal_i32("second cuda decode page tensor should keep the second-page key offset stable", &
    page_key_storage_offsets(2), 160_i32)
  call expect_equal_i32("second cuda decode page tensor should grow the decode page key bytes", &
    page_key_committed_bytes(2), 8_i32)
  call expect_equal_i32("second cuda decode page tensor should grow the decode page value bytes", &
    page_value_committed_bytes(2), 8_i32)
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 64_i64, 1_i64, emitted_token_count, token_value_page_3, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, decode_context_bytes, &
    decode_context_byte_count, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("third cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("third cuda decode should deterministically emit the third build-specific reference token", &
    token_value_page_3, expected_token_value_page_3)
  call expect_equal_i64("third cuda decode should emit one token", emitted_token_count, 1_i64)
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 96_i64, 1_i64, emitted_token_count, token_value_page_4, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, decode_context_bytes, &
    decode_context_byte_count, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("fourth cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("fourth cuda decode should deterministically emit the fourth build-specific reference token", &
    token_value_page_4, expected_token_value_page_4)
  call expect_equal_i64("fourth cuda decode should emit one token", emitted_token_count, 1_i64)
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 128_i64, 1_i64, emitted_token_count, token_value_page_5, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, decode_context_bytes, &
    decode_context_byte_count, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("fifth cuda decode should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("fifth cuda decode should deterministically emit the fifth build-specific reference token", &
    token_value_page_5, expected_token_value_page_5)
  call expect_equal_i64("fifth cuda decode should emit one token", emitted_token_count, 1_i64)
  call extract_cuda_context_window_snapshot(updated_context_bytes, updated_context_byte_count, page_anchors, &
    page_token_counts, page_kinds, current_page_index, valid_page_count, recent_tokens, recent_token_count, &
    state_image_digest, snapshot_valid)
  call expect_true("overflowed cuda decode window snapshot should be readable", snapshot_valid)
  call expect_equal_i32("overflowed cuda decode window should keep four resident pages", valid_page_count, 4_i32)
  call expect_equal_i32("overflowed cuda decode window should point at the recycled last page", current_page_index, 3_i32)
  call expect_equal_i64("overflowed cuda decode window should drop the original prefill page", page_anchors(1), 42_i64)
  call expect_equal_i64("overflowed cuda decode window should retain the second logical page anchor", page_anchors(2), 64_i64)
  call expect_equal_i64("overflowed cuda decode window should retain the third logical page anchor", page_anchors(3), 96_i64)
  call expect_equal_i64("overflowed cuda decode window should anchor the recycled page at the newest kv jump", &
    page_anchors(4), 128_i64)
  call expect_equal_i32("overflowed cuda decode window should keep the latest four emitted tokens in the ring", &
    recent_tokens(1), token_value_step_2)
  call expect_equal_i32("overflowed cuda decode window should end with the newest emitted token", &
    recent_tokens(4), token_value_page_5)
  call extract_cuda_context_page_control_snapshot(updated_context_bytes, updated_context_byte_count, page_owner_kinds, &
    page_usable_capacities, page_committed_tokens, page_free_slots, page_epochs, page_recycle_epochs, &
    page_logical_ids, page_flags, snapshot_valid)
  call expect_true("overflowed cuda decode page control snapshot should be readable", snapshot_valid)
  call expect_equal_i32("overflowed cuda decode should preserve the second logical page id in the first slot", &
    page_logical_ids(1), 2_i32)
  call expect_equal_i32("overflowed cuda decode should preserve the third logical page id in the second slot", &
    page_logical_ids(2), 3_i32)
  call expect_equal_i32("overflowed cuda decode should preserve the fourth logical page id in the third slot", &
    page_logical_ids(3), 4_i32)
  call expect_equal_i32("overflowed cuda decode should assign a fifth logical page id to the recycled slot", &
    page_logical_ids(4), 5_i32)
  call expect_equal_i32("overflowed cuda decode should preserve the old two-token decode page fill", &
    page_committed_tokens(1), 2_i32)
  call expect_equal_i32("overflowed cuda decode should seed the recycled page with one token", &
    page_committed_tokens(4), 1_i32)
  call expect_equal_i32("overflowed cuda decode should advance the recycled page epoch", page_epochs(4), 5_i32)
  call expect_equal_i32("overflowed cuda decode should mark the recycled physical slot", &
    page_recycle_epochs(4), 1_i32)
  call expect_equal_i32("overflowed cuda decode should keep the recycled page decode-owned", page_owner_kinds(4), 2_i32)
  call expect_equal_i32("overflowed cuda decode should set the recycled page flags", page_flags(4), &
    PAGE_FLAG_RESIDENT + PAGE_FLAG_DECODE_OWNED + PAGE_FLAG_RECYCLED)
  call extract_cuda_context_page_tensor_snapshot(updated_context_bytes, updated_context_byte_count, &
    page_key_storage_offsets, page_key_committed_bytes, page_key_capacity_bytes, page_key_row_stride_bytes, &
    page_value_storage_offsets, page_value_committed_bytes, page_value_capacity_bytes, &
    page_value_row_stride_bytes, snapshot_valid)
  call expect_true("overflowed cuda decode page tensor snapshot should be readable", snapshot_valid)
  call expect_equal_i32("overflowed cuda decode should rotate the oldest decode page into the first key slot", &
    page_key_storage_offsets(1), 128_i32)
  call expect_equal_i32("overflowed cuda decode should keep the recycled page in the last key slot", &
    page_key_storage_offsets(4), 224_i32)
  call expect_equal_i32("overflowed cuda decode should keep the recycled value page in the last value slot", &
    page_value_storage_offsets(4), 352_i32)
  call expect_equal_i32("overflowed cuda decode should retain eight key bytes for the oldest surviving decode page", &
    page_key_committed_bytes(1), 8_i32)
  call expect_equal_i32("overflowed cuda decode should commit four key bytes on the recycled page", &
    page_key_committed_bytes(4), 4_i32)
  call expect_equal_i32("overflowed cuda decode should preserve thirty-two key bytes of capacity on the recycled page", &
    page_key_capacity_bytes(4), 32_i32)
  call expect_equal_i32("overflowed cuda decode should preserve a four-byte row stride on the recycled page", &
    page_key_row_stride_bytes(4), 4_i32)
  decode_context_bytes = updated_context_bytes
  decode_context_byte_count = updated_context_byte_count

  call execute_cuda_decode(cache_root, decode_path, 42_i64, 1_i64, emitted_token_count, token_value_with_other_context, &
    stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, context_bytes_b, &
    context_byte_count_b, updated_context_bytes, updated_context_byte_count)
  call expect_equal_i32("cuda decode with another context should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i32("cuda decode should deterministically reflect alternate context identity", &
    token_value_with_other_context, expected_token_value_with_other_context)
  call expect_true("cuda decode should reflect direct context buffer identity", &
    token_value_with_other_context /= token_value)

  call execute_cuda_decode(cache_root, decode_usage_path, 42_i64, 1_i64, emitted_token_count, &
    token_value_with_other_context, stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, &
    usage_context_bytes, usage_context_byte_count, usage_decode_context_bytes, usage_decode_context_byte_count)
  call expect_equal_i32("cuda decode with explicit pack usage should succeed", status_code, MIZU_STATUS_OK)
  call expect_equal_i64("cuda decode with explicit pack usage should emit one token", emitted_token_count, 1_i64)
  call expect_true("cuda decode with explicit pack usage should emit a positive token", &
    token_value_with_other_context > 0_i32)
  token_value_with_pack_cache = token_value_with_other_context
  call extract_cuda_context_pack_usage_snapshot(usage_decode_context_bytes, usage_decode_context_byte_count, &
    pack_usage_hash, pack_usage_bytes, first_pack_offset, last_pack_offset, last_pack_bytes, pack_usage_count, &
    snapshot_valid)
  call expect_true("cuda decode pack-usage snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode pack-usage snapshot should record four selected tensors", &
    pack_usage_count, 4_i32)
  call expect_equal_i64("cuda decode pack-usage snapshot should record decode usage bytes", &
    pack_usage_bytes, 2205693952_i64)
  call expect_equal_i64("cuda decode pack-usage snapshot should start at the first packed offset", &
    first_pack_offset, 0_i64)
  call expect_equal_i64("cuda decode pack-usage snapshot should end at the token projection tensor offset", &
    last_pack_offset, 1115699200_i64)
  call expect_equal_i64("cuda decode pack-usage snapshot should record the token projection bytes", &
    last_pack_bytes, 1089994752_i64)
  call extract_cuda_context_pack_dispatch_snapshot(usage_decode_context_bytes, usage_decode_context_byte_count, &
    pack_dispatch_offsets, pack_dispatch_bytes, pack_dispatch_role_codes, pack_dispatch_layout_codes, &
    pack_dispatch_count, snapshot_valid)
  call expect_true("cuda decode pack-dispatch snapshot should be readable", snapshot_valid)
  call expect_equal_i32("cuda decode pack-dispatch snapshot should record four live entries", &
    pack_dispatch_count, 4_i32)
  call expect_equal_i64("cuda decode pack-dispatch snapshot should keep the token projection offset", &
    pack_dispatch_offsets(4), 1115699200_i64)
  call expect_equal_i64("cuda decode pack-dispatch snapshot should keep the token projection bytes", &
    pack_dispatch_bytes(4), 1089994752_i64)
  call expect_equal_i32("cuda decode pack-dispatch snapshot should label the token projection role", &
    pack_dispatch_role_codes(4), 4_i32)
  call expect_equal_i32("cuda decode pack-dispatch snapshot should label the token projection layout", &
    pack_dispatch_layout_codes(4), 1_i32)

  call write_pack_tile_buffer_fixture(trim(cache_root) // "/" // trim(pack_tile_buffer_path), .true.)

  call execute_cuda_decode(cache_root, decode_usage_path, 42_i64, 1_i64, emitted_token_count, &
    token_value_without_pack_cache, stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, &
    usage_context_bytes, usage_context_byte_count, usage_decode_context_bytes, usage_decode_context_byte_count)
  call expect_equal_i32("cuda decode with rewritten pack-owned buffer should still succeed", status_code, &
    MIZU_STATUS_OK)
  call expect_true("cuda decode should reflect direct pack-owned buffer bytes", &
    token_value_without_pack_cache /= token_value_with_pack_cache)

  open(unit=13, file=trim(cache_root) // "/" // trim(decode_usage_path), status="replace", action="write")
  write(13, "(A)") "candidate=decode_usage;stage=4;format=cuda_bf16_decode_plan_v1;" // &
    "pack_use_kind=cuda_decode_pack_usage_v1;" // &
    "pack_dispatch_kind=cuda_pack_dispatch_v1;" // &
    "pack_span_root=" // import_bundle_root // ";" // &
    "pack_use1=token_embeddings|embedding_table|offset=0|" // &
    "bytes=1089994752|layout=row_major;" // &
    "pack_dispatch1=offset=0|bytes=1089994752|role=1|layout=1;" // &
    "pack_span1=weights/token_embeddings.bin|sample_bytes=64;" // &
    "pack_use2=decoder_blocks|decoder_stack|offset=1089994752|" // &
    "bytes=25690112|layout=packed;" // &
    "pack_dispatch2=offset=1089994752|bytes=25690112|role=2|layout=2;" // &
    "pack_span2=weights/decoder_blocks.bin|sample_bytes=64;" // &
    "pack_use3=final_norm|normalization|offset=1115684864|" // &
    "bytes=14336|layout=vector;" // &
    "pack_dispatch3=offset=1115684864|bytes=14336|role=3|layout=3;" // &
    "pack_span3=weights/final_norm.bin|sample_bytes=64;" // &
    "pack_use4=lm_head|token_projection|offset=1115699200|" // &
    "bytes=1089994752|layout=row_major;" // &
    "pack_dispatch4=offset=1115699200|bytes=1089994752|role=4|layout=1;" // &
    "pack_span4=weights/lm_head.bin|sample_bytes=64;" // &
    "pack_dispatch_count=4;pack_use_count=4;pack_use_bytes=2205693952;" // &
    "pack_use_first_offset=0;pack_use_last_offset=1115699200;" // &
    "pack_use_last_bytes=1089994752;" // &
    "pack_use_hash=2222222222222222"
  close(13)

  call execute_cuda_decode(cache_root, decode_usage_path, 42_i64, 1_i64, emitted_token_count, &
    token_value_with_other_context, stop_reason, status_code, workspace%host_buffer, workspace%bytes_in_use, &
    usage_context_bytes, usage_context_byte_count, usage_decode_context_bytes, usage_decode_context_byte_count)
  call expect_equal_i32("cuda decode without pack-owned tile cache should still succeed", status_code, MIZU_STATUS_OK)
  call expect_true("cuda decode should reflect pack-owned materialized identity when available", &
    token_value_with_other_context /= token_value_with_pack_cache)

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
      write(*, '(A,": expected ",I0,", got ",I0)') trim(label), expected, actual
      error stop 1
    end if
  end subroutine expect_equal_i32

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, '(A,": expected ",I0,", got ",I0)') trim(label), expected, actual
      error stop 1
    end if
  end subroutine expect_equal_i64

  subroutine write_pack_tile_buffer_fixture(full_path, use_rewritten_bytes)
    character(len=*), intent(in) :: full_path
    logical, intent(in)          :: use_rewritten_bytes
    character(len=64)            :: hex_records(8)
    integer(i8)                  :: buffer_bytes(256)
    integer(i32)                 :: record_index
    integer(i32)                 :: byte_count
    integer(i32)                 :: offset
    integer                      :: unit_id

    if (use_rewritten_bytes) then
      hex_records = [character(len=64) :: &
        "FFEEDDCCBBAA99887766554433221100FEDCBA98765432100123456789ABCDEF", &
        "F0E1D2C3B4A5968778695A4B3C2D1E0F00112233445566778899AABBCCDDEEFF", &
        "EEDDCCBBAA99887766554433221100FEDCBA98765432100123456789ABCDEFF0", &
        "E1D2C3B4A5968778695A4B3C2D1E0F102132435465768798A9BACBDCEDFE0F10", &
        "DDCCBBAA99887766554433221100FEDCBA98765432100123456789ABCDEFF001", &
        "D2C3B4A5968778695A4B3C2D1E0F102132435465768798A9BACBDCEDFE0F1021", &
        "CCBBAA99887766554433221100FEDCBA98765432100123456789ABCDEFF00112", &
        "C3B4A5968778695A4B3C2D1E0F102132435465768798A9BACBDCEDFE0F102132" ]
    else
      hex_records = [character(len=64) :: &
        "00112233445566778899AABBCCDDEEFF0123456789ABCDEFFEDCBA9876543210", &
        "102132435465768798A9BACBDCEDFE0F1E2D3C4B5A69788796A5B4C3D2E1F001", &
        "112233445566778899AABBCCDDEEFF00123456789ABCDEFF0FEDCBA987654321", &
        "2132435465768798A9BACBDCEDFE0F102F3E4D5C6B7A8998A7B6C5D4E3F20110", &
        "2233445566778899AABBCCDDEEFF001223456789ABCDEFF01FEDCBA987654322", &
        "32435465768798A9BACBDCEDFE0F1021404F5E6D7C8B9AA9B8C7D6E5F4031221", &
        "33445566778899AABBCCDDEEFF0011223456789ABCDEFF012FEDCBA987654323", &
        "435465768798A9BACBDCEDFE0F10213251606F7E8D9CABBAC9D8E7F605142332" ]
    end if

    buffer_bytes = 0_i8
    offset = 0_i32
    do record_index = 1_i32, size(hex_records)
      call decode_fixture_hex(hex_records(record_index), buffer_bytes(offset + 1_i32:), byte_count)
      offset = offset + byte_count
    end do

    open(newunit=unit_id, file=trim(full_path), status="replace", access="stream", form="unformatted", action="write")
    write(unit_id) buffer_bytes(1:offset)
    close(unit_id)
  end subroutine write_pack_tile_buffer_fixture

  subroutine decode_fixture_hex(hex_text, byte_values, byte_count)
    character(len=*), intent(in) :: hex_text
    integer(i8), intent(out)     :: byte_values(:)
    integer(i32), intent(out)    :: byte_count
    integer(i32)                 :: hex_len
    integer(i32)                 :: pair_count
    integer(i32)                 :: pair_index
    integer(i32)                 :: byte_value

    byte_values = 0_i8
    hex_len = len_trim(hex_text)
    if (mod(hex_len, 2) /= 0) error stop 1
    pair_count = min(hex_len / 2_i32, int(size(byte_values), kind=i32))
    do pair_index = 1_i32, pair_count
      byte_value = 16_i32 * fixture_hex_digit_value(hex_text(((pair_index - 1_i32) * 2_i32) + 1_i32: &
        ((pair_index - 1_i32) * 2_i32) + 1_i32)) + &
        fixture_hex_digit_value(hex_text(((pair_index - 1_i32) * 2_i32) + 2_i32: &
        ((pair_index - 1_i32) * 2_i32) + 2_i32))
      if (byte_value < 0_i32) error stop 1
      if (byte_value > 127_i32) byte_value = byte_value - 256_i32
      byte_values(pair_index) = int(byte_value, kind=i8)
    end do
    byte_count = pair_count
  end subroutine decode_fixture_hex

  pure integer(i32) function fixture_hex_digit_value(hex_char) result(digit_value)
    character(len=*), intent(in) :: hex_char
    integer(i32)                 :: ascii_code

    digit_value = -1_i32
    if (len_trim(hex_char) <= 0) return

    ascii_code = iachar(hex_char(1:1))
    select case (ascii_code)
    case (iachar("0"):iachar("9"))
      digit_value = ascii_code - iachar("0")
    case (iachar("A"):iachar("F"))
      digit_value = 10_i32 + ascii_code - iachar("A")
    case (iachar("a"):iachar("f"))
      digit_value = 10_i32 + ascii_code - iachar("a")
    case default
      digit_value = -1_i32
    end select
  end function fixture_hex_digit_value

end program test_cuda_executor
