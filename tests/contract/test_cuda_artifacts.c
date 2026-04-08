#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mizu.h"

static int expect_status(const char *label, mizu_status_code_t actual, mizu_status_code_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected status %d, got %d\n", label, (int)expected, (int)actual);
        return 0;
    }
    return 1;
}

static int expect_true(const char *label, int condition) {
    if (!condition) {
        fprintf(stderr, "%s\n", label);
        return 0;
    }
    return 1;
}

static int expect_i64(const char *label, int64_t actual, int64_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected %lld, got %lld\n", label,
                (long long)expected, (long long)actual);
        return 0;
    }
    return 1;
}

static int expect_u64(const char *label, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected %llu, got %llu\n", label,
                (unsigned long long)expected, (unsigned long long)actual);
        return 0;
    }
    return 1;
}

static int file_contains_substring(const char *path, const char *needle) {
    FILE *file = NULL;
    char line[2048];

    file = fopen(path, "r");
    if (file == NULL) return 0;

    while (fgets(line, sizeof(line), file) != NULL) {
      if (strstr(line, needle) != NULL) {
        fclose(file);
        return 1;
      }
    }

    fclose(file);
    return 0;
}

static int overwrite_file_prefix(const char *path, const uint8_t *bytes, size_t byte_count) {
    FILE *file = fopen(path, "r+b");
    size_t bytes_written;

    if (file == NULL) return 0;
    bytes_written = fwrite(bytes, 1, byte_count, file);
    fclose(file);
    return bytes_written == byte_count;
}

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_runtime_t *runtime_warm = NULL;
    mizu_runtime_t *runtime_fallback = NULL;
    mizu_model_t *model = NULL;
    mizu_model_t *model_warm = NULL;
    mizu_model_t *model_fallback = NULL;
    mizu_session_t *session = NULL;
    mizu_session_t *session_warm = NULL;
    mizu_session_t *session_fallback = NULL;
    mizu_status_code_t status;
    int command_status;
    int32_t tokens[3] = {11, 22, 33};
    uint8_t image_bytes[8] = {9, 8, 7, 6, 5, 4, 3, 2};
    uint8_t mutated_prefix[64];
    int32_t decode_tokens[1] = {0};
    int32_t output_tokens[1] = {0};
    int32_t decode_tokens_warm[1] = {0};
    int32_t output_tokens_warm[1] = {0};
    int32_t decode_tokens_fallback[1] = {0};
    int32_t output_tokens_fallback[1] = {0};
    const char *persist_root = "/tmp/mizu_cuda_artifacts";
    const char *artifact_cache_path = "/tmp/mizu_cuda_artifacts/artifact_cache_v1.txt";
    const char *fixture_source_root = "tests/fixtures/models/fixture_import_bundle_tiny";
    mizu_execution_report_t prefill_reports[2];
    mizu_execution_report_t prefill_reports_warm[2];
    mizu_execution_report_t prefill_reports_fallback[2];
    mizu_execution_report_t decode_reports[1];
    mizu_execution_report_t decode_reports_warm[1];
    mizu_execution_report_t decode_reports_fallback[1];
    mizu_execution_report_t park_reports[1];
    mizu_execution_report_t resume_reports[1];
    mizu_execution_report_t model_report;
    mizu_execution_report_t model_report_warm;
    mizu_execution_report_t model_report_fallback;
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t prefill_buffer_warm;
    mizu_report_buffer_t prefill_buffer_fallback;
    mizu_report_buffer_t decode_buffer;
    mizu_report_buffer_t decode_buffer_warm;
    mizu_report_buffer_t decode_buffer_fallback;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t resume_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_session_info_t session_info;
    mizu_session_info_t session_info_warm;
    mizu_session_info_t session_info_fallback;
    mizu_modal_input_desc_t modal_input;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;
    mizu_decode_result_t decode_result_warm;
    mizu_decode_result_t decode_result_fallback;
    mizu_output_buffer_t output_buffer;
    mizu_output_buffer_t output_buffer_warm;
    mizu_output_buffer_t output_buffer_fallback;
    char command_buffer[2048];
    char fixture_runtime_root[512];
    char fixture_bundle_root[640];
    char mutated_weight_path[768];
    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(prefill_reports_warm, 0, sizeof(prefill_reports_warm));
    memset(prefill_reports_fallback, 0, sizeof(prefill_reports_fallback));
    memset(decode_reports, 0, sizeof(decode_reports));
    memset(decode_reports_warm, 0, sizeof(decode_reports_warm));
    memset(decode_reports_fallback, 0, sizeof(decode_reports_fallback));
    memset(park_reports, 0, sizeof(park_reports));
    memset(resume_reports, 0, sizeof(resume_reports));
    memset(&model_report, 0, sizeof(model_report));
    memset(&model_report_warm, 0, sizeof(model_report_warm));
    memset(&model_report_fallback, 0, sizeof(model_report_fallback));
    memset(&session_info, 0, sizeof(session_info));
    memset(&session_info_warm, 0, sizeof(session_info_warm));
    memset(&session_info_fallback, 0, sizeof(session_info_fallback));

    command_status = system("rm -rf /tmp/mizu_cuda_artifacts && mkdir -p /tmp/mizu_cuda_artifacts");
    if (!expect_true("cuda persist root setup should succeed", command_status == 0)) return 1;

    snprintf(fixture_runtime_root, sizeof(fixture_runtime_root), "%s/runtime_fixture_model", persist_root);
    snprintf(fixture_bundle_root, sizeof(fixture_bundle_root), "%s/mizu_import", fixture_runtime_root);
    snprintf(mutated_weight_path, sizeof(mutated_weight_path), "%s/weights/token_embeddings.bin", fixture_bundle_root);
    snprintf(command_buffer, sizeof(command_buffer), "cp -R %s %s", fixture_source_root, fixture_runtime_root);
    command_status = system(command_buffer);
    if (!expect_true("cuda fixture model copy should succeed", command_status == 0)) return 1;

    if (setenv("MIZU_FORCE_CUDA_AVAILABLE", "1", 1) != 0) {
        fprintf(stderr, "failed to set MIZU_FORCE_CUDA_AVAILABLE\n");
        return 1;
    }

    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.abi_version = mizu_get_abi_version();
    runtime_config.cache_root_z = persist_root;
    runtime_config.optimization_mode = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY;
    runtime_config.exploration_budget = 0;
    runtime_config.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    status = mizu_runtime_create(&runtime_config, &runtime);
    if (!expect_status("cuda runtime create", status, MIZU_STATUS_OK)) return 1;

    model_config.struct_size = sizeof(model_config);
    model_config.abi_version = mizu_get_abi_version();
    model_config.model_root_z = fixture_runtime_root;
    model_config.allowed_backend_mask = MIZU_BACKEND_MASK_CUDA;
    model_config.model_flags = MIZU_MODEL_FLAG_NONE;

    status = mizu_model_open(runtime, &model_config, &model);
    if (!expect_status("cuda model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model, &model_report);
    if (!expect_status("cuda model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda model load should route to CUDA", model_report.execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }

    session_config.struct_size = sizeof(session_config);
    session_config.abi_version = mizu_get_abi_version();
    session_config.max_context_tokens = 4096;
    session_config.max_decode_tokens = 128;
    session_config.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_config.seed = 0;
    session_config.temperature = 0.0f;
    session_config.top_k = 0;
    session_config.top_p = 0.0f;
    session_config.session_flags = MIZU_SESSION_FLAG_NONE;

    status = mizu_session_open(model, &session_config, &session);
    if (!expect_status("cuda session open", status, MIZU_STATUS_OK)) return 1;
    session_info.struct_size = sizeof(session_info);
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after open", status, MIZU_STATUS_OK)) return 1;
    if (!expect_u64("cuda session should start with no flags", session_info.session_state_flags, MIZU_SESSION_STATE_NONE)) {
        return 1;
    }
    if (!expect_i64("cuda session should start with zero kv tokens", session_info.kv_token_count, 0)) return 1;
    status = mizu_session_attach_tokens(session, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("cuda attach tokens", status, MIZU_STATUS_OK)) return 1;

    modal_input.struct_size = sizeof(modal_input);
    modal_input.slot_name_z = "image";
    modal_input.placeholder_ordinal = 1;
    modal_input.modality_kind = MIZU_MODALITY_KIND_IMAGE;
    modal_input.storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES;
    modal_input.dtype = MIZU_DTYPE_U8;
    modal_input.rank = 0;
    modal_input.shape = NULL;
    modal_input.data = image_bytes;
    modal_input.byte_count = sizeof(image_bytes);
    modal_input.lifetime_policy = MIZU_LIFETIME_POLICY_COPY;
    modal_input.input_flags = MIZU_INPUT_FLAG_NONE;

    status = mizu_session_attach_modal_input(session, &modal_input);
    if (!expect_status("cuda attach modal", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after staging", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should report pending inputs after staging",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) != 0)) {
        return 1;
    }
    if (!expect_i64("cuda session should retain staged token count", session_info.staged_token_count, 3)) return 1;
    if (!expect_true("cuda session should retain one staged modal input", session_info.staged_modal_count == 1U)) {
        return 1;
    }

    prefill_buffer.struct_size = sizeof(prefill_buffer);
    prefill_buffer.reports = prefill_reports;
    prefill_buffer.report_capacity = 2;
    prefill_buffer.report_count = 0;

    status = mizu_session_prefill(session, &prefill_buffer);
    if (!expect_status("cuda prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda projector should route to CUDA", prefill_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    if (!expect_true("cuda prefill should route to CUDA", prefill_reports[1].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    if (!expect_true("cuda multimodal prefill should report projector then prefill", 
                     prefill_reports[0].stage_kind == MIZU_STAGE_PROJECTOR &&
                     prefill_reports[1].stage_kind == MIZU_STAGE_PREFILL)) {
        return 1;
    }
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should expose a live context after prefill",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda session should clear pending inputs after prefill",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PENDING_INPUTS) == 0)) {
        return 1;
    }
    if (!expect_i64("cuda session should advance kv count after prefill", session_info.kv_token_count, 3)) return 1;
    if (!expect_i64("cuda session should clear staged token count after prefill", session_info.staged_token_count, 0)) {
        return 1;
    }
    if (!expect_true("cuda session should clear staged modal count after prefill", session_info.staged_modal_count == 0U)) {
        return 1;
    }

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    decode_options.stop_flags = MIZU_STOP_FLAG_NONE;
    decode_options.decode_flags = MIZU_DECODE_FLAG_NONE;

    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = decode_tokens;
    decode_result.token_capacity = 1;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result.result_flags = 0;
    output_buffer.struct_size = sizeof(output_buffer);
    output_buffer.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer.data = output_tokens;
    output_buffer.byte_capacity = sizeof(output_tokens);
    output_buffer.bytes_written = 0;
    output_buffer.output_flags = 0;

    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;
    decode_buffer.report_count = 0;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("cuda decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda decode should route to CUDA", decode_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    if (!expect_true("cuda decode should emit one token", decode_result.token_count == 1U)) return 1;
    if (!expect_true("cuda decode should emit a positive placeholder token", decode_tokens[0] > 0)) {
        return 1;
    }
    status = mizu_session_read_output(session, &output_buffer);
    if (!expect_status("cuda read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda output buffer should report one written token", output_buffer.bytes_written == sizeof(int32_t))) {
        return 1;
    }
    if (!expect_true("cuda read output should match decode result", output_tokens[0] == decode_tokens[0])) return 1;
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_i64("cuda session should advance kv count after decode", session_info.kv_token_count, 4)) return 1;
    if (!expect_true("cuda session should remain live after decode",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;

    status = mizu_session_park(session, &park_buffer);
    if (!expect_status("cuda park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda park should route to CUDA", park_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after park", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should report parked after park",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) != 0)) {
        return 1;
    }
    if (!expect_true("cuda session should retain live-context flag while parked",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("cuda resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda resume should route to CUDA", resume_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }
    status = mizu_session_get_info(session, &session_info);
    if (!expect_status("cuda session info after resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda session should clear parked flag after resume",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_PARKED) == 0)) {
        return 1;
    }
    if (!expect_true("cuda session should still expose live context after resume",
                     (session_info.session_state_flags & MIZU_SESSION_STATE_LIVE_CONTEXT) != 0)) {
        return 1;
    }

    status = mizu_session_close(session);
    if (!expect_status("cuda session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("cuda model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("cuda runtime destroy", status, MIZU_STATUS_OK)) return 1;
    runtime = NULL;
    model = NULL;
    session = NULL;

    if (!expect_true("cuda artifact cache should contain weight format",
                     file_contains_substring(artifact_cache_path, "cuda_bf16_weight_pack_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain projector format",
                     file_contains_substring(artifact_cache_path, "cuda_u8_bf16_projector_plan_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain prefill format",
                     file_contains_substring(artifact_cache_path, "cuda_bf16_prefill_plan_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain decode format",
                     file_contains_substring(artifact_cache_path, "cuda_bf16_decode_plan_v1"))) return 1;
    if (!expect_true("cuda artifact cache should contain session metadata",
                     file_contains_substring(artifact_cache_path, "meta session"))) return 1;

    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights -type f | grep -q .");
    if (!expect_true("cuda weight artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"weights/token_embeddings.bin\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain imported tensor lineage", command_status == 0)) return 1;
    command_status = system("grep -R \"tensor_bytes=2214868992\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain exact imported tensor byte estimates", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_kind=cuda_import_weight_pack_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should materialize an import-driven pack record", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_count=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the expected packed tensor count", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_total_bytes=2205693952\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the expected packed tensor bytes", command_status == 0)) return 1;
    command_status = system("grep -R \"pack1=token_embeddings|embedding_table|weights/token_embeddings.bin|offset=0|bytes=1089994752\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the first packed tensor entry", command_status == 0)) return 1;
    command_status = system("grep -R \"pack4=lm_head|token_projection|weights/lm_head.bin|offset=1115699200|bytes=1089994752\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/weights >/dev/null");
    if (!expect_true("cuda weight artifact should retain the final packed tensor entry", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector -type f | grep -q .");
    if (!expect_true("cuda projector artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"stage=2;.*shape0=8\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain staged modal byte count", command_status == 0)) return 1;
    command_status = system("grep -R \"projector/vision_projector.bin\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain imported projector lineage", command_status == 0)) return 1;
    command_status = system("grep -R \"projector_bytes=9175040\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain exact imported projector byte estimates", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dependency=cuda_import_weight_pack_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should depend on the import-driven weight pack", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_ref_count=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain the packed tensor count dependency", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill -type f | grep -q .");
    if (!expect_true("cuda prefill artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_ref_bytes=2205693952\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the packed tensor byte dependency", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use_kind=cuda_prefill_pack_usage_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain a prefill tensor-usage record", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use_count=3\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the expected prefill tensor-usage count", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use_bytes=1115699200\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the expected prefill tensor-usage bytes", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dispatch_count=3\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the expected compact dispatch count", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dispatch1=offset=0|bytes=1089994752|role=1|layout=1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the first compact dispatch entry", command_status == 0)) return 1;
    snprintf(command_buffer, sizeof(command_buffer),
             "grep -R \"pack_span_root=%s/mizu_import\" %s/artifacts/cuda/cuda/plans/prefill >/dev/null",
             fixture_runtime_root, persist_root);
    command_status = system(command_buffer);
    if (!expect_true("cuda prefill artifact should retain the imported bundle root for tensor spans", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_span1=weights/token_embeddings.bin|sample_bytes=64\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the first imported tensor-span record", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_span_cache=\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain a span-cache sidecar reference", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use1=token_embeddings|embedding_table|offset=0|bytes=1089994752\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill >/dev/null");
    if (!expect_true("cuda prefill artifact should retain the first prefill tensor-usage entry", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode -type f | grep -q .");
    if (!expect_true("cuda decode artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use_kind=cuda_decode_pack_usage_v1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain a decode tensor-usage record", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use_count=4\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain the expected decode tensor-usage count", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_dispatch4=offset=1115699200|bytes=1089994752|role=4|layout=1\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain the final compact dispatch entry", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_span4=weights/lm_head.bin|sample_bytes=64\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain the final imported tensor-span record", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_span_cache=\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain a span-cache sidecar reference", command_status == 0)) return 1;
    command_status = system("grep -R \"pack_use4=lm_head|token_projection|offset=1115699200|bytes=1089994752\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode >/dev/null");
    if (!expect_true("cuda decode artifact should retain the final decode tensor-usage entry", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spancache' | grep -q .");
    if (!expect_true("cuda span-cache sidecar should exist", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spancache' -exec grep -q \"kind=cuda_pack_span_cache_v1\" {} +");
    if (!expect_true("cuda span-cache sidecar should store the expected format marker", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spancache' -exec grep -q \"entry1_sample_hex=\" {} +");
    if (!expect_true("cuda span-cache sidecar should store staged sample bytes", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/sessions -type f | grep -q .");
    if (!expect_true("cuda session artifact file should exist", command_status == 0)) return 1;

    for (size_t byte_index = 0; byte_index < sizeof(mutated_prefix); ++byte_index) {
        mutated_prefix[byte_index] = (uint8_t)(0xF0u - (uint8_t)byte_index);
    }
    if (!expect_true("cuda imported tensor span mutation should succeed",
                     overwrite_file_prefix(mutated_weight_path, mutated_prefix, sizeof(mutated_prefix)))) {
        return 1;
    }

    status = mizu_runtime_create(&runtime_config, &runtime_warm);
    if (!expect_status("cuda warm runtime create", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime_warm, &model_config, &model_warm);
    if (!expect_status("cuda warm model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_warm, &model_report_warm);
    if (!expect_status("cuda warm model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm model load should hit weight cache",
                     (model_report_warm.cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm model load should reuse winner",
                     (model_report_warm.cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }

    status = mizu_session_open(model_warm, &session_config, &session_warm);
    if (!expect_status("cuda warm session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_warm, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("cuda warm attach tokens", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_warm, &modal_input);
    if (!expect_status("cuda warm attach modal", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_warm.struct_size = sizeof(prefill_buffer_warm);
    prefill_buffer_warm.reports = prefill_reports_warm;
    prefill_buffer_warm.report_capacity = 2;
    prefill_buffer_warm.report_count = 0;

    status = mizu_session_prefill(session_warm, &prefill_buffer_warm);
    if (!expect_status("cuda warm prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm projector should hit multimodal cache",
                     (prefill_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_MM_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm projector should reuse winner",
                     (prefill_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm prefill should hit plan cache",
                     (prefill_reports_warm[1].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm prefill should reuse winner",
                     (prefill_reports_warm[1].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }
    status = mizu_session_get_info(session_warm, &session_info_warm);
    if (!expect_status("cuda warm session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_i64("cuda warm session should advance kv count after prefill", session_info_warm.kv_token_count, 3)) {
        return 1;
    }

    decode_result_warm.struct_size = sizeof(decode_result_warm);
    decode_result_warm.token_buffer = decode_tokens_warm;
    decode_result_warm.token_capacity = 1;
    decode_result_warm.token_count = 0;
    decode_result_warm.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result_warm.result_flags = 0;

    output_buffer_warm.struct_size = sizeof(output_buffer_warm);
    output_buffer_warm.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer_warm.data = output_tokens_warm;
    output_buffer_warm.byte_capacity = sizeof(output_tokens_warm);
    output_buffer_warm.bytes_written = 0;
    output_buffer_warm.output_flags = 0;

    decode_buffer_warm.struct_size = sizeof(decode_buffer_warm);
    decode_buffer_warm.reports = decode_reports_warm;
    decode_buffer_warm.report_capacity = 1;
    decode_buffer_warm.report_count = 0;

    status = mizu_session_decode_step(session_warm, &decode_options, &decode_result_warm, &decode_buffer_warm);
    if (!expect_status("cuda warm decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm decode should hit plan cache",
                     (decode_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_PLAN_HIT) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm decode should reuse winner",
                     (decode_reports_warm[0].cache_flags & MIZU_CACHE_FLAG_WINNER_REUSED) != 0)) {
        return 1;
    }
    if (!expect_true("cuda warm decode should reproduce the same token for the same multimodal context",
                     decode_tokens_warm[0] == decode_tokens[0])) {
        return 1;
    }
    status = mizu_session_read_output(session_warm, &output_buffer_warm);
    if (!expect_status("cuda warm read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda warm output should match warm decode token", output_tokens_warm[0] == decode_tokens_warm[0])) {
        return 1;
    }

    status = mizu_session_close(session_warm);
    if (!expect_status("cuda warm session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_warm);
    if (!expect_status("cuda warm model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_warm);
    if (!expect_status("cuda warm runtime destroy", status, MIZU_STATUS_OK)) return 1;
    runtime_warm = NULL;
    model_warm = NULL;
    session_warm = NULL;

    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans -name '*.spancache' -delete");
    if (!expect_true("cuda span-cache sidecar removal should succeed", command_status == 0)) return 1;

    status = mizu_runtime_create(&runtime_config, &runtime_fallback);
    if (!expect_status("cuda fallback runtime create", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_open(runtime_fallback, &model_config, &model_fallback);
    if (!expect_status("cuda fallback model open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_get_last_report(model_fallback, &model_report_fallback);
    if (!expect_status("cuda fallback model report", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda fallback model load should still hit weight cache",
                     (model_report_fallback.cache_flags & MIZU_CACHE_FLAG_WEIGHT_HIT) != 0)) {
        return 1;
    }

    status = mizu_session_open(model_fallback, &session_config, &session_fallback);
    if (!expect_status("cuda fallback session open", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_tokens(session_fallback, tokens, 3, MIZU_ATTACH_FLAG_NONE);
    if (!expect_status("cuda fallback attach tokens", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_attach_modal_input(session_fallback, &modal_input);
    if (!expect_status("cuda fallback attach modal", status, MIZU_STATUS_OK)) return 1;

    prefill_buffer_fallback.struct_size = sizeof(prefill_buffer_fallback);
    prefill_buffer_fallback.reports = prefill_reports_fallback;
    prefill_buffer_fallback.report_capacity = 2;
    prefill_buffer_fallback.report_count = 0;

    status = mizu_session_prefill(session_fallback, &prefill_buffer_fallback);
    if (!expect_status("cuda fallback prefill", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_get_info(session_fallback, &session_info_fallback);
    if (!expect_status("cuda fallback session info after prefill", status, MIZU_STATUS_OK)) return 1;
    if (!expect_i64("cuda fallback session should advance kv count after prefill", session_info_fallback.kv_token_count, 3)) {
        return 1;
    }

    decode_result_fallback.struct_size = sizeof(decode_result_fallback);
    decode_result_fallback.token_buffer = decode_tokens_fallback;
    decode_result_fallback.token_capacity = 1;
    decode_result_fallback.token_count = 0;
    decode_result_fallback.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result_fallback.result_flags = 0;

    output_buffer_fallback.struct_size = sizeof(output_buffer_fallback);
    output_buffer_fallback.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output_buffer_fallback.data = output_tokens_fallback;
    output_buffer_fallback.byte_capacity = sizeof(output_tokens_fallback);
    output_buffer_fallback.bytes_written = 0;
    output_buffer_fallback.output_flags = 0;

    decode_buffer_fallback.struct_size = sizeof(decode_buffer_fallback);
    decode_buffer_fallback.reports = decode_reports_fallback;
    decode_buffer_fallback.report_capacity = 1;
    decode_buffer_fallback.report_count = 0;

    status = mizu_session_decode_step(session_fallback, &decode_options, &decode_result_fallback,
                                      &decode_buffer_fallback);
    if (!expect_status("cuda fallback decode", status, MIZU_STATUS_OK)) return 1;
    status = mizu_session_read_output(session_fallback, &output_buffer_fallback);
    if (!expect_status("cuda fallback read output", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda fallback output should match fallback decode token",
                     output_tokens_fallback[0] == decode_tokens_fallback[0])) {
        return 1;
    }

    park_buffer.struct_size = sizeof(park_buffer);
    park_buffer.reports = park_reports;
    park_buffer.report_capacity = 1;
    park_buffer.report_count = 0;

    status = mizu_session_park(session_fallback, &park_buffer);
    if (!expect_status("cuda fallback park", status, MIZU_STATUS_OK)) return 1;

    status = mizu_session_close(session_fallback);
    if (!expect_status("cuda fallback session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model_fallback);
    if (!expect_status("cuda fallback model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime_fallback);
    if (!expect_status("cuda fallback runtime destroy", status, MIZU_STATUS_OK)) return 1;

    command_status = system("rm -rf /tmp/mizu_cuda_artifacts");
    if (!expect_true("cuda persist root cleanup should succeed", command_status == 0)) return 1;

    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
    puts("test_cuda_artifacts: PASS");
    return 0;
}
