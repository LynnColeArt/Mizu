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

int main(void) {
    mizu_runtime_t *runtime = NULL;
    mizu_model_t *model = NULL;
    mizu_session_t *session = NULL;
    mizu_status_code_t status;
    int command_status;
    int32_t tokens[3] = {11, 22, 33};
    uint8_t image_bytes[8] = {9, 8, 7, 6, 5, 4, 3, 2};
    int32_t decode_tokens[1] = {0};
    const char *persist_root = "/tmp/mizu_cuda_artifacts";
    const char *artifact_cache_path = "/tmp/mizu_cuda_artifacts/artifact_cache_v1.txt";
    mizu_execution_report_t prefill_reports[2];
    mizu_execution_report_t decode_reports[1];
    mizu_execution_report_t park_reports[1];
    mizu_execution_report_t resume_reports[1];
    mizu_execution_report_t model_report;
    mizu_report_buffer_t prefill_buffer;
    mizu_report_buffer_t decode_buffer;
    mizu_report_buffer_t park_buffer;
    mizu_report_buffer_t resume_buffer;
    mizu_runtime_config_t runtime_config;
    mizu_model_open_config_t model_config;
    mizu_session_config_t session_config;
    mizu_modal_input_desc_t modal_input;
    mizu_decode_options_t decode_options;
    mizu_decode_result_t decode_result;

    memset(prefill_reports, 0, sizeof(prefill_reports));
    memset(decode_reports, 0, sizeof(decode_reports));
    memset(park_reports, 0, sizeof(park_reports));
    memset(resume_reports, 0, sizeof(resume_reports));
    memset(&model_report, 0, sizeof(model_report));

    command_status = system("rm -rf /tmp/mizu_cuda_artifacts && mkdir -p /tmp/mizu_cuda_artifacts");
    if (!expect_true("cuda persist root setup should succeed", command_status == 0)) return 1;

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
    model_config.model_root_z = "tests/fixtures/models/fixture_mm_tiny";
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

    decode_buffer.struct_size = sizeof(decode_buffer);
    decode_buffer.reports = decode_reports;
    decode_buffer.report_capacity = 1;
    decode_buffer.report_count = 0;

    status = mizu_session_decode_step(session, &decode_options, &decode_result, &decode_buffer);
    if (!expect_status("cuda decode", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda decode should route to CUDA", decode_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
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

    resume_buffer.struct_size = sizeof(resume_buffer);
    resume_buffer.reports = resume_reports;
    resume_buffer.report_capacity = 1;
    resume_buffer.report_count = 0;

    status = mizu_session_resume(session, &resume_buffer);
    if (!expect_status("cuda resume", status, MIZU_STATUS_OK)) return 1;
    if (!expect_true("cuda resume should route to CUDA", resume_reports[0].execution_route == MIZU_EXEC_ROUTE_CUDA)) {
        return 1;
    }

    status = mizu_session_close(session);
    if (!expect_status("cuda session close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_model_close(model);
    if (!expect_status("cuda model close", status, MIZU_STATUS_OK)) return 1;
    status = mizu_runtime_destroy(runtime);
    if (!expect_status("cuda runtime destroy", status, MIZU_STATUS_OK)) return 1;

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
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector -type f | grep -q .");
    if (!expect_true("cuda projector artifact file should exist", command_status == 0)) return 1;
    command_status = system("grep -R \"stage=2;.*shape0=8\" /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/projector >/dev/null");
    if (!expect_true("cuda projector artifact should retain staged modal byte count", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/prefill -type f | grep -q .");
    if (!expect_true("cuda prefill artifact file should exist", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/plans/decode -type f | grep -q .");
    if (!expect_true("cuda decode artifact file should exist", command_status == 0)) return 1;
    command_status = system("find /tmp/mizu_cuda_artifacts/artifacts/cuda/cuda/misc -type f | grep -q .");
    if (!expect_true("cuda session artifact file should exist", command_status == 0)) return 1;

    command_status = system("rm -rf /tmp/mizu_cuda_artifacts");
    if (!expect_true("cuda persist root cleanup should succeed", command_status == 0)) return 1;

    unsetenv("MIZU_FORCE_CUDA_AVAILABLE");
    puts("test_cuda_artifacts: PASS");
    return 0;
}
