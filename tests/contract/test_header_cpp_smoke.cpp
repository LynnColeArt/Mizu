#include "mizu.h"

int main() {
    mizu_model_info_t model_info{};
    mizu_session_info_t session_info{};
    mizu_decode_options_t decode_options{};
    mizu_decode_result_t decode_result{};
    mizu_output_buffer_t output{};
    mizu_report_buffer_t reports{};

    model_info.struct_size = sizeof(model_info);
    model_info.model_family = MIZU_MODEL_FAMILY_QWEN3_5;
    model_info.allowed_backend_mask = MIZU_BACKEND_MASK_CUDA;
    model_info.model_features = MIZU_MODEL_FEATURE_MULTIMODAL |
        MIZU_MODEL_FEATURE_PROJECTOR;

    session_info.struct_size = sizeof(session_info);
    session_info.session_state_flags = MIZU_SESSION_STATE_PENDING_INPUTS;

    decode_options.struct_size = sizeof(decode_options);
    decode_options.token_budget = 1;
    decode_options.stop_flags = MIZU_STOP_FLAG_NONE;
    decode_options.decode_flags = MIZU_DECODE_FLAG_NONE;

    decode_result.struct_size = sizeof(decode_result);
    decode_result.token_buffer = nullptr;
    decode_result.token_capacity = 0;
    decode_result.token_count = 0;
    decode_result.stop_reason = MIZU_STOP_REASON_NONE;
    decode_result.result_flags = 0;

    output.struct_size = sizeof(output);
    output.output_kind = MIZU_OUTPUT_KIND_TOKEN_IDS;
    output.data = nullptr;
    output.byte_capacity = 0;
    output.bytes_written = 0;
    output.output_flags = MIZU_OUTPUT_FLAG_NONE;

    reports.struct_size = sizeof(reports);
    reports.reports = nullptr;
    reports.report_capacity = 0;
    reports.report_count = 0;

    return (int)(model_info.struct_size +
        session_info.struct_size +
        decode_options.struct_size +
        decode_result.struct_size +
        output.struct_size +
        reports.struct_size);
}
