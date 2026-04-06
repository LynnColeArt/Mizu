#include "mizu.h"

int main(void) {
    mizu_runtime_config_t runtime_cfg;
    mizu_model_open_config_t model_cfg;
    mizu_session_config_t session_cfg;
    mizu_execution_report_t report;

    runtime_cfg.struct_size = sizeof(runtime_cfg);
    runtime_cfg.abi_version = mizu_get_abi_version();
    runtime_cfg.cache_root_z = 0;
    runtime_cfg.optimization_mode = MIZU_OPTIMIZATION_MODE_MEASURE_ONLY;
    runtime_cfg.exploration_budget = 0;
    runtime_cfg.runtime_flags = MIZU_RUNTIME_FLAG_NONE;

    model_cfg.struct_size = sizeof(model_cfg);
    model_cfg.abi_version = mizu_get_abi_version();
    model_cfg.model_root_z = 0;
    model_cfg.allowed_backend_mask = MIZU_BACKEND_MASK_APPLE_ANE |
        MIZU_BACKEND_MASK_APPLE_METAL;
    model_cfg.model_flags = MIZU_MODEL_FLAG_NONE;

    session_cfg.struct_size = sizeof(session_cfg);
    session_cfg.abi_version = mizu_get_abi_version();
    session_cfg.max_context_tokens = 0;
    session_cfg.max_decode_tokens = 0;
    session_cfg.sampler_kind = MIZU_SAMPLER_KIND_GREEDY;
    session_cfg.seed = 0;
    session_cfg.temperature = 0.0f;
    session_cfg.top_k = 0;
    session_cfg.top_p = 0.0f;
    session_cfg.session_flags = MIZU_SESSION_FLAG_NONE;

    report.struct_size = sizeof(report);
    report.stage_kind = MIZU_STAGE_NONE;
    report.backend_family = MIZU_BACKEND_FAMILY_NONE;
    report.execution_route = MIZU_EXEC_ROUTE_NONE;
    report.plan_id = 0;
    report.selection_mode = MIZU_SELECTION_MODE_NONE;
    report.cold_state = MIZU_COLD_STATE_UNKNOWN;
    report.fallback_reason = MIZU_FALLBACK_REASON_NONE;
    report.cache_flags = MIZU_CACHE_FLAG_NONE;
    report.elapsed_us = 0;

    return (int)(runtime_cfg.struct_size +
        model_cfg.struct_size +
        session_cfg.struct_size +
        report.struct_size);
}
