#ifndef MIZU_H
#define MIZU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIZU_ABI_VERSION UINT32_C(0x00010000)
#define MIZU_VERSION_MAJOR 0
#define MIZU_VERSION_MINOR 1
#define MIZU_VERSION_PATCH 0

#if defined(_WIN32) && defined(MIZU_SHARED)
#if defined(MIZU_BUILD_SHARED)
#define MIZU_API __declspec(dllexport)
#else
#define MIZU_API __declspec(dllimport)
#endif
#else
#define MIZU_API
#endif

typedef struct mizu_runtime mizu_runtime_t;
typedef struct mizu_model mizu_model_t;
typedef struct mizu_session mizu_session_t;

typedef int32_t mizu_status_code_t;
typedef int32_t mizu_optimization_mode_t;
typedef int32_t mizu_backend_family_t;
typedef int32_t mizu_execution_route_t;
typedef int32_t mizu_model_family_t;
typedef int32_t mizu_stage_kind_t;
typedef int32_t mizu_selection_mode_t;
typedef int32_t mizu_cold_state_t;
typedef int32_t mizu_fallback_reason_t;
typedef int32_t mizu_sampler_kind_t;
typedef int32_t mizu_modality_kind_t;
typedef int32_t mizu_storage_kind_t;
typedef int32_t mizu_dtype_t;
typedef int32_t mizu_lifetime_policy_t;
typedef int32_t mizu_output_kind_t;
typedef int32_t mizu_stop_reason_t;

enum {
    MIZU_STATUS_OK = 0,
    MIZU_STATUS_END_OF_SEQUENCE = 1,

    MIZU_STATUS_INVALID_ARGUMENT = 1000,
    MIZU_STATUS_INVALID_STATE = 1001,
    MIZU_STATUS_BUFFER_TOO_SMALL = 1002,
    MIZU_STATUS_ABI_MISMATCH = 1003,
    MIZU_STATUS_BUSY = 1004,

    MIZU_STATUS_UNSUPPORTED_MODEL = 2000,
    MIZU_STATUS_UNSUPPORTED_MODALITY = 2001,
    MIZU_STATUS_BACKEND_UNAVAILABLE = 2002,
    MIZU_STATUS_NO_VALID_PLAN = 2003,
    MIZU_STATUS_SESSION_EVICTED = 2004,

    MIZU_STATUS_IO_ERROR = 3000,
    MIZU_STATUS_CACHE_ERROR = 3001,
    MIZU_STATUS_EXECUTION_ERROR = 3002,
    MIZU_STATUS_INTERNAL_ERROR = 3003
};

enum {
    MIZU_OPTIMIZATION_MODE_DISABLED = 0,
    MIZU_OPTIMIZATION_MODE_MEASURE_ONLY = 1,
    MIZU_OPTIMIZATION_MODE_LEARN_AND_REUSE = 2
};

enum {
    MIZU_BACKEND_FAMILY_NONE = 0,
    MIZU_BACKEND_FAMILY_APPLE = 1,
    MIZU_BACKEND_FAMILY_CUDA = 2
};

enum {
    MIZU_EXEC_ROUTE_NONE = 0,
    MIZU_EXEC_ROUTE_ANE = 1,
    MIZU_EXEC_ROUTE_METAL = 2,
    MIZU_EXEC_ROUTE_CUDA = 3
};

enum {
    MIZU_MODEL_FAMILY_UNKNOWN = 0,
    MIZU_MODEL_FAMILY_QWEN3_5 = 1,
    MIZU_MODEL_FAMILY_GEMMA4 = 2
};

enum {
    MIZU_STAGE_NONE = 0,
    MIZU_STAGE_MODEL_LOAD = 1,
    MIZU_STAGE_PROJECTOR = 2,
    MIZU_STAGE_PREFILL = 3,
    MIZU_STAGE_DECODE = 4,
    MIZU_STAGE_PARK = 5,
    MIZU_STAGE_RESUME = 6
};

enum {
    MIZU_SELECTION_MODE_NONE = 0,
    MIZU_SELECTION_MODE_DIRECT = 1,
    MIZU_SELECTION_MODE_EXPLORATORY = 2,
    MIZU_SELECTION_MODE_REUSE = 3
};

enum {
    MIZU_COLD_STATE_UNKNOWN = 0,
    MIZU_COLD_STATE_COLD = 1,
    MIZU_COLD_STATE_WARM = 2
};

enum {
    MIZU_FALLBACK_REASON_NONE = 0,
    MIZU_FALLBACK_REASON_UNSUPPORTED_OP = 1,
    MIZU_FALLBACK_REASON_UNSUPPORTED_SHAPE = 2,
    MIZU_FALLBACK_REASON_BACKEND_UNAVAILABLE = 3,
    MIZU_FALLBACK_REASON_ROUTE_DISALLOWED = 4,
    MIZU_FALLBACK_REASON_PLANNER_POLICY = 5
};

enum {
    MIZU_SAMPLER_KIND_NONE = 0,
    MIZU_SAMPLER_KIND_GREEDY = 1,
    MIZU_SAMPLER_KIND_TOP_K_TOP_P = 2
};

enum {
    MIZU_MODALITY_KIND_UNKNOWN = 0,
    MIZU_MODALITY_KIND_IMAGE = 1,
    MIZU_MODALITY_KIND_TENSOR = 2,
    MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS = 3
};

enum {
    MIZU_STORAGE_KIND_UNKNOWN = 0,
    MIZU_STORAGE_KIND_ENCODED_BYTES = 1,
    MIZU_STORAGE_KIND_HOST_TENSOR = 2,
    MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS = 3
};

enum {
    MIZU_DTYPE_UNKNOWN = 0,
    MIZU_DTYPE_U8 = 1,
    MIZU_DTYPE_I32 = 2,
    MIZU_DTYPE_F16 = 3,
    MIZU_DTYPE_BF16 = 4,
    MIZU_DTYPE_F32 = 5
};

enum {
    MIZU_LIFETIME_POLICY_COPY = 0,
    MIZU_LIFETIME_POLICY_BORROW_UNTIL_PREFILL = 1
};

enum {
    MIZU_OUTPUT_KIND_NONE = 0,
    MIZU_OUTPUT_KIND_TOKEN_IDS = 1
};

enum {
    MIZU_STOP_REASON_NONE = 0,
    MIZU_STOP_REASON_EOS = 1,
    MIZU_STOP_REASON_TOKEN_BUDGET = 2,
    MIZU_STOP_REASON_STOP_SEQUENCE = 3,
    MIZU_STOP_REASON_CANCELLED = 4
};

#define MIZU_BACKEND_MASK_NONE UINT64_C(0)
#define MIZU_BACKEND_MASK_APPLE_ANE (UINT64_C(1) << 0)
#define MIZU_BACKEND_MASK_APPLE_METAL (UINT64_C(1) << 1)
#define MIZU_BACKEND_MASK_CUDA (UINT64_C(1) << 2)

#define MIZU_RUNTIME_FLAG_NONE UINT64_C(0)
#define MIZU_MODEL_FLAG_NONE UINT64_C(0)
#define MIZU_SESSION_FLAG_NONE UINT64_C(0)
#define MIZU_ATTACH_FLAG_NONE UINT32_C(0)
#define MIZU_INPUT_FLAG_NONE UINT64_C(0)
#define MIZU_DECODE_FLAG_NONE UINT64_C(0)
#define MIZU_STOP_FLAG_NONE UINT64_C(0)
#define MIZU_OUTPUT_FLAG_NONE UINT64_C(0)

#define MIZU_MODEL_FEATURE_NONE UINT64_C(0)
#define MIZU_MODEL_FEATURE_MULTIMODAL (UINT64_C(1) << 0)
#define MIZU_MODEL_FEATURE_PROJECTOR (UINT64_C(1) << 1)

#define MIZU_SESSION_STATE_NONE UINT64_C(0)
#define MIZU_SESSION_STATE_PENDING_INPUTS (UINT64_C(1) << 0)
#define MIZU_SESSION_STATE_LIVE_CONTEXT (UINT64_C(1) << 1)
#define MIZU_SESSION_STATE_PARKED (UINT64_C(1) << 2)

#define MIZU_CACHE_FLAG_NONE UINT64_C(0)
#define MIZU_CACHE_FLAG_WEIGHT_HIT (UINT64_C(1) << 0)
#define MIZU_CACHE_FLAG_PLAN_HIT (UINT64_C(1) << 1)
#define MIZU_CACHE_FLAG_SESSION_HIT (UINT64_C(1) << 2)
#define MIZU_CACHE_FLAG_MM_HIT (UINT64_C(1) << 3)
#define MIZU_CACHE_FLAG_WINNER_REUSED (UINT64_C(1) << 4)

typedef struct {
    size_t struct_size;
    uint32_t abi_version;
    const char *cache_root_z;
    mizu_optimization_mode_t optimization_mode;
    uint32_t exploration_budget;
    uint64_t runtime_flags;
} mizu_runtime_config_t;

typedef struct {
    size_t struct_size;
    uint32_t abi_version;
    const char *model_root_z;
    uint64_t allowed_backend_mask;
    uint64_t model_flags;
} mizu_model_open_config_t;

typedef struct {
    size_t struct_size;
    uint32_t abi_version;
    int64_t max_context_tokens;
    int64_t max_decode_tokens;
    mizu_sampler_kind_t sampler_kind;
    uint64_t seed;
    float temperature;
    int32_t top_k;
    float top_p;
    uint64_t session_flags;
} mizu_session_config_t;

typedef struct {
    size_t struct_size;
    mizu_model_family_t model_family;
    uint64_t allowed_backend_mask;
    uint64_t model_features;
    uint32_t projector_slot_count;
    uint32_t reserved_u32;
} mizu_model_info_t;

typedef struct {
    size_t struct_size;
    uint64_t session_state_flags;
    int64_t kv_token_count;
    int64_t staged_token_count;
    uint32_t staged_modal_count;
    uint32_t reserved_u32;
} mizu_session_info_t;

typedef struct {
    size_t struct_size;
    const char *slot_name_z;
    uint32_t placeholder_ordinal;
    mizu_modality_kind_t modality_kind;
    mizu_storage_kind_t storage_kind;
    mizu_dtype_t dtype;
    uint32_t rank;
    const int64_t *shape;
    const void *data;
    size_t byte_count;
    mizu_lifetime_policy_t lifetime_policy;
    uint64_t input_flags;
} mizu_modal_input_desc_t;

typedef struct {
    size_t struct_size;
    uint64_t token_budget;
    uint64_t stop_flags;
    uint64_t decode_flags;
} mizu_decode_options_t;

typedef struct {
    size_t struct_size;
    int32_t *token_buffer;
    size_t token_capacity;
    size_t token_count;
    mizu_stop_reason_t stop_reason;
    uint64_t result_flags;
} mizu_decode_result_t;

typedef struct {
    size_t struct_size;
    mizu_output_kind_t output_kind;
    void *data;
    size_t byte_capacity;
    size_t bytes_written;
    uint64_t output_flags;
} mizu_output_buffer_t;

typedef struct {
    size_t struct_size;
    mizu_stage_kind_t stage_kind;
    mizu_backend_family_t backend_family;
    mizu_execution_route_t execution_route;
    uint64_t plan_id;
    mizu_selection_mode_t selection_mode;
    mizu_cold_state_t cold_state;
    mizu_fallback_reason_t fallback_reason;
    uint64_t cache_flags;
    uint64_t elapsed_us;
} mizu_execution_report_t;

typedef struct {
    size_t struct_size;
    mizu_execution_report_t *reports;
    size_t report_capacity;
    size_t report_count;
} mizu_report_buffer_t;

MIZU_API uint32_t mizu_get_abi_version(void);

MIZU_API mizu_status_code_t mizu_runtime_create(
    const mizu_runtime_config_t *config,
    mizu_runtime_t **out_runtime);

MIZU_API mizu_status_code_t mizu_runtime_destroy(
    mizu_runtime_t *runtime);

MIZU_API mizu_status_code_t mizu_runtime_copy_last_error(
    const mizu_runtime_t *runtime,
    char *buffer,
    size_t capacity,
    size_t *out_required);

MIZU_API mizu_status_code_t mizu_model_open(
    mizu_runtime_t *runtime,
    const mizu_model_open_config_t *config,
    mizu_model_t **out_model);

MIZU_API mizu_status_code_t mizu_model_close(
    mizu_model_t *model);

MIZU_API mizu_status_code_t mizu_model_get_info(
    const mizu_model_t *model,
    mizu_model_info_t *out_info);

MIZU_API mizu_status_code_t mizu_model_get_last_report(
    const mizu_model_t *model,
    mizu_execution_report_t *out_report);

MIZU_API mizu_status_code_t mizu_session_open(
    mizu_model_t *model,
    const mizu_session_config_t *config,
    mizu_session_t **out_session);

MIZU_API mizu_status_code_t mizu_session_close(
    mizu_session_t *session);

MIZU_API mizu_status_code_t mizu_session_park(
    mizu_session_t *session,
    mizu_report_buffer_t *out_reports);

MIZU_API mizu_status_code_t mizu_session_resume(
    mizu_session_t *session,
    mizu_report_buffer_t *out_reports);

MIZU_API mizu_status_code_t mizu_session_get_info(
    const mizu_session_t *session,
    mizu_session_info_t *out_info);

MIZU_API mizu_status_code_t mizu_session_attach_tokens(
    mizu_session_t *session,
    const int32_t *tokens,
    size_t token_count,
    uint32_t attach_flags);

MIZU_API mizu_status_code_t mizu_session_attach_modal_input(
    mizu_session_t *session,
    const mizu_modal_input_desc_t *input);

MIZU_API mizu_status_code_t mizu_session_clear_pending_inputs(
    mizu_session_t *session);

MIZU_API mizu_status_code_t mizu_session_prefill(
    mizu_session_t *session,
    mizu_report_buffer_t *out_reports);

MIZU_API mizu_status_code_t mizu_session_decode_step(
    mizu_session_t *session,
    const mizu_decode_options_t *options,
    mizu_decode_result_t *out_result,
    mizu_report_buffer_t *out_reports);

MIZU_API mizu_status_code_t mizu_session_read_output(
    mizu_session_t *session,
    mizu_output_buffer_t *out_output);

MIZU_API mizu_status_code_t mizu_session_get_last_report(
    const mizu_session_t *session,
    mizu_execution_report_t *out_report);

#ifdef __cplusplus
}
#endif

#endif
