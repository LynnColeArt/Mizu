#include "apple_bridge.h"
#include "apple_bridge_common.h"

#include <string.h>

static void copy_name(const char *source_name, char *device_name, size_t device_name_capacity) {
    if (device_name == NULL || device_name_capacity == 0) return;
    if (source_name == NULL) {
        device_name[0] = '\0';
        return;
    }

    strncpy(device_name, source_name, device_name_capacity - 1);
    device_name[device_name_capacity - 1] = '\0';
}

void mizu_apple_bridge_get_device_info(int32_t *metal_available,
                                       int32_t *ane_available,
                                       char *device_name,
                                       size_t device_name_capacity,
                                       int32_t *status_code) {
    if (metal_available != NULL) *metal_available = 0;
    if (ane_available != NULL) *ane_available = 0;
    copy_name("apple_stub", device_name, device_name_capacity);
    if (status_code != NULL) *status_code = MIZU_STATUS_OK;
}

void mizu_apple_bridge_projector(int32_t execution_route,
                                 int64_t payload_hash,
                                 int64_t modal_byte_count,
                                 int32_t placeholder_count,
                                 void *workspace_buffer,
                                 int64_t workspace_bytes,
                                 int64_t *embedding_count,
                                 int32_t *status_code) {
    mizu_apple_core_projector(execution_route, payload_hash, modal_byte_count, placeholder_count,
                              workspace_buffer, workspace_bytes, embedding_count, status_code);
}

void mizu_apple_bridge_prefill(int32_t execution_route,
                               int64_t payload_hash,
                               int64_t artifact_hash,
                               const int32_t *token_values,
                               int64_t token_count,
                               const int8_t *modal_bytes,
                               int64_t modal_byte_count,
                               int32_t staged_modal_count,
                               void *workspace_buffer,
                               int64_t workspace_bytes,
                               int8_t *context_bytes,
                               int32_t context_capacity,
                               int32_t *context_byte_count,
                               int64_t *consumed_token_count,
                               int32_t *status_code) {
    mizu_apple_core_prefill(execution_route, payload_hash, artifact_hash, token_values, token_count,
                            modal_bytes, modal_byte_count, staged_modal_count, workspace_buffer,
                            workspace_bytes, context_bytes, context_capacity, context_byte_count,
                            consumed_token_count, status_code);
}

void mizu_apple_bridge_decode(int32_t execution_route,
                              int64_t payload_hash,
                              int64_t artifact_hash,
                              int64_t kv_before,
                              int64_t token_budget,
                              const int8_t *context_bytes,
                              int32_t context_byte_count,
                              void *workspace_buffer,
                              int64_t workspace_bytes,
                              int8_t *updated_context_bytes,
                              int32_t updated_context_capacity,
                              int32_t *updated_context_byte_count,
                              int64_t *emitted_token_count,
                              int32_t *token_value,
                              int32_t *stop_reason,
                              int32_t *status_code) {
    mizu_apple_core_decode(execution_route, payload_hash, artifact_hash, kv_before, token_budget,
                           context_bytes, context_byte_count, workspace_buffer, workspace_bytes,
                           updated_context_bytes, updated_context_capacity, updated_context_byte_count,
                           emitted_token_count, token_value, stop_reason, status_code);
}
