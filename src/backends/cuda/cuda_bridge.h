#ifndef MIZU_CUDA_BRIDGE_H
#define MIZU_CUDA_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mizu_cuda_bridge_get_device_info(int32_t *device_count,
                                      int64_t *total_memory_bytes,
                                      int32_t *compute_major,
                                      int32_t *compute_minor,
                                      int32_t *multiprocessor_count,
                                      char *device_name,
                                      size_t device_name_capacity,
                                      int32_t *status_code);

void mizu_cuda_bridge_prefill(int64_t payload_hash,
                              int64_t artifact_hash,
                              int64_t pack_usage_hash,
                              int64_t pack_usage_bytes,
                              int64_t first_pack_offset,
                              int64_t last_pack_offset,
                              int64_t last_pack_bytes,
                              int32_t pack_usage_count,
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
                              int32_t *status_code);

void mizu_cuda_bridge_projector(int64_t payload_hash,
                                int64_t modal_byte_count,
                                int32_t placeholder_count,
                                void *workspace_buffer,
                                int64_t workspace_bytes,
                                int64_t *embedding_count,
                                int32_t *status_code);

void mizu_cuda_bridge_decode(int64_t payload_hash,
                             int64_t artifact_hash,
                             int64_t pack_usage_hash,
                             int64_t pack_usage_bytes,
                             int64_t first_pack_offset,
                             int64_t last_pack_offset,
                             int64_t last_pack_bytes,
                             int32_t pack_usage_count,
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
                             int32_t *status_code);

#ifdef __cplusplus
}
#endif

#endif
