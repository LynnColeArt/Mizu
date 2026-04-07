#include "mizu.h"
#include "cuda_bridge.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define MIZU_CUDA_CONTEXT_MAGIC_0 ((uint8_t)'M')
#define MIZU_CUDA_CONTEXT_MAGIC_1 ((uint8_t)'Z')
#define MIZU_CUDA_CONTEXT_MAGIC_2 ((uint8_t)'C')
#define MIZU_CUDA_CONTEXT_MAGIC_3 ((uint8_t)'T')
#define MIZU_CUDA_CONTEXT_VERSION UINT8_C(1)
#define MIZU_CUDA_CONTEXT_KIND_PREFILL UINT8_C(1)
#define MIZU_CUDA_CONTEXT_KIND_DECODE UINT8_C(2)
#define MIZU_CUDA_CONTEXT_HEADER_SIZE INT32_C(16)
#define MIZU_CUDA_CONTEXT_CHECKSUM_OFFSET UINT32_C(2166136261)
#define MIZU_CUDA_CONTEXT_CHECKSUM_PRIME UINT32_C(16777619)

static uint64_t mix_u64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint32_t compute_context_checksum(const uint8_t *bytes, int32_t stored_count) {
    uint32_t checksum;
    int32_t index;

    checksum = MIZU_CUDA_CONTEXT_CHECKSUM_OFFSET;
    if (bytes == NULL || stored_count <= MIZU_CUDA_CONTEXT_HEADER_SIZE) {
        checksum ^= (uint32_t)(stored_count > 0 ? stored_count : 0);
        checksum *= MIZU_CUDA_CONTEXT_CHECKSUM_PRIME;
        return checksum == 0U ? 1U : checksum;
    }

    for (index = MIZU_CUDA_CONTEXT_HEADER_SIZE; index < stored_count; ++index) {
        checksum ^= (uint32_t)bytes[index];
        checksum *= MIZU_CUDA_CONTEXT_CHECKSUM_PRIME;
    }
    checksum ^= (uint32_t)stored_count;
    checksum *= MIZU_CUDA_CONTEXT_CHECKSUM_PRIME;
    return checksum == 0U ? 1U : checksum;
}

static void copy_name(const char *source_name, char *device_name, size_t device_name_capacity) {
    if (device_name == NULL || device_name_capacity == 0) return;
    if (source_name == NULL) {
        device_name[0] = '\0';
        return;
    }

    strncpy(device_name, source_name, device_name_capacity - 1);
    device_name[device_name_capacity - 1] = '\0';
}

static void stamp_workspace_buffer(void *workspace_buffer,
                                   int64_t workspace_bytes,
                                   uint64_t seed,
                                   uint8_t stage_tag) {
    uint8_t *bytes;
    int64_t index;

    if (workspace_buffer == NULL || workspace_bytes <= 0) return;

    bytes = (uint8_t *)workspace_buffer;
    seed = mix_u64(seed ^ (uint64_t)stage_tag);
    for (index = 0; index < workspace_bytes && index < 32; ++index) {
        bytes[index] = (uint8_t)((seed >> ((index % 8) * 8)) & UINT64_C(0xff));
        seed = mix_u64(seed + (uint64_t)index + (uint64_t)stage_tag);
    }
}

static void fill_prefill_context_bytes(uint64_t seed,
                                       uint64_t artifact_hash,
                                       int64_t token_count,
                                       int64_t modal_byte_count,
                                       int32_t staged_modal_count,
                                       int64_t consumed_token_count,
                                       int8_t *context_bytes,
                                       int32_t context_capacity,
                                       int32_t *context_byte_count) {
    uint8_t *bytes;
    uint32_t checksum;
    uint64_t seed_copy;
    int32_t stored_count;
    int32_t index;

    if (context_byte_count == NULL) return;

    *context_byte_count = 0;
    if (context_bytes == NULL || context_capacity <= 0) return;

    bytes = (uint8_t *)context_bytes;
    stored_count = context_capacity < 64 ? context_capacity : 64;
    memset(bytes, 0, (size_t)stored_count);

    if (stored_count >= 1) bytes[0] = MIZU_CUDA_CONTEXT_MAGIC_0;
    if (stored_count >= 2) bytes[1] = MIZU_CUDA_CONTEXT_MAGIC_1;
    if (stored_count >= 3) bytes[2] = MIZU_CUDA_CONTEXT_MAGIC_2;
    if (stored_count >= 4) bytes[3] = MIZU_CUDA_CONTEXT_MAGIC_3;
    if (stored_count >= 5) bytes[4] = MIZU_CUDA_CONTEXT_VERSION;
    if (stored_count >= 6) bytes[5] = MIZU_CUDA_CONTEXT_KIND_PREFILL;
    if (stored_count >= 8) {
        bytes[6] = (uint8_t)(stored_count & UINT8_C(0xff));
        bytes[7] = (uint8_t)((stored_count >> 8) & UINT8_C(0xff));
    }
    if (stored_count > 16) {
        memcpy(bytes + 16, &token_count, (size_t)(stored_count - 16 < 8 ? stored_count - 16 : 8));
    }
    if (stored_count > 24) {
        memcpy(bytes + 24, &modal_byte_count, (size_t)(stored_count - 24 < 8 ? stored_count - 24 : 8));
    }
    if (stored_count > 32) {
        memcpy(bytes + 32, &staged_modal_count, (size_t)(stored_count - 32 < 4 ? stored_count - 32 : 4));
    }
    if (stored_count > 36) {
        memcpy(bytes + 36, &consumed_token_count, (size_t)(stored_count - 36 < 8 ? stored_count - 36 : 8));
    }
    seed_copy = seed;
    for (index = MIZU_CUDA_CONTEXT_HEADER_SIZE; index < stored_count; ++index) {
        seed_copy = mix_u64(seed_copy ^
                            ((uint64_t)token_count << 1) ^
                            ((uint64_t)modal_byte_count << 9) ^
                            ((uint64_t)(uint32_t)staged_modal_count << 17) ^
                            ((uint64_t)consumed_token_count << 25) ^
                            (uint64_t)(uint32_t)index);
        bytes[index] ^= (uint8_t)(seed_copy & UINT64_C(0xff));
    }

    if (stored_count > 48) {
        memcpy(bytes + 48, &artifact_hash, (size_t)(stored_count - 48 < 8 ? stored_count - 48 : 8));
    }

    checksum = compute_context_checksum(bytes, stored_count);
    if (stored_count > 8) {
        memcpy(bytes + 8, &checksum, (size_t)(stored_count - 8 < 4 ? stored_count - 8 : 4));
    }

    *context_byte_count = stored_count;
}

static void fill_decode_context_bytes(uint64_t seed,
                                      uint64_t artifact_hash,
                                      int64_t kv_after,
                                      int64_t emitted_token_count,
                                      int32_t token_value,
                                      int32_t stop_reason,
                                      int8_t *context_bytes,
                               int32_t context_capacity,
                                      int32_t *context_byte_count) {
  uint8_t *bytes;
    uint32_t checksum;
    uint64_t seed_copy;
    int32_t index;
    int32_t stored_count;

    if (context_byte_count == NULL) return;

    *context_byte_count = 0;
    if (context_bytes == NULL || context_capacity <= 0) return;

    bytes = (uint8_t *)context_bytes;
    stored_count = context_capacity < 64 ? context_capacity : 64;
    memset(bytes, 0, (size_t)stored_count);

    if (stored_count >= 1) bytes[0] = MIZU_CUDA_CONTEXT_MAGIC_0;
    if (stored_count >= 2) bytes[1] = MIZU_CUDA_CONTEXT_MAGIC_1;
    if (stored_count >= 3) bytes[2] = MIZU_CUDA_CONTEXT_MAGIC_2;
    if (stored_count >= 4) bytes[3] = MIZU_CUDA_CONTEXT_MAGIC_3;
    if (stored_count >= 5) bytes[4] = MIZU_CUDA_CONTEXT_VERSION;
    if (stored_count >= 6) bytes[5] = MIZU_CUDA_CONTEXT_KIND_DECODE;
    if (stored_count >= 8) {
        bytes[6] = (uint8_t)(stored_count & UINT8_C(0xff));
        bytes[7] = (uint8_t)((stored_count >> 8) & UINT8_C(0xff));
    }
    if (stored_count > 16) {
        memcpy(bytes + 16, &kv_after, (size_t)(stored_count - 16 < 8 ? stored_count - 16 : 8));
    }
    if (stored_count > 24) {
        memcpy(bytes + 24, &emitted_token_count, (size_t)(stored_count - 24 < 8 ? stored_count - 24 : 8));
    }
    if (stored_count > 32) {
        memcpy(bytes + 32, &token_value, (size_t)(stored_count - 32 < 4 ? stored_count - 32 : 4));
    }
    if (stored_count > 36) {
        memcpy(bytes + 36, &stop_reason, (size_t)(stored_count - 36 < 4 ? stored_count - 36 : 4));
    }
    seed_copy = seed;
    for (index = MIZU_CUDA_CONTEXT_HEADER_SIZE; index < stored_count; ++index) {
        seed_copy = mix_u64(seed_copy ^
                            ((uint64_t)kv_after << 1) ^
                            ((uint64_t)emitted_token_count << 9) ^
                            ((uint64_t)(uint32_t)token_value << 17) ^
                            ((uint64_t)(uint32_t)stop_reason << 25) ^
                            (uint64_t)(uint32_t)index);
        bytes[index] ^= (uint8_t)(seed_copy & UINT64_C(0xff));
    }

    if (stored_count > 48) {
        memcpy(bytes + 48, &artifact_hash, (size_t)(stored_count - 48 < 8 ? stored_count - 48 : 8));
    }

    checksum = compute_context_checksum(bytes, stored_count);
    if (stored_count > 8) {
        memcpy(bytes + 8, &checksum, (size_t)(stored_count - 8 < 4 ? stored_count - 8 : 4));
    }

    *context_byte_count = stored_count;
}

void mizu_cuda_bridge_get_device_info(int32_t *device_count,
                                      int64_t *total_memory_bytes,
                                      int32_t *compute_major,
                                      int32_t *compute_minor,
                                      int32_t *multiprocessor_count,
                                      char *device_name,
                                      size_t device_name_capacity,
                                      int32_t *status_code) {
    if (device_count == NULL || total_memory_bytes == NULL || compute_major == NULL ||
        compute_minor == NULL || multiprocessor_count == NULL || status_code == NULL) {
        return;
    }

    *device_count = 0;
    *total_memory_bytes = 0;
    *compute_major = 0;
    *compute_minor = 0;
    *multiprocessor_count = 0;
    copy_name("cuda_stub", device_name, device_name_capacity);
    *status_code = MIZU_STATUS_OK;
}

void mizu_cuda_bridge_prefill(int64_t payload_hash,
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
    uint64_t seed;
    int64_t index;

    if (consumed_token_count == NULL || status_code == NULL) return;

    *consumed_token_count = token_count > 0 ? token_count : 0;
    if (*consumed_token_count == 0 && staged_modal_count > 0) *consumed_token_count = 1;
    seed = mix_u64((uint64_t)token_count);
    if (token_values != NULL && token_count > 0) {
        for (index = 0; index < token_count; ++index) {
            seed = mix_u64(seed ^ (uint64_t)(uint32_t)token_values[index]);
        }
    }
    if (modal_bytes != NULL && modal_byte_count > 0) {
        for (index = 0; index < modal_byte_count; ++index) {
            seed = mix_u64(seed ^ (uint64_t)(uint8_t)modal_bytes[index]);
        }
    }
    seed = mix_u64((uint64_t)payload_hash ^ seed ^ ((uint64_t)(uint32_t)staged_modal_count << 33));
    fill_prefill_context_bytes(seed, (uint64_t)artifact_hash, token_count, modal_byte_count, staged_modal_count,
                               *consumed_token_count, context_bytes, context_capacity, context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, seed, UINT8_C(3));
    *status_code = MIZU_STATUS_OK;
}

void mizu_cuda_bridge_projector(int64_t payload_hash,
                                int64_t modal_byte_count,
                                int32_t placeholder_count,
                                void *workspace_buffer,
                                int64_t workspace_bytes,
                                int64_t *embedding_count,
                                int32_t *status_code) {
    uint64_t seed;

    if (embedding_count == NULL || status_code == NULL) return;

    seed = mix_u64((uint64_t)payload_hash ^ ((uint64_t)modal_byte_count << 1));
    *embedding_count = (placeholder_count > 0) ? (int64_t)placeholder_count : 1;
    if (modal_byte_count > 0) *embedding_count += (int64_t)(seed % UINT64_C(2));
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, seed, UINT8_C(2));
    *status_code = MIZU_STATUS_OK;
}

void mizu_cuda_bridge_decode(int64_t payload_hash,
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
    uint64_t seed;
    int32_t index;

    if (emitted_token_count == NULL || token_value == NULL || stop_reason == NULL ||
        status_code == NULL) {
        return;
    }

    seed = (uint64_t)payload_hash;
    seed ^= (uint64_t)kv_before * UINT64_C(0x9e3779b97f4a7c15);
    seed ^= (uint64_t)token_budget * UINT64_C(0xbf58476d1ce4e5b9);
    if (context_bytes != NULL && context_byte_count > 0) {
        for (index = 0; index < context_byte_count; ++index) {
            seed = mix_u64(seed ^ (uint64_t)(uint8_t)context_bytes[index]);
        }
    }
    seed = mix_u64(seed);

    *emitted_token_count = token_budget > 0 ? 1 : 0;
    *token_value = 1 + (int32_t)(seed % UINT64_C(4095));
    *stop_reason = MIZU_STOP_REASON_NONE;
    fill_decode_context_bytes(seed ^ (uint64_t)(uint32_t)(*token_value), (uint64_t)artifact_hash,
                              kv_before + *emitted_token_count, *emitted_token_count, *token_value,
                              *stop_reason, updated_context_bytes, updated_context_capacity,
                              updated_context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, seed, UINT8_C(4));
    *status_code = MIZU_STATUS_OK;
}
