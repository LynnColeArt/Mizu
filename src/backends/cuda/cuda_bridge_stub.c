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
#define MIZU_CUDA_CONTEXT_STATE_LANES 4
#define MIZU_CUDA_CONTEXT_ARTIFACT_OFFSET INT32_C(48)
#define MIZU_CUDA_CONTEXT_SUMMARY_OFFSET INT32_C(56)
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

static uint64_t pack_context_summary(int64_t leading_count,
                                     int64_t auxiliary_count,
                                     int32_t control_a,
                                     int32_t control_b) {
    uint64_t summary;
    uint64_t leading_clamped;
    uint64_t auxiliary_clamped;
    uint64_t control_a_bits;
    uint64_t control_b_bits;

    leading_clamped = (uint64_t)(leading_count < 0 ? 0 : (leading_count > 65535 ? 65535 : leading_count));
    auxiliary_clamped = (uint64_t)(auxiliary_count < 0 ? 0 : (auxiliary_count > 65535 ? 65535 : auxiliary_count));
    control_a_bits = (uint64_t)((uint32_t)control_a & UINT32_C(0xffff));
    control_b_bits = (uint64_t)((uint32_t)control_b & UINT32_C(0xffff));

    summary = leading_clamped;
    summary |= (auxiliary_clamped << 16);
    summary |= (control_a_bits << 32);
    summary |= (control_b_bits << 48);
    return summary;
}

static void write_context_u64(uint8_t *bytes, int32_t stored_count, int32_t offset, uint64_t value) {
    size_t copy_bytes;

    if (bytes == NULL || stored_count <= offset) return;

    copy_bytes = (size_t)(stored_count - offset < 8 ? stored_count - offset : 8);
    memcpy(bytes + offset, &value, copy_bytes);
}

static uint64_t read_context_u64(const int8_t *context_bytes, int32_t context_byte_count, int32_t offset) {
    uint64_t value;
    size_t copy_bytes;

    value = UINT64_C(0);
    if (context_bytes == NULL || context_byte_count <= offset) return value;

    copy_bytes = (size_t)(context_byte_count - offset < 8 ? context_byte_count - offset : 8);
    memcpy(&value, ((const uint8_t *)context_bytes) + offset, copy_bytes);
    return value;
}

static void build_prefill_state_block(uint64_t seed,
                                      uint64_t artifact_hash,
                                      int64_t token_count,
                                      int64_t modal_byte_count,
                                      int32_t staged_modal_count,
                                      int64_t consumed_token_count,
                                      uint64_t state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      uint64_t *summary_word) {
    state_lanes[0] = mix_u64(seed ^ artifact_hash ^ (uint64_t)token_count);
    state_lanes[1] = mix_u64(seed ^ ((uint64_t)modal_byte_count << 1) ^
                             ((uint64_t)(uint32_t)staged_modal_count << 33));
    state_lanes[2] = mix_u64(state_lanes[0] ^ ((uint64_t)consumed_token_count << 1) ^ artifact_hash);
    state_lanes[3] = mix_u64(state_lanes[1] ^ state_lanes[2] ^ UINT64_C(0xC0DA5EED5EED1234));
    if (summary_word != NULL) {
        *summary_word = pack_context_summary(token_count, modal_byte_count, staged_modal_count,
                                             (int32_t)(consumed_token_count > 2147483647LL ? 2147483647LL :
                                                       (consumed_token_count < -2147483647LL - 1LL ?
                                                        -2147483647LL - 1LL : consumed_token_count)));
    }
}

static void build_decode_state_block(const uint64_t current_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                     uint64_t artifact_hash,
                                     int64_t kv_before,
                                     int64_t token_budget,
                                     int64_t emitted_token_count,
                                     int32_t token_value,
                                     int32_t stop_reason,
                                     uint64_t next_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                     uint64_t *summary_word) {
    next_state_lanes[0] = mix_u64(current_state_lanes[0] ^ (uint64_t)kv_before ^ artifact_hash);
    next_state_lanes[1] = mix_u64(current_state_lanes[1] ^ (uint64_t)token_budget ^
                                  ((uint64_t)(uint32_t)token_value << 32));
    next_state_lanes[2] = mix_u64(current_state_lanes[2] ^ (uint64_t)emitted_token_count ^
                                  ((uint64_t)(uint32_t)stop_reason << 40));
    next_state_lanes[3] = mix_u64(current_state_lanes[3] ^ next_state_lanes[0] ^ next_state_lanes[1] ^
                                  next_state_lanes[2] ^ UINT64_C(0x1EAFCAFE5EED4321));
    if (summary_word != NULL) {
        *summary_word = pack_context_summary(kv_before + emitted_token_count, emitted_token_count, token_value,
                                             stop_reason);
    }
}

static void extract_context_state_block(const int8_t *context_bytes,
                                        int32_t context_byte_count,
                                        uint64_t state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                        uint64_t *artifact_hash,
                                        uint64_t *summary_word) {
    int32_t lane_index;

    for (lane_index = 0; lane_index < MIZU_CUDA_CONTEXT_STATE_LANES; ++lane_index) {
        state_lanes[lane_index] = read_context_u64(context_bytes, context_byte_count, 16 + (lane_index * 8));
    }
    if (artifact_hash != NULL) {
        *artifact_hash = read_context_u64(context_bytes, context_byte_count, MIZU_CUDA_CONTEXT_ARTIFACT_OFFSET);
    }
    if (summary_word != NULL) {
        *summary_word = read_context_u64(context_bytes, context_byte_count, MIZU_CUDA_CONTEXT_SUMMARY_OFFSET);
    }
}

static void write_context_state_block(uint8_t *bytes,
                                      int32_t stored_count,
                                      const uint64_t state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      uint64_t artifact_hash,
                                      uint64_t summary_word) {
    int32_t lane_index;

    for (lane_index = 0; lane_index < MIZU_CUDA_CONTEXT_STATE_LANES; ++lane_index) {
        write_context_u64(bytes, stored_count, 16 + (lane_index * 8), state_lanes[lane_index]);
    }
    write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_ARTIFACT_OFFSET, artifact_hash);
    write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_SUMMARY_OFFSET, summary_word);
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
    uint64_t state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES];
    uint64_t summary_word;
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
    if (stored_count >= 6) bytes[5] = MIZU_CUDA_CONTEXT_KIND_PREFILL;
    if (stored_count >= 8) {
        bytes[6] = (uint8_t)(stored_count & UINT8_C(0xff));
        bytes[7] = (uint8_t)((stored_count >> 8) & UINT8_C(0xff));
    }
    build_prefill_state_block(seed, artifact_hash, token_count, modal_byte_count, staged_modal_count,
                              consumed_token_count, state_lanes, &summary_word);
    write_context_state_block(bytes, stored_count, state_lanes, artifact_hash, summary_word);

    checksum = compute_context_checksum(bytes, stored_count);
    if (stored_count > 8) {
        memcpy(bytes + 8, &checksum, (size_t)(stored_count - 8 < 4 ? stored_count - 8 : 4));
    }

    *context_byte_count = stored_count;
}

static void fill_decode_context_bytes(uint64_t seed,
                                      uint64_t artifact_hash,
                                      const uint64_t next_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      uint64_t summary_word,
                                      int8_t *context_bytes,
                               int32_t context_capacity,
                                      int32_t *context_byte_count) {
  uint8_t *bytes;
    uint32_t checksum;
    int32_t stored_count;

    (void)seed;

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
    write_context_state_block(bytes, stored_count, next_state_lanes, artifact_hash, summary_word);

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
    uint64_t state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES];
    uint64_t summary_word;
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
    build_prefill_state_block(seed, (uint64_t)artifact_hash, token_count, modal_byte_count, staged_modal_count,
                              *consumed_token_count, state_lanes, &summary_word);
    fill_prefill_context_bytes(seed, (uint64_t)artifact_hash, token_count, modal_byte_count, staged_modal_count,
                               *consumed_token_count, context_bytes, context_capacity, context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, state_lanes[0] ^ state_lanes[3] ^ summary_word,
                           UINT8_C(3));
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
    uint64_t current_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES];
    uint64_t next_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES];
    uint64_t summary_word;

    if (emitted_token_count == NULL || token_value == NULL || stop_reason == NULL ||
        status_code == NULL) {
        return;
    }

    extract_context_state_block(context_bytes, context_byte_count, current_state_lanes, NULL, &summary_word);
    seed = (uint64_t)payload_hash;
    seed = mix_u64(seed ^ current_state_lanes[0] ^ current_state_lanes[1]);
    seed = mix_u64(seed ^ current_state_lanes[2] ^ current_state_lanes[3] ^ summary_word);
    seed ^= (uint64_t)kv_before * UINT64_C(0x9e3779b97f4a7c15);
    seed ^= (uint64_t)token_budget * UINT64_C(0xbf58476d1ce4e5b9);
    seed = mix_u64(seed);

    *emitted_token_count = token_budget > 0 ? 1 : 0;
    *token_value = 1 + (int32_t)(seed % UINT64_C(4095));
    *stop_reason = MIZU_STOP_REASON_NONE;
    build_decode_state_block(current_state_lanes, (uint64_t)artifact_hash, kv_before, token_budget,
                             *emitted_token_count, *token_value, *stop_reason, next_state_lanes, &summary_word);
    fill_decode_context_bytes(seed ^ (uint64_t)(uint32_t)(*token_value), (uint64_t)artifact_hash,
                              next_state_lanes, summary_word, updated_context_bytes, updated_context_capacity,
                              updated_context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, next_state_lanes[1] ^ next_state_lanes[3] ^
                           summary_word, UINT8_C(4));
    *status_code = MIZU_STATUS_OK;
}
