#ifndef MIZU_APPLE_BRIDGE_COMMON_H
#define MIZU_APPLE_BRIDGE_COMMON_H

#include "mizu.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define MIZU_APPLE_CONTEXT_MAGIC_0 ((uint8_t)'M')
#define MIZU_APPLE_CONTEXT_MAGIC_1 ((uint8_t)'Z')
#define MIZU_APPLE_CONTEXT_MAGIC_2 ((uint8_t)'A')
#define MIZU_APPLE_CONTEXT_MAGIC_3 ((uint8_t)'P')
#define MIZU_APPLE_CONTEXT_VERSION UINT8_C(1)
#define MIZU_APPLE_CONTEXT_KIND_PREFILL UINT8_C(1)
#define MIZU_APPLE_CONTEXT_KIND_DECODE UINT8_C(2)
#define MIZU_APPLE_CONTEXT_HEADER_SIZE INT32_C(16)
#define MIZU_APPLE_CONTEXT_TOTAL_BYTES INT32_C(96)
#define MIZU_APPLE_CONTEXT_CHECKSUM_OFFSET UINT32_C(2166136261)
#define MIZU_APPLE_CONTEXT_CHECKSUM_PRIME UINT32_C(16777619)
#define MIZU_APPLE_CONTEXT_ARTIFACT_OFFSET INT32_C(16)
#define MIZU_APPLE_CONTEXT_TOKEN_DIGEST_OFFSET INT32_C(24)
#define MIZU_APPLE_CONTEXT_MODAL_DIGEST_OFFSET INT32_C(32)
#define MIZU_APPLE_CONTEXT_COUNTER_OFFSET INT32_C(40)
#define MIZU_APPLE_CONTEXT_SUMMARY_OFFSET INT32_C(48)
#define MIZU_APPLE_CONTEXT_STATE_DIGEST_OFFSET INT32_C(56)
#define MIZU_APPLE_CONTEXT_SEED_OFFSET INT32_C(64)

static inline uint64_t mizu_apple_mix_u64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static inline uint64_t mizu_apple_route_seed(int32_t execution_route) {
    switch (execution_route) {
        case MIZU_EXEC_ROUTE_ANE:
            return UINT64_C(0xA4E4A4E4A4E4A4E4);
        case MIZU_EXEC_ROUTE_METAL:
            return UINT64_C(0x4D37414C4D37414C);
        default:
            return UINT64_C(0x4150504C45534545);
    }
}

static inline uint64_t mizu_apple_pack_counters(int64_t kv_token_count, int64_t decode_step_count) {
    uint64_t kv_bits = 0U;
    uint64_t decode_bits = 0U;

    if (kv_token_count > 0) kv_bits = (uint64_t)(kv_token_count > 0xffffffffLL ? 0xffffffffULL : kv_token_count);
    if (decode_step_count > 0) decode_bits = (uint64_t)(decode_step_count > 0xffffffffLL ? 0xffffffffULL : decode_step_count);
    return kv_bits | (decode_bits << 32);
}

static inline uint32_t mizu_apple_compute_context_checksum(const uint8_t *bytes, int32_t stored_count) {
    uint32_t checksum = MIZU_APPLE_CONTEXT_CHECKSUM_OFFSET;
    int32_t index = 0;

    if (bytes == NULL || stored_count <= MIZU_APPLE_CONTEXT_HEADER_SIZE) {
        checksum ^= (uint32_t)(stored_count > 0 ? stored_count : 0);
        checksum *= MIZU_APPLE_CONTEXT_CHECKSUM_PRIME;
        return checksum == 0U ? 1U : checksum;
    }

    for (index = MIZU_APPLE_CONTEXT_HEADER_SIZE; index < stored_count; ++index) {
        checksum ^= (uint32_t)bytes[index];
        checksum *= MIZU_APPLE_CONTEXT_CHECKSUM_PRIME;
    }
    checksum ^= (uint32_t)stored_count;
    checksum *= MIZU_APPLE_CONTEXT_CHECKSUM_PRIME;
    return checksum == 0U ? 1U : checksum;
}

static inline void mizu_apple_write_u32(uint8_t *bytes, int32_t stored_count, int32_t offset, uint32_t value) {
    size_t copy_bytes = 0;

    if (bytes == NULL || stored_count <= offset) return;
    copy_bytes = (size_t)(stored_count - offset < 4 ? stored_count - offset : 4);
    memcpy(bytes + offset, &value, copy_bytes);
}

static inline void mizu_apple_write_u64(uint8_t *bytes, int32_t stored_count, int32_t offset, uint64_t value) {
    size_t copy_bytes = 0;

    if (bytes == NULL || stored_count <= offset) return;
    copy_bytes = (size_t)(stored_count - offset < 8 ? stored_count - offset : 8);
    memcpy(bytes + offset, &value, copy_bytes);
}

static inline uint32_t mizu_apple_read_u32(const int8_t *bytes, int32_t stored_count, int32_t offset) {
    uint32_t value = 0U;
    size_t copy_bytes = 0;

    if (bytes == NULL || stored_count <= offset) return value;
    copy_bytes = (size_t)(stored_count - offset < 4 ? stored_count - offset : 4);
    memcpy(&value, ((const uint8_t *)bytes) + offset, copy_bytes);
    return value;
}

static inline uint64_t mizu_apple_read_u64(const int8_t *bytes, int32_t stored_count, int32_t offset) {
    uint64_t value = 0ULL;
    size_t copy_bytes = 0;

    if (bytes == NULL || stored_count <= offset) return value;
    copy_bytes = (size_t)(stored_count - offset < 8 ? stored_count - offset : 8);
    memcpy(&value, ((const uint8_t *)bytes) + offset, copy_bytes);
    return value;
}

static inline uint64_t mizu_apple_pack_summary(int32_t last_token,
                                               int32_t stop_reason,
                                               int32_t execution_route,
                                               int32_t context_kind) {
    uint64_t summary = 0ULL;

    summary |= (uint64_t)((uint32_t)last_token & 0xffffU);
    summary |= ((uint64_t)((uint32_t)stop_reason & 0xffffU) << 16);
    summary |= ((uint64_t)((uint32_t)execution_route & 0xffffU) << 32);
    summary |= ((uint64_t)((uint32_t)context_kind & 0xffffU) << 48);
    return summary;
}

static inline uint64_t mizu_apple_digest_tokens(const int32_t *token_values, int64_t token_count, uint64_t seed) {
    int64_t index = 0;

    seed = mizu_apple_mix_u64(seed ^ UINT64_C(0x54F4C0DA12345678));
    if (token_values == NULL || token_count <= 0) return seed;
    for (index = 0; index < token_count; ++index) {
        seed = mizu_apple_mix_u64(seed ^ (uint64_t)(uint32_t)token_values[index] ^ ((uint64_t)index << 24));
    }
    return seed;
}

static inline uint64_t mizu_apple_digest_modal(const int8_t *modal_bytes, int64_t modal_byte_count, uint64_t seed) {
    int64_t index = 0;

    seed = mizu_apple_mix_u64(seed ^ UINT64_C(0x4D4F44414C123456));
    if (modal_bytes == NULL || modal_byte_count <= 0) return seed;
    for (index = 0; index < modal_byte_count; ++index) {
        seed = mizu_apple_mix_u64(seed ^ (uint64_t)(uint8_t)modal_bytes[index] ^ ((uint64_t)index << 16));
    }
    return seed;
}

static inline void mizu_apple_stamp_workspace(void *workspace_buffer,
                                              int64_t workspace_bytes,
                                              uint64_t seed,
                                              uint8_t stage_tag) {
    uint8_t *bytes = (uint8_t *)workspace_buffer;
    int64_t index = 0;

    if (bytes == NULL || workspace_bytes <= 0) return;
    seed = mizu_apple_mix_u64(seed ^ (uint64_t)stage_tag);
    for (index = 0; index < workspace_bytes && index < 32; ++index) {
        bytes[index] = (uint8_t)((seed >> ((index % 8) * 8)) & 0xffU);
        seed = mizu_apple_mix_u64(seed + (uint64_t)index + (uint64_t)stage_tag);
    }
}

static inline void mizu_apple_write_context(int32_t execution_route,
                                            uint8_t context_kind,
                                            uint64_t artifact_hash,
                                            uint64_t token_digest,
                                            uint64_t modal_digest,
                                            int64_t kv_token_count,
                                            int64_t decode_step_count,
                                            int32_t last_token,
                                            int32_t stop_reason,
                                            uint64_t state_digest,
                                            uint64_t state_seed,
                                            int8_t *context_bytes,
                                            int32_t context_capacity,
                                            int32_t *context_byte_count) {
    uint8_t *bytes = (uint8_t *)context_bytes;
    int32_t stored_count = 0;
    uint32_t checksum = 0U;

    if (context_byte_count != NULL) *context_byte_count = 0;
    if (context_bytes == NULL) return;
    if (context_capacity < MIZU_APPLE_CONTEXT_TOTAL_BYTES) return;

    stored_count = MIZU_APPLE_CONTEXT_TOTAL_BYTES;
    memset(bytes, 0, (size_t)stored_count);
    bytes[0] = MIZU_APPLE_CONTEXT_MAGIC_0;
    bytes[1] = MIZU_APPLE_CONTEXT_MAGIC_1;
    bytes[2] = MIZU_APPLE_CONTEXT_MAGIC_2;
    bytes[3] = MIZU_APPLE_CONTEXT_MAGIC_3;
    bytes[4] = MIZU_APPLE_CONTEXT_VERSION;
    bytes[5] = context_kind;
    bytes[6] = (uint8_t)((uint32_t)execution_route & 0xffU);
    bytes[7] = 0U;

    mizu_apple_write_u32(bytes, stored_count, 12, (uint32_t)stored_count);
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_ARTIFACT_OFFSET, artifact_hash);
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_TOKEN_DIGEST_OFFSET, token_digest);
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_MODAL_DIGEST_OFFSET, modal_digest);
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_COUNTER_OFFSET,
                         mizu_apple_pack_counters(kv_token_count, decode_step_count));
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_SUMMARY_OFFSET,
                         mizu_apple_pack_summary(last_token, stop_reason, execution_route, context_kind));
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_STATE_DIGEST_OFFSET, state_digest);
    mizu_apple_write_u64(bytes, stored_count, MIZU_APPLE_CONTEXT_SEED_OFFSET, state_seed);

    checksum = mizu_apple_compute_context_checksum(bytes, stored_count);
    mizu_apple_write_u32(bytes, stored_count, 8, checksum);
    if (context_byte_count != NULL) *context_byte_count = stored_count;
}

static inline int mizu_apple_context_is_valid(const int8_t *context_bytes, int32_t context_byte_count) {
    uint32_t stored_count = 0U;
    uint32_t expected_checksum = 0U;
    uint32_t actual_checksum = 0U;

    if (context_bytes == NULL) return 0;
    if (context_byte_count < MIZU_APPLE_CONTEXT_TOTAL_BYTES) return 0;
    if ((uint8_t)context_bytes[0] != MIZU_APPLE_CONTEXT_MAGIC_0) return 0;
    if ((uint8_t)context_bytes[1] != MIZU_APPLE_CONTEXT_MAGIC_1) return 0;
    if ((uint8_t)context_bytes[2] != MIZU_APPLE_CONTEXT_MAGIC_2) return 0;
    if ((uint8_t)context_bytes[3] != MIZU_APPLE_CONTEXT_MAGIC_3) return 0;
    if ((uint8_t)context_bytes[4] != MIZU_APPLE_CONTEXT_VERSION) return 0;
    if ((uint8_t)context_bytes[5] != MIZU_APPLE_CONTEXT_KIND_PREFILL &&
        (uint8_t)context_bytes[5] != MIZU_APPLE_CONTEXT_KIND_DECODE) return 0;

    stored_count = mizu_apple_read_u32(context_bytes, context_byte_count, 12);
    if ((int32_t)stored_count != context_byte_count) return 0;
    expected_checksum = mizu_apple_read_u32(context_bytes, context_byte_count, 8);
    actual_checksum = mizu_apple_compute_context_checksum((const uint8_t *)context_bytes, context_byte_count);
    return expected_checksum == actual_checksum;
}

static inline void mizu_apple_extract_lineage(const int8_t *context_bytes,
                                              int32_t context_byte_count,
                                              int32_t *producer_stage,
                                              int32_t *execution_route,
                                              uint64_t *artifact_hash,
                                              int *lineage_known) {
    if (producer_stage != NULL) *producer_stage = MIZU_STAGE_NONE;
    if (execution_route != NULL) *execution_route = MIZU_EXEC_ROUTE_NONE;
    if (artifact_hash != NULL) *artifact_hash = 0ULL;
    if (lineage_known != NULL) *lineage_known = 0;
    if (!mizu_apple_context_is_valid(context_bytes, context_byte_count)) return;

    if (producer_stage != NULL) {
        if ((uint8_t)context_bytes[5] == MIZU_APPLE_CONTEXT_KIND_PREFILL) {
            *producer_stage = MIZU_STAGE_PREFILL;
        } else if ((uint8_t)context_bytes[5] == MIZU_APPLE_CONTEXT_KIND_DECODE) {
            *producer_stage = MIZU_STAGE_DECODE;
        }
    }
    if (execution_route != NULL) *execution_route = (int32_t)((uint8_t)context_bytes[6]);
    if (artifact_hash != NULL) *artifact_hash = mizu_apple_read_u64(context_bytes, context_byte_count, MIZU_APPLE_CONTEXT_ARTIFACT_OFFSET);
    if (lineage_known != NULL) *lineage_known = 1;
}

static inline void mizu_apple_core_projector(int32_t execution_route,
                                             int64_t payload_hash,
                                             int64_t modal_byte_count,
                                             int32_t placeholder_count,
                                             void *workspace_buffer,
                                             int64_t workspace_bytes,
                                             int64_t *embedding_count,
                                             int32_t *status_code) {
    uint64_t seed = 0ULL;
    int64_t resolved_count = 0;

    if (embedding_count == NULL || status_code == NULL) return;

    seed = mizu_apple_mix_u64((uint64_t)payload_hash ^ mizu_apple_route_seed(execution_route) ^
                              ((uint64_t)(modal_byte_count > 0 ? modal_byte_count : 0) << 1));
    resolved_count = placeholder_count > 0 ? (int64_t)placeholder_count : 1LL;
    if (modal_byte_count > 0) resolved_count += (int64_t)(seed % 2ULL);
    if (execution_route == MIZU_EXEC_ROUTE_METAL) resolved_count += 1LL;
    *embedding_count = resolved_count;
    mizu_apple_stamp_workspace(workspace_buffer, workspace_bytes, seed ^ (uint64_t)resolved_count, (uint8_t)(0x20 + execution_route));
    *status_code = MIZU_STATUS_OK;
}

static inline void mizu_apple_core_prefill(int32_t execution_route,
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
    uint64_t seed = 0ULL;
    uint64_t token_digest = 0ULL;
    uint64_t modal_digest = 0ULL;
    uint64_t state_digest = 0ULL;
    int64_t resolved_count = 0;
    int32_t last_token = 0;

    if (consumed_token_count == NULL || status_code == NULL || context_byte_count == NULL) return;

    *consumed_token_count = 0;
    *context_byte_count = 0;
    if (context_bytes == NULL || context_capacity < MIZU_APPLE_CONTEXT_TOTAL_BYTES) {
        *status_code = MIZU_STATUS_INVALID_ARGUMENT;
        return;
    }

    resolved_count = token_count > 0 ? token_count : 0;
    if (resolved_count == 0 && staged_modal_count > 0) resolved_count = 1;
    seed = mizu_apple_mix_u64((uint64_t)payload_hash ^ (uint64_t)artifact_hash ^ mizu_apple_route_seed(execution_route));
    token_digest = mizu_apple_digest_tokens(token_values, token_count, seed);
    modal_digest = mizu_apple_digest_modal(modal_bytes, modal_byte_count, seed ^ UINT64_C(0x1111222233334444));
    if (token_values != NULL && token_count > 0) last_token = token_values[token_count - 1];
    state_digest = mizu_apple_mix_u64(seed ^ token_digest ^ modal_digest ^
                                      mizu_apple_pack_counters(resolved_count, 0) ^
                                      (uint64_t)(uint32_t)staged_modal_count);

    mizu_apple_write_context(execution_route, MIZU_APPLE_CONTEXT_KIND_PREFILL, (uint64_t)artifact_hash,
                             token_digest, modal_digest, resolved_count, 0, last_token,
                             MIZU_STOP_REASON_NONE, state_digest, seed, context_bytes,
                             context_capacity, context_byte_count);
    *consumed_token_count = resolved_count;
    mizu_apple_stamp_workspace(workspace_buffer, workspace_bytes, state_digest, (uint8_t)(0x30 + execution_route));
    *status_code = (*context_byte_count > 0) ? MIZU_STATUS_OK : MIZU_STATUS_INVALID_ARGUMENT;
}

static inline void mizu_apple_core_decode(int32_t execution_route,
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
    uint64_t current_artifact_hash = 0ULL;
    uint64_t current_token_digest = 0ULL;
    uint64_t current_modal_digest = 0ULL;
    uint64_t current_counters = 0ULL;
    uint64_t current_state_digest = 0ULL;
    uint64_t decode_seed = 0ULL;
    uint64_t next_token_digest = 0ULL;
    uint64_t next_state_digest = 0ULL;
    uint32_t current_decode_steps = 0U;
    int32_t stored_route = MIZU_EXEC_ROUTE_NONE;
    int32_t stored_kind = 0;
    int64_t next_kv_tokens = 0;
    int64_t next_decode_steps = 0;

    if (updated_context_byte_count != NULL) *updated_context_byte_count = 0;
    if (emitted_token_count == NULL || token_value == NULL || stop_reason == NULL || status_code == NULL) return;

    *emitted_token_count = 0;
    *token_value = 0;
    *stop_reason = MIZU_STOP_REASON_NONE;
    if (!mizu_apple_context_is_valid(context_bytes, context_byte_count) ||
        updated_context_bytes == NULL || updated_context_byte_count == NULL ||
        updated_context_capacity < MIZU_APPLE_CONTEXT_TOTAL_BYTES || token_budget <= 0) {
        *status_code = MIZU_STATUS_INVALID_STATE;
        return;
    }

    stored_kind = (int32_t)((uint8_t)context_bytes[5]);
    stored_route = (int32_t)((uint8_t)context_bytes[6]);
    current_artifact_hash = mizu_apple_read_u64(context_bytes, context_byte_count, MIZU_APPLE_CONTEXT_ARTIFACT_OFFSET);
    current_token_digest = mizu_apple_read_u64(context_bytes, context_byte_count, MIZU_APPLE_CONTEXT_TOKEN_DIGEST_OFFSET);
    current_modal_digest = mizu_apple_read_u64(context_bytes, context_byte_count, MIZU_APPLE_CONTEXT_MODAL_DIGEST_OFFSET);
    current_counters = mizu_apple_read_u64(context_bytes, context_byte_count, MIZU_APPLE_CONTEXT_COUNTER_OFFSET);
    current_state_digest = mizu_apple_read_u64(context_bytes, context_byte_count, MIZU_APPLE_CONTEXT_STATE_DIGEST_OFFSET);
    current_decode_steps = (uint32_t)((current_counters >> 32) & 0xffffffffULL);

    if (stored_kind == MIZU_APPLE_CONTEXT_KIND_DECODE) {
        if (current_artifact_hash != (uint64_t)artifact_hash) {
            *status_code = MIZU_STATUS_INVALID_STATE;
            return;
        }
        if (stored_route != execution_route) {
            *status_code = MIZU_STATUS_INVALID_STATE;
            return;
        }
    }

    decode_seed = mizu_apple_mix_u64((uint64_t)payload_hash ^ (uint64_t)artifact_hash ^ mizu_apple_route_seed(execution_route) ^
                                     current_token_digest ^ current_modal_digest ^ current_state_digest ^
                                     (uint64_t)kv_before ^ ((uint64_t)token_budget << 32));
    *emitted_token_count = 1;
    *token_value = 1 + (int32_t)(decode_seed % 4095ULL);
    next_kv_tokens = kv_before + 1;
    next_decode_steps = (int64_t)current_decode_steps + 1;
    next_token_digest = mizu_apple_mix_u64(current_token_digest ^ (uint64_t)(uint32_t)(*token_value) ^
                                           mizu_apple_route_seed(execution_route) ^ (uint64_t)next_kv_tokens);
    next_state_digest = mizu_apple_mix_u64(current_state_digest ^ (uint64_t)artifact_hash ^
                                           next_token_digest ^ (uint64_t)next_decode_steps ^
                                           (uint64_t)(uint32_t)(*token_value));

    mizu_apple_write_context(execution_route, MIZU_APPLE_CONTEXT_KIND_DECODE, (uint64_t)artifact_hash,
                             next_token_digest, current_modal_digest, next_kv_tokens, next_decode_steps,
                             *token_value, *stop_reason, next_state_digest, decode_seed,
                             updated_context_bytes, updated_context_capacity, updated_context_byte_count);
    mizu_apple_stamp_workspace(workspace_buffer, workspace_bytes, next_state_digest, (uint8_t)(0x40 + execution_route));
    *status_code = (*updated_context_byte_count > 0) ? MIZU_STATUS_OK : MIZU_STATUS_INVALID_STATE;
}

#endif
