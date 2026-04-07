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
#define MIZU_CUDA_CONTEXT_PAGE_WORD_OFFSET INT32_C(64)
#define MIZU_CUDA_CONTEXT_RECENT_TOKEN_OFFSET INT32_C(96)
#define MIZU_CUDA_CONTEXT_WINDOW_META_OFFSET INT32_C(112)
#define MIZU_CUDA_CONTEXT_STATE_IMAGE_DIGEST_OFFSET INT32_C(120)
#define MIZU_CUDA_CONTEXT_KEY_PAYLOAD_OFFSET INT32_C(128)
#define MIZU_CUDA_CONTEXT_VALUE_PAYLOAD_OFFSET INT32_C(256)
#define MIZU_CUDA_CONTEXT_PAGE_DIGEST_OFFSET INT32_C(384)
#define MIZU_CUDA_CONTEXT_TOTAL_BYTES INT32_C(512)
#define MIZU_CUDA_CONTEXT_PAGE_COUNT 4
#define MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT 4
#define MIZU_CUDA_CONTEXT_PAGE_CAPACITY 8
#define MIZU_CUDA_CONTEXT_SLOT_COUNT (MIZU_CUDA_CONTEXT_PAGE_COUNT * MIZU_CUDA_CONTEXT_PAGE_CAPACITY)
#define MIZU_CUDA_CONTEXT_CHECKSUM_OFFSET UINT32_C(2166136261)
#define MIZU_CUDA_CONTEXT_CHECKSUM_PRIME UINT32_C(16777619)

static int32_t page_slot_base_index(int32_t page_index);
static int32_t unpack_summary_auxiliary_u16(uint64_t summary_word);

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

static uint64_t clamp_state_count32(int64_t count) {
    if (count <= 0) return UINT64_C(0);
    if (count > 0xffffffffLL) return UINT64_C(0xffffffff);
    return (uint64_t)count;
}

static uint64_t pack_state_counters(int64_t kv_token_count, int64_t decode_step_count) {
    return clamp_state_count32(kv_token_count) | (clamp_state_count32(decode_step_count) << 32);
}

static int64_t unpack_state_kv_tokens(uint64_t counters_word) {
    return (int64_t)(counters_word & UINT64_C(0xffffffff));
}

static int64_t unpack_state_decode_steps(uint64_t counters_word) {
    return (int64_t)((counters_word >> 32) & UINT64_C(0xffffffff));
}

static void write_context_u64(uint8_t *bytes, int32_t stored_count, int32_t offset, uint64_t value) {
    size_t copy_bytes;

    if (bytes == NULL || stored_count <= offset) return;

    copy_bytes = (size_t)(stored_count - offset < 8 ? stored_count - offset : 8);
    memcpy(bytes + offset, &value, copy_bytes);
}

static void write_context_i32(uint8_t *bytes, int32_t stored_count, int32_t offset, int32_t value) {
    size_t copy_bytes;

    if (bytes == NULL || stored_count <= offset) return;

    copy_bytes = (size_t)(stored_count - offset < 4 ? stored_count - offset : 4);
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

static int32_t read_context_i32(const int8_t *context_bytes, int32_t context_byte_count, int32_t offset) {
    int32_t value;
    size_t copy_bytes;

    value = 0;
    if (context_bytes == NULL || context_byte_count <= offset) return value;

    copy_bytes = (size_t)(context_byte_count - offset < 4 ? context_byte_count - offset : 4);
    memcpy(&value, ((const uint8_t *)context_bytes) + offset, copy_bytes);
    return value;
}

static uint64_t pack_kv_page_word(int64_t page_anchor,
                                  int64_t page_token_count,
                                  int32_t page_slot,
                                  int32_t page_kind) {
    return pack_context_summary(page_anchor, page_token_count, page_slot, page_kind);
}

static int32_t unpack_summary_leading_u16(uint64_t summary_word) {
    return (int32_t)(summary_word & UINT64_C(0xffff));
}

static int32_t unpack_summary_auxiliary_u16(uint64_t summary_word) {
    return (int32_t)((summary_word >> 16) & UINT64_C(0xffff));
}

static int32_t unpack_summary_control_a_u16(uint64_t summary_word) {
    return (int32_t)((summary_word >> 32) & UINT64_C(0xffff));
}

static int32_t unpack_summary_control_b_u16(uint64_t summary_word) {
    return (int32_t)((summary_word >> 48) & UINT64_C(0xffff));
}

static int32_t synthesize_value_lane(uint64_t base_seed,
                                     int32_t token_value,
                                     int32_t page_index,
                                     int32_t slot_index,
                                     int64_t page_anchor,
                                     int32_t page_kind) {
    uint64_t lane_seed;

    lane_seed = mix_u64(base_seed ^
        ((uint64_t)(uint32_t)token_value << 1) ^
        ((uint64_t)(uint32_t)(page_index + 1) << 17) ^
        ((uint64_t)(uint32_t)(slot_index + 1) << 33) ^
        ((uint64_t)(uint32_t)page_kind << 49) ^
        (uint64_t)(page_anchor < 0 ? 0 : page_anchor));
    return (int32_t)(lane_seed & UINT64_C(0x7fffffff));
}

static uint64_t digest_page_lane_state(uint64_t page_word,
                                       const int32_t key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                       const int32_t value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                       int32_t page_index) {
    uint64_t digest;
    int32_t slot_index;
    int32_t slot_base;

    if (unpack_summary_auxiliary_u16(page_word) <= 0) return UINT64_C(0);

    slot_base = page_slot_base_index(page_index);
    digest = mix_u64(UINT64_C(0xC0DEC0DE12345678) ^ page_word ^ ((uint64_t)(uint32_t)(page_index + 1) << 52));
    for (slot_index = 0; slot_index < MIZU_CUDA_CONTEXT_PAGE_CAPACITY; ++slot_index) {
        digest = mix_u64(digest ^
            (uint64_t)(uint32_t)key_slot_lanes[slot_base + slot_index] ^
            ((uint64_t)(uint32_t)value_slot_lanes[slot_base + slot_index] << 32) ^
            (uint64_t)(uint32_t)slot_index);
    }
    return digest;
}

static uint64_t digest_window_state(const uint64_t page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                    const int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                    const int32_t key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                    const int32_t value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                    const uint64_t page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                    uint64_t extra_seed) {
    uint64_t digest;
    int32_t index;

    digest = mix_u64(extra_seed ^ UINT64_C(0xD1E57A7E12345678));
    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        digest = mix_u64(digest ^ page_words[index] ^ page_lane_digests[index] ^ (uint64_t)index);
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
        digest = mix_u64(digest ^ (uint64_t)(uint32_t)recent_tokens[index] ^ ((uint64_t)index << 40));
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_SLOT_COUNT; ++index) {
        digest = mix_u64(digest ^
            (uint64_t)(uint32_t)key_slot_lanes[index] ^
            ((uint64_t)(uint32_t)value_slot_lanes[index] << 32) ^
            ((uint64_t)index << 28));
    }
    return digest;
}

static void build_prefill_state_block(uint64_t seed,
                                      uint64_t artifact_hash,
                                      int64_t token_count,
                                      int64_t modal_byte_count,
                                      int32_t staged_modal_count,
                                      int64_t consumed_token_count,
                                      uint64_t state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      uint64_t *summary_word) {
    int64_t kv_token_count;

    kv_token_count = consumed_token_count > 0 ? consumed_token_count : (token_count > 0 ? token_count : 0);
    state_lanes[0] = mix_u64(seed ^ artifact_hash ^ (uint64_t)token_count ^ UINT64_C(0x54F4C0DA12345678));
    state_lanes[1] = mix_u64(seed ^ ((uint64_t)modal_byte_count << 1) ^
                             ((uint64_t)(uint32_t)staged_modal_count << 33) ^
                             UINT64_C(0x1D1A7E5EABCDEF01));
    state_lanes[2] = pack_state_counters(kv_token_count, 0);
    state_lanes[3] = mix_u64(state_lanes[0] ^ state_lanes[1] ^ state_lanes[2] ^ artifact_hash ^
                             UINT64_C(0xC0DA5EED5EED1234));
    if (summary_word != NULL) {
        *summary_word = pack_context_summary(kv_token_count, modal_byte_count, staged_modal_count, 0);
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
    int64_t current_kv_tokens;
    int64_t current_decode_steps;
    int64_t effective_kv_before;
    int64_t next_kv_tokens;
    int64_t next_decode_steps;

    current_kv_tokens = unpack_state_kv_tokens(current_state_lanes[2]);
    current_decode_steps = unpack_state_decode_steps(current_state_lanes[2]);
    effective_kv_before = kv_before > current_kv_tokens ? kv_before : current_kv_tokens;
    next_kv_tokens = effective_kv_before + (emitted_token_count > 0 ? emitted_token_count : 0);
    next_decode_steps = current_decode_steps + (emitted_token_count > 0 ? 1 : 0);

    next_state_lanes[0] = mix_u64(current_state_lanes[0] ^ artifact_hash ^ (uint64_t)next_kv_tokens ^
                                  ((uint64_t)(uint32_t)token_value << 32));
    next_state_lanes[1] = current_state_lanes[1];
    next_state_lanes[2] = pack_state_counters(next_kv_tokens, next_decode_steps);
    next_state_lanes[3] = mix_u64(current_state_lanes[3] ^ next_state_lanes[0] ^ next_state_lanes[1] ^
                                  next_state_lanes[2] ^ (uint64_t)token_budget ^
                                  ((uint64_t)(uint32_t)stop_reason << 48) ^
                                  UINT64_C(0x1EAFCAFE5EED4321));
    if (summary_word != NULL) {
        *summary_word = pack_context_summary(next_kv_tokens, next_decode_steps, token_value, stop_reason);
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

static int32_t page_slot_base_index(int32_t page_index) {
    return page_index * MIZU_CUDA_CONTEXT_PAGE_CAPACITY;
}

static void build_prefill_window_block(uint64_t seed,
                                       const int32_t *token_values,
                                       int64_t token_count,
                                       int64_t kv_token_count,
                                       uint64_t page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                       int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                       int32_t key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                       int32_t value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                       uint64_t page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                       uint64_t *window_meta,
                                       uint64_t *state_image_digest) {
    int32_t valid_page_count;
    int32_t current_page_index;
    int32_t recent_token_count;
    int64_t remaining_tokens;
    int64_t page_anchor;
    int32_t index;

    valid_page_count = 0;
    current_page_index = 0;
    recent_token_count = 0;
    remaining_tokens = kv_token_count > 0 ? kv_token_count : 0;
    page_anchor = 0;

    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        page_words[index] = UINT64_C(0);
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
        recent_tokens[index] = 0;
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_SLOT_COUNT; ++index) {
        key_slot_lanes[index] = 0;
        value_slot_lanes[index] = 0;
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        page_lane_digests[index] = UINT64_C(0);
    }

    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT && remaining_tokens > 0; ++index) {
        int64_t page_token_count;
        int32_t slot_base;
        int32_t slot_index;
        int32_t page_kind;

        page_token_count = remaining_tokens > MIZU_CUDA_CONTEXT_PAGE_CAPACITY ?
            MIZU_CUDA_CONTEXT_PAGE_CAPACITY : remaining_tokens;
        page_kind = 1;
        page_words[index] = pack_kv_page_word(page_anchor, page_token_count, index, page_kind);
        slot_base = page_slot_base_index(index);
        for (slot_index = 0; slot_index < (int32_t)page_token_count; ++slot_index) {
            int64_t token_index;
            int32_t token_value;

            token_index = page_anchor + slot_index;
            if (token_values != NULL && token_index < token_count) {
                token_value = token_values[token_index];
            } else {
                token_value = (int32_t)((token_index + 1) & 0x7fffffff);
            }
            key_slot_lanes[slot_base + slot_index] = token_value;
            value_slot_lanes[slot_base + slot_index] = synthesize_value_lane(seed, token_value, index, slot_index,
                page_anchor, page_kind);
        }
        page_lane_digests[index] = digest_page_lane_state(page_words[index], key_slot_lanes, value_slot_lanes, index);
        page_anchor += page_token_count;
        remaining_tokens -= page_token_count;
        valid_page_count = index + 1;
    }
    if (valid_page_count > 0) current_page_index = valid_page_count - 1;

    recent_token_count = (int32_t)(token_count > MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT ?
        MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT : (token_count > 0 ? token_count : 0));
    if (token_values != NULL && recent_token_count > 0) {
        int64_t token_start;

        token_start = token_count - recent_token_count;
        for (index = 0; index < recent_token_count; ++index) {
            recent_tokens[index] = token_values[token_start + index];
        }
    }

    if (window_meta != NULL) {
        *window_meta = pack_context_summary(current_page_index, valid_page_count, recent_token_count, 0);
    }
    if (state_image_digest != NULL) {
        *state_image_digest = digest_window_state(page_words, recent_tokens, key_slot_lanes, value_slot_lanes,
            page_lane_digests, seed ^ (uint64_t)kv_token_count);
    }
}

static void extract_context_window_block(const int8_t *context_bytes,
                                         int32_t context_byte_count,
                                         uint64_t page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                         int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                         int32_t key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                         int32_t value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                         uint64_t page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                         uint64_t *window_meta,
                                         uint64_t *state_image_digest) {
    int32_t index;

    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        page_words[index] = read_context_u64(context_bytes, context_byte_count,
            MIZU_CUDA_CONTEXT_PAGE_WORD_OFFSET + (index * 8));
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
        recent_tokens[index] = read_context_i32(context_bytes, context_byte_count,
            MIZU_CUDA_CONTEXT_RECENT_TOKEN_OFFSET + (index * 4));
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_SLOT_COUNT; ++index) {
        key_slot_lanes[index] = read_context_i32(context_bytes, context_byte_count,
            MIZU_CUDA_CONTEXT_KEY_PAYLOAD_OFFSET + (index * 4));
        value_slot_lanes[index] = read_context_i32(context_bytes, context_byte_count,
            MIZU_CUDA_CONTEXT_VALUE_PAYLOAD_OFFSET + (index * 4));
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        page_lane_digests[index] = read_context_u64(context_bytes, context_byte_count,
            MIZU_CUDA_CONTEXT_PAGE_DIGEST_OFFSET + (index * 8));
    }
    if (window_meta != NULL) {
        *window_meta = read_context_u64(context_bytes, context_byte_count, MIZU_CUDA_CONTEXT_WINDOW_META_OFFSET);
    }
    if (state_image_digest != NULL) {
        *state_image_digest = read_context_u64(context_bytes, context_byte_count,
            MIZU_CUDA_CONTEXT_STATE_IMAGE_DIGEST_OFFSET);
    }
}

static void write_context_window_block(uint8_t *bytes,
                                       int32_t stored_count,
                                       const uint64_t page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                       const int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                       const int32_t key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                       const int32_t value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                       const uint64_t page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                       uint64_t window_meta,
                                       uint64_t state_image_digest) {
    int32_t index;

    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_PAGE_WORD_OFFSET + (index * 8), page_words[index]);
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
        write_context_i32(bytes, stored_count, MIZU_CUDA_CONTEXT_RECENT_TOKEN_OFFSET + (index * 4), recent_tokens[index]);
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_SLOT_COUNT; ++index) {
        write_context_i32(bytes, stored_count, MIZU_CUDA_CONTEXT_KEY_PAYLOAD_OFFSET + (index * 4), key_slot_lanes[index]);
        write_context_i32(bytes, stored_count, MIZU_CUDA_CONTEXT_VALUE_PAYLOAD_OFFSET + (index * 4), value_slot_lanes[index]);
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_PAGE_DIGEST_OFFSET + (index * 8), page_lane_digests[index]);
    }
    write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_WINDOW_META_OFFSET, window_meta);
    write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_STATE_IMAGE_DIGEST_OFFSET, state_image_digest);
}

static void build_decode_window_block(const uint64_t current_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      const int32_t current_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                      const int32_t current_key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                      const int32_t current_value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                      const uint64_t current_page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      uint64_t current_window_meta,
                                      uint64_t current_state_image_digest,
                                      int64_t next_kv_tokens,
                                      int64_t emitted_token_count,
                                      int32_t token_value,
                                      uint64_t next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      int32_t next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                      int32_t next_key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                      int32_t next_value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                      uint64_t next_page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      uint64_t *next_window_meta,
                                      uint64_t *next_state_image_digest) {
    int32_t current_page_index;
    int32_t valid_page_count;
    int32_t recent_token_count;
    int32_t index;
    int emitted;
    int appended_to_existing_page;
    int32_t target_page_index;

    current_page_index = unpack_summary_leading_u16(current_window_meta);
    valid_page_count = unpack_summary_auxiliary_u16(current_window_meta);
    recent_token_count = unpack_summary_control_a_u16(current_window_meta);

    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        next_page_words[index] = current_page_words[index];
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
        next_recent_tokens[index] = current_recent_tokens[index];
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_SLOT_COUNT; ++index) {
        next_key_slot_lanes[index] = current_key_slot_lanes[index];
        next_value_slot_lanes[index] = current_value_slot_lanes[index];
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        next_page_lane_digests[index] = current_page_lane_digests[index];
    }

    if (valid_page_count < 0) valid_page_count = 0;
    if (valid_page_count > MIZU_CUDA_CONTEXT_PAGE_COUNT) valid_page_count = MIZU_CUDA_CONTEXT_PAGE_COUNT;
    if (recent_token_count < 0) recent_token_count = 0;
    if (recent_token_count > MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT) {
        recent_token_count = MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT;
    }
    if (current_page_index < 0) current_page_index = 0;
    if (current_page_index >= MIZU_CUDA_CONTEXT_PAGE_COUNT) {
        current_page_index = MIZU_CUDA_CONTEXT_PAGE_COUNT - 1;
    }

    emitted = emitted_token_count > 0 ? (int)emitted_token_count : 0;
    if (emitted > 0) {
        int64_t new_page_anchor;

        new_page_anchor = next_kv_tokens - emitted_token_count;
        appended_to_existing_page = 0;
        target_page_index = current_page_index;
        if (valid_page_count > 0) {
            uint64_t current_page_word;
            int32_t current_page_anchor;
            int32_t current_page_fill;

            current_page_word = next_page_words[current_page_index];
            current_page_anchor = unpack_summary_leading_u16(current_page_word);
            current_page_fill = unpack_summary_auxiliary_u16(current_page_word);
            if (current_page_fill > 0 &&
                current_page_fill < MIZU_CUDA_CONTEXT_PAGE_CAPACITY &&
                current_page_anchor + current_page_fill == new_page_anchor) {
                next_page_words[current_page_index] = pack_kv_page_word(current_page_anchor,
                    current_page_fill + emitted, current_page_index, 2);
                next_key_slot_lanes[page_slot_base_index(current_page_index) + current_page_fill] = token_value;
                next_value_slot_lanes[page_slot_base_index(current_page_index) + current_page_fill] =
                    synthesize_value_lane(current_state_image_digest, token_value, current_page_index,
                    current_page_fill, new_page_anchor, 2);
                appended_to_existing_page = 1;
                target_page_index = current_page_index;
            }
        }

        if (!appended_to_existing_page) {
            if (valid_page_count < MIZU_CUDA_CONTEXT_PAGE_COUNT) {
                current_page_index = valid_page_count;
                valid_page_count += 1;
            } else {
                for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT - 1; ++index) {
                    uint64_t shifted_page_word;
                    int32_t dst_slot_base;
                    int32_t src_slot_base;
                    int32_t slot_index;

                    shifted_page_word = next_page_words[index + 1];
                    next_page_words[index] = pack_kv_page_word(unpack_summary_leading_u16(shifted_page_word),
                        unpack_summary_auxiliary_u16(shifted_page_word), index,
                        unpack_summary_control_b_u16(shifted_page_word));
                    dst_slot_base = page_slot_base_index(index);
                    src_slot_base = page_slot_base_index(index + 1);
                    for (slot_index = 0; slot_index < MIZU_CUDA_CONTEXT_PAGE_CAPACITY; ++slot_index) {
                        next_key_slot_lanes[dst_slot_base + slot_index] = next_key_slot_lanes[src_slot_base + slot_index];
                        next_value_slot_lanes[dst_slot_base + slot_index] = next_value_slot_lanes[src_slot_base + slot_index];
                    }
                    next_page_lane_digests[index] = next_page_lane_digests[index + 1];
                }
                next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT - 1] = UINT64_C(0);
                {
                    int32_t last_slot_base;
                    int32_t slot_index;

                    last_slot_base = page_slot_base_index(MIZU_CUDA_CONTEXT_PAGE_COUNT - 1);
                    for (slot_index = 0; slot_index < MIZU_CUDA_CONTEXT_PAGE_CAPACITY; ++slot_index) {
                        next_key_slot_lanes[last_slot_base + slot_index] = 0;
                        next_value_slot_lanes[last_slot_base + slot_index] = 0;
                    }
                }
                next_page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT - 1] = UINT64_C(0);
                current_page_index = MIZU_CUDA_CONTEXT_PAGE_COUNT - 1;
            }
            next_page_words[current_page_index] = pack_kv_page_word(new_page_anchor, emitted, current_page_index, 2);
            next_key_slot_lanes[page_slot_base_index(current_page_index)] = token_value;
            next_value_slot_lanes[page_slot_base_index(current_page_index)] = synthesize_value_lane(
                current_state_image_digest, token_value, current_page_index, 0, new_page_anchor, 2);
            target_page_index = current_page_index;
        }

        if (recent_token_count < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT) {
            next_recent_tokens[recent_token_count] = token_value;
            recent_token_count += 1;
        } else {
            for (index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT - 1; ++index) {
                next_recent_tokens[index] = next_recent_tokens[index + 1];
            }
            next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT - 1] = token_value;
        }
        current_page_index = target_page_index;
    }

    if (next_window_meta != NULL) {
        *next_window_meta = pack_context_summary(current_page_index, valid_page_count, recent_token_count, 0);
    }
    for (index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
        next_page_lane_digests[index] = digest_page_lane_state(next_page_words[index], next_key_slot_lanes,
            next_value_slot_lanes, index);
    }
    if (next_state_image_digest != NULL) {
        *next_state_image_digest = digest_window_state(next_page_words, next_recent_tokens, next_key_slot_lanes,
            next_value_slot_lanes, next_page_lane_digests, current_state_image_digest ^
            (uint64_t)next_kv_tokens ^ (uint64_t)(uint32_t)token_value);
    }
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
                                       const int32_t *token_values,
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
    uint64_t page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT];
    int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT];
    int32_t key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT];
    int32_t value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT];
    uint64_t page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT];
    uint64_t window_meta;
    uint64_t state_image_digest;
    int32_t stored_count;

    if (context_byte_count == NULL) return;

    *context_byte_count = 0;
    if (context_bytes == NULL || context_capacity <= 0) return;

    bytes = (uint8_t *)context_bytes;
    stored_count = context_capacity < MIZU_CUDA_CONTEXT_TOTAL_BYTES ? context_capacity : MIZU_CUDA_CONTEXT_TOTAL_BYTES;
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
    build_prefill_window_block(seed, token_values, token_count, consumed_token_count, page_words, recent_tokens,
                               key_slot_lanes, value_slot_lanes, page_lane_digests,
                               &window_meta, &state_image_digest);
    write_context_state_block(bytes, stored_count, state_lanes, artifact_hash, summary_word);
    write_context_window_block(bytes, stored_count, page_words, recent_tokens, key_slot_lanes, value_slot_lanes,
                               page_lane_digests, window_meta, state_image_digest);

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
                                      const uint64_t next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      const int32_t next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                      const int32_t next_key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                      const int32_t next_value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT],
                                      const uint64_t next_page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      uint64_t next_window_meta,
                                      uint64_t next_state_image_digest,
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
    stored_count = context_capacity < MIZU_CUDA_CONTEXT_TOTAL_BYTES ? context_capacity : MIZU_CUDA_CONTEXT_TOTAL_BYTES;
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
    write_context_window_block(bytes, stored_count, next_page_words, next_recent_tokens, next_key_slot_lanes,
                               next_value_slot_lanes, next_page_lane_digests, next_window_meta, next_state_image_digest);

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
    fill_prefill_context_bytes(seed, (uint64_t)artifact_hash, token_values, token_count, modal_byte_count,
                               staged_modal_count, *consumed_token_count, context_bytes, context_capacity,
                               context_byte_count);
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
    uint64_t current_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT];
    uint64_t next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT];
    uint64_t summary_word;
    uint64_t current_window_meta;
    uint64_t next_window_meta;
    uint64_t current_state_image_digest;
    uint64_t next_state_image_digest;
    int32_t current_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT];
    int32_t next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT];
    int32_t current_key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT];
    int32_t current_value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT];
    int32_t next_key_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT];
    int32_t next_value_slot_lanes[MIZU_CUDA_CONTEXT_SLOT_COUNT];
    uint64_t current_page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT];
    uint64_t next_page_lane_digests[MIZU_CUDA_CONTEXT_PAGE_COUNT];

    if (emitted_token_count == NULL || token_value == NULL || stop_reason == NULL ||
        status_code == NULL) {
        return;
    }

    extract_context_state_block(context_bytes, context_byte_count, current_state_lanes, NULL, &summary_word);
    extract_context_window_block(context_bytes, context_byte_count, current_page_words, current_recent_tokens,
                                 current_key_slot_lanes, current_value_slot_lanes, current_page_lane_digests,
                                 &current_window_meta, &current_state_image_digest);
    seed = (uint64_t)payload_hash;
    seed = mix_u64(seed ^ current_state_lanes[0] ^ current_state_lanes[1]);
    seed = mix_u64(seed ^ current_state_lanes[2] ^ current_state_lanes[3] ^ summary_word);
    seed = mix_u64(seed ^ current_window_meta ^ current_state_image_digest);
    seed ^= (uint64_t)kv_before * UINT64_C(0x9e3779b97f4a7c15);
    seed ^= (uint64_t)token_budget * UINT64_C(0xbf58476d1ce4e5b9);
    seed = mix_u64(seed);

    *emitted_token_count = token_budget > 0 ? 1 : 0;
    *token_value = 1 + (int32_t)(seed % UINT64_C(4095));
    *stop_reason = MIZU_STOP_REASON_NONE;
    build_decode_state_block(current_state_lanes, (uint64_t)artifact_hash, kv_before, token_budget,
                             *emitted_token_count, *token_value, *stop_reason, next_state_lanes, &summary_word);
    build_decode_window_block(current_page_words, current_recent_tokens, current_key_slot_lanes,
                              current_value_slot_lanes, current_page_lane_digests, current_window_meta,
                              current_state_image_digest, unpack_state_kv_tokens(next_state_lanes[2]),
                              *emitted_token_count, *token_value, next_page_words, next_recent_tokens,
                              next_key_slot_lanes, next_value_slot_lanes, next_page_lane_digests,
                              &next_window_meta, &next_state_image_digest);
    fill_decode_context_bytes(seed ^ (uint64_t)(uint32_t)(*token_value), (uint64_t)artifact_hash,
                              next_state_lanes, summary_word, next_page_words, next_recent_tokens,
                              next_key_slot_lanes, next_value_slot_lanes, next_page_lane_digests,
                              next_window_meta, next_state_image_digest, updated_context_bytes,
                              updated_context_capacity, updated_context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, next_state_lanes[1] ^ next_state_lanes[3] ^
                           next_state_image_digest ^ summary_word, UINT8_C(4));
    *status_code = MIZU_STATUS_OK;
}
