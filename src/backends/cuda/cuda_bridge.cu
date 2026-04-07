#include "mizu.h"
#include "cuda_bridge.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>
#include <string.h>

namespace {

constexpr unsigned char MIZU_CUDA_CONTEXT_MAGIC_0 = static_cast<unsigned char>('M');
constexpr unsigned char MIZU_CUDA_CONTEXT_MAGIC_1 = static_cast<unsigned char>('Z');
constexpr unsigned char MIZU_CUDA_CONTEXT_MAGIC_2 = static_cast<unsigned char>('C');
constexpr unsigned char MIZU_CUDA_CONTEXT_MAGIC_3 = static_cast<unsigned char>('T');
constexpr unsigned char MIZU_CUDA_CONTEXT_VERSION = 1U;
constexpr unsigned char MIZU_CUDA_CONTEXT_KIND_PREFILL = 1U;
constexpr unsigned char MIZU_CUDA_CONTEXT_KIND_DECODE = 2U;
constexpr int32_t MIZU_CUDA_CONTEXT_HEADER_SIZE = 16;
constexpr int32_t MIZU_CUDA_CONTEXT_STATE_LANES = 4;
constexpr int32_t MIZU_CUDA_CONTEXT_ARTIFACT_OFFSET = 48;
constexpr int32_t MIZU_CUDA_CONTEXT_SUMMARY_OFFSET = 56;
constexpr int32_t MIZU_CUDA_CONTEXT_PAGE_WORD_OFFSET = 64;
constexpr int32_t MIZU_CUDA_CONTEXT_RECENT_TOKEN_OFFSET = 96;
constexpr int32_t MIZU_CUDA_CONTEXT_WINDOW_META_OFFSET = 112;
constexpr int32_t MIZU_CUDA_CONTEXT_STATE_IMAGE_DIGEST_OFFSET = 120;
constexpr int32_t MIZU_CUDA_CONTEXT_TOTAL_BYTES = 128;
constexpr int32_t MIZU_CUDA_CONTEXT_PAGE_COUNT = 4;
constexpr int32_t MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT = 4;
constexpr int32_t MIZU_CUDA_CONTEXT_PAGE_CAPACITY = 16;
constexpr uint32_t MIZU_CUDA_CONTEXT_CHECKSUM_OFFSET = 2166136261U;
constexpr uint32_t MIZU_CUDA_CONTEXT_CHECKSUM_PRIME = 16777619U;

__device__ __forceinline__ unsigned long long mix_u64(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline unsigned long long mix_u64_host(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline uint32_t compute_context_checksum(const unsigned char *bytes, int32_t stored_count) {
  uint32_t checksum = MIZU_CUDA_CONTEXT_CHECKSUM_OFFSET;
  if (bytes == nullptr || stored_count <= MIZU_CUDA_CONTEXT_HEADER_SIZE) {
    checksum ^= static_cast<uint32_t>(stored_count > 0 ? stored_count : 0);
    checksum *= MIZU_CUDA_CONTEXT_CHECKSUM_PRIME;
    return checksum == 0U ? 1U : checksum;
  }

  for (int32_t index = MIZU_CUDA_CONTEXT_HEADER_SIZE; index < stored_count; ++index) {
    checksum ^= static_cast<uint32_t>(bytes[index]);
    checksum *= MIZU_CUDA_CONTEXT_CHECKSUM_PRIME;
  }
  checksum ^= static_cast<uint32_t>(stored_count);
  checksum *= MIZU_CUDA_CONTEXT_CHECKSUM_PRIME;
  return checksum == 0U ? 1U : checksum;
}

inline unsigned long long pack_context_summary(int64_t leading_count,
                                               int64_t auxiliary_count,
                                               int32_t control_a,
                                               int32_t control_b) {
  const unsigned long long leading_clamped =
      static_cast<unsigned long long>(leading_count < 0 ? 0 : (leading_count > 65535 ? 65535 : leading_count));
  const unsigned long long auxiliary_clamped =
      static_cast<unsigned long long>(auxiliary_count < 0 ? 0 : (auxiliary_count > 65535 ? 65535 : auxiliary_count));
  const unsigned long long control_a_bits = static_cast<unsigned long long>(static_cast<uint32_t>(control_a) & 0xffffU);
  const unsigned long long control_b_bits = static_cast<unsigned long long>(static_cast<uint32_t>(control_b) & 0xffffU);

  return leading_clamped | (auxiliary_clamped << 16) | (control_a_bits << 32) | (control_b_bits << 48);
}

inline unsigned long long clamp_state_count32(int64_t count) {
  if (count <= 0) return 0ULL;
  if (count > 0xffffffffLL) return 0xffffffffULL;
  return static_cast<unsigned long long>(count);
}

inline unsigned long long pack_state_counters(int64_t kv_token_count, int64_t decode_step_count) {
  return clamp_state_count32(kv_token_count) | (clamp_state_count32(decode_step_count) << 32);
}

inline int64_t unpack_state_kv_tokens(unsigned long long counters_word) {
  return static_cast<int64_t>(counters_word & 0xffffffffULL);
}

inline int64_t unpack_state_decode_steps(unsigned long long counters_word) {
  return static_cast<int64_t>((counters_word >> 32) & 0xffffffffULL);
}

inline void write_context_u64(unsigned char *bytes, int32_t stored_count, int32_t offset,
                              unsigned long long value) {
  if (bytes == nullptr || stored_count <= offset) return;
  const size_t copy_bytes = static_cast<size_t>(stored_count - offset < 8 ? stored_count - offset : 8);
  memcpy(bytes + offset, &value, copy_bytes);
}

inline void write_context_i32(unsigned char *bytes, int32_t stored_count, int32_t offset, int32_t value) {
  if (bytes == nullptr || stored_count <= offset) return;
  const size_t copy_bytes = static_cast<size_t>(stored_count - offset < 4 ? stored_count - offset : 4);
  memcpy(bytes + offset, &value, copy_bytes);
}

inline unsigned long long read_context_u64(const int8_t *context_bytes, int32_t context_byte_count, int32_t offset) {
  unsigned long long value = 0ULL;
  if (context_bytes == nullptr || context_byte_count <= offset) return value;
  const size_t copy_bytes = static_cast<size_t>(context_byte_count - offset < 8 ? context_byte_count - offset : 8);
  memcpy(&value, reinterpret_cast<const unsigned char *>(context_bytes) + offset, copy_bytes);
  return value;
}

inline int32_t read_context_i32(const int8_t *context_bytes, int32_t context_byte_count, int32_t offset) {
  int32_t value = 0;
  if (context_bytes == nullptr || context_byte_count <= offset) return value;
  const size_t copy_bytes = static_cast<size_t>(context_byte_count - offset < 4 ? context_byte_count - offset : 4);
  memcpy(&value, reinterpret_cast<const unsigned char *>(context_bytes) + offset, copy_bytes);
  return value;
}

inline unsigned long long pack_kv_page_word(int64_t page_anchor,
                                            int64_t page_token_count,
                                            int32_t page_slot,
                                            int32_t page_kind) {
  return pack_context_summary(page_anchor, page_token_count, page_slot, page_kind);
}

inline int32_t unpack_summary_leading_u16(unsigned long long summary_word) {
  return static_cast<int32_t>(summary_word & 0xffffULL);
}

inline int32_t unpack_summary_auxiliary_u16(unsigned long long summary_word) {
  return static_cast<int32_t>((summary_word >> 16) & 0xffffULL);
}

inline int32_t unpack_summary_control_a_u16(unsigned long long summary_word) {
  return static_cast<int32_t>((summary_word >> 32) & 0xffffULL);
}

inline unsigned long long digest_window_state(const unsigned long long page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                              const int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                              unsigned long long extra_seed) {
  unsigned long long digest = mix_u64_host(extra_seed ^ 0xD1E57A7E12345678ULL);
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
    digest = mix_u64_host(digest ^ page_words[index] ^ static_cast<unsigned long long>(index));
  }
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
    digest = mix_u64_host(digest ^ static_cast<unsigned long long>(static_cast<uint32_t>(recent_tokens[index])) ^
                          (static_cast<unsigned long long>(index) << 40));
  }
  return digest;
}

inline void build_prefill_state_block(unsigned long long seed,
                                      unsigned long long artifact_hash,
                                      int64_t token_count,
                                      int64_t modal_byte_count,
                                      int32_t staged_modal_count,
                                      int64_t consumed_token_count,
                                      unsigned long long state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      unsigned long long *summary_word) {
  const int64_t kv_token_count = consumed_token_count > 0 ? consumed_token_count : (token_count > 0 ? token_count : 0);
  state_lanes[0] = mix_u64_host(seed ^ artifact_hash ^ static_cast<unsigned long long>(token_count) ^
                                0x54F4C0DA12345678ULL);
  state_lanes[1] = mix_u64_host(seed ^ (static_cast<unsigned long long>(modal_byte_count) << 1) ^
                                (static_cast<unsigned long long>(staged_modal_count) << 33) ^
                                0x1D1A7E5EABCDEF01ULL);
  state_lanes[2] = pack_state_counters(kv_token_count, 0);
  state_lanes[3] = mix_u64_host(state_lanes[0] ^ state_lanes[1] ^ state_lanes[2] ^ artifact_hash ^
                                0xC0DA5EED5EED1234ULL);
  if (summary_word != nullptr) {
    *summary_word = pack_context_summary(kv_token_count, modal_byte_count, staged_modal_count, 0);
  }
}

inline void build_decode_state_block(const unsigned long long current_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                     unsigned long long artifact_hash,
                                     int64_t kv_before,
                                     int64_t token_budget,
                                     int64_t emitted_token_count,
                                     int32_t token_value,
                                     int32_t stop_reason,
                                     unsigned long long next_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                     unsigned long long *summary_word) {
  const int64_t current_kv_tokens = unpack_state_kv_tokens(current_state_lanes[2]);
  const int64_t current_decode_steps = unpack_state_decode_steps(current_state_lanes[2]);
  const int64_t effective_kv_before = kv_before > current_kv_tokens ? kv_before : current_kv_tokens;
  const int64_t next_kv_tokens = effective_kv_before + (emitted_token_count > 0 ? emitted_token_count : 0);
  const int64_t next_decode_steps = current_decode_steps + (emitted_token_count > 0 ? 1 : 0);

  next_state_lanes[0] = mix_u64_host(current_state_lanes[0] ^ artifact_hash ^
                                     static_cast<unsigned long long>(next_kv_tokens) ^
                                     (static_cast<unsigned long long>(static_cast<uint32_t>(token_value)) << 32));
  next_state_lanes[1] = current_state_lanes[1];
  next_state_lanes[2] = pack_state_counters(next_kv_tokens, next_decode_steps);
  next_state_lanes[3] = mix_u64_host(current_state_lanes[3] ^ next_state_lanes[0] ^ next_state_lanes[1] ^
                                     next_state_lanes[2] ^ static_cast<unsigned long long>(token_budget) ^
                                     (static_cast<unsigned long long>(static_cast<uint32_t>(stop_reason)) << 48) ^
                                     0x1EAFCAFE5EED4321ULL);
  if (summary_word != nullptr) {
    *summary_word = pack_context_summary(next_kv_tokens, next_decode_steps, token_value, stop_reason);
  }
}

inline void extract_context_state_block(const int8_t *context_bytes,
                                        int32_t context_byte_count,
                                        unsigned long long state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                        unsigned long long *artifact_hash,
                                        unsigned long long *summary_word) {
  for (int32_t lane_index = 0; lane_index < MIZU_CUDA_CONTEXT_STATE_LANES; ++lane_index) {
    state_lanes[lane_index] = read_context_u64(context_bytes, context_byte_count, 16 + (lane_index * 8));
  }
  if (artifact_hash != nullptr) {
    *artifact_hash = read_context_u64(context_bytes, context_byte_count, MIZU_CUDA_CONTEXT_ARTIFACT_OFFSET);
  }
  if (summary_word != nullptr) {
    *summary_word = read_context_u64(context_bytes, context_byte_count, MIZU_CUDA_CONTEXT_SUMMARY_OFFSET);
  }
}

inline void write_context_state_block(unsigned char *bytes,
                                      int32_t stored_count,
                                      const unsigned long long state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      unsigned long long artifact_hash,
                                      unsigned long long summary_word) {
  for (int32_t lane_index = 0; lane_index < MIZU_CUDA_CONTEXT_STATE_LANES; ++lane_index) {
    write_context_u64(bytes, stored_count, 16 + (lane_index * 8), state_lanes[lane_index]);
  }
  write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_ARTIFACT_OFFSET, artifact_hash);
  write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_SUMMARY_OFFSET, summary_word);
}

inline void build_prefill_window_block(const int32_t *token_values,
                                       int64_t token_count,
                                       int64_t kv_token_count,
                                       unsigned long long page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                       int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                       unsigned long long *window_meta,
                                       unsigned long long *state_image_digest) {
  int32_t valid_page_count = 0;
  int32_t current_page_index = 0;
  int32_t recent_token_count = 0;
  int64_t remaining_tokens = kv_token_count > 0 ? kv_token_count : 0;
  int64_t page_anchor = 0;

  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
    page_words[index] = 0ULL;
  }
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
    recent_tokens[index] = 0;
  }

  for (int32_t page_index = 0; page_index < MIZU_CUDA_CONTEXT_PAGE_COUNT && remaining_tokens > 0; ++page_index) {
    const int64_t page_token_count = remaining_tokens > MIZU_CUDA_CONTEXT_PAGE_CAPACITY ?
      MIZU_CUDA_CONTEXT_PAGE_CAPACITY : remaining_tokens;
    page_words[page_index] = pack_kv_page_word(page_anchor, page_token_count, page_index, 1);
    page_anchor += page_token_count;
    remaining_tokens -= page_token_count;
    valid_page_count = page_index + 1;
  }
  if (valid_page_count > 0) current_page_index = valid_page_count - 1;

  recent_token_count = static_cast<int32_t>(token_count > MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT ?
    MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT : (token_count > 0 ? token_count : 0));
  if (token_values != nullptr && recent_token_count > 0) {
    const int64_t token_start = token_count - recent_token_count;
    for (int32_t index = 0; index < recent_token_count; ++index) {
      recent_tokens[index] = token_values[token_start + index];
    }
  }

  if (window_meta != nullptr) {
    *window_meta = pack_context_summary(current_page_index, valid_page_count, recent_token_count, 0);
  }
  if (state_image_digest != nullptr) {
    *state_image_digest = digest_window_state(page_words, recent_tokens,
      static_cast<unsigned long long>(kv_token_count));
  }
}

inline void extract_context_window_block(const int8_t *context_bytes,
                                         int32_t context_byte_count,
                                         unsigned long long page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                         int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                         unsigned long long *window_meta,
                                         unsigned long long *state_image_digest) {
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
    page_words[index] = read_context_u64(context_bytes, context_byte_count,
      MIZU_CUDA_CONTEXT_PAGE_WORD_OFFSET + (index * 8));
  }
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
    recent_tokens[index] = read_context_i32(context_bytes, context_byte_count,
      MIZU_CUDA_CONTEXT_RECENT_TOKEN_OFFSET + (index * 4));
  }
  if (window_meta != nullptr) {
    *window_meta = read_context_u64(context_bytes, context_byte_count, MIZU_CUDA_CONTEXT_WINDOW_META_OFFSET);
  }
  if (state_image_digest != nullptr) {
    *state_image_digest = read_context_u64(context_bytes, context_byte_count,
      MIZU_CUDA_CONTEXT_STATE_IMAGE_DIGEST_OFFSET);
  }
}

inline void write_context_window_block(unsigned char *bytes,
                                       int32_t stored_count,
                                       const unsigned long long page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                       const int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                       unsigned long long window_meta,
                                       unsigned long long state_image_digest) {
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
    write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_PAGE_WORD_OFFSET + (index * 8), page_words[index]);
  }
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
    write_context_i32(bytes, stored_count, MIZU_CUDA_CONTEXT_RECENT_TOKEN_OFFSET + (index * 4), recent_tokens[index]);
  }
  write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_WINDOW_META_OFFSET, window_meta);
  write_context_u64(bytes, stored_count, MIZU_CUDA_CONTEXT_STATE_IMAGE_DIGEST_OFFSET, state_image_digest);
}

inline void build_decode_window_block(const unsigned long long current_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      const int32_t current_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                      unsigned long long current_window_meta,
                                      unsigned long long current_state_image_digest,
                                      int64_t next_kv_tokens,
                                      int64_t emitted_token_count,
                                      int32_t token_value,
                                      unsigned long long next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      int32_t next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                      unsigned long long *next_window_meta,
                                      unsigned long long *next_state_image_digest) {
  int32_t current_page_index = unpack_summary_leading_u16(current_window_meta);
  int32_t valid_page_count = unpack_summary_auxiliary_u16(current_window_meta);
  int32_t recent_token_count = unpack_summary_control_a_u16(current_window_meta);

  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT; ++index) {
    next_page_words[index] = current_page_words[index];
  }
  for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT; ++index) {
    next_recent_tokens[index] = current_recent_tokens[index];
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

  if (emitted_token_count > 0) {
    const int64_t new_page_anchor = next_kv_tokens - emitted_token_count;
    bool appended_to_existing_page = false;

    if (valid_page_count > 0) {
      const unsigned long long current_page_word = next_page_words[current_page_index];
      const int32_t current_page_anchor = unpack_summary_leading_u16(current_page_word);
      const int32_t current_page_fill = unpack_summary_auxiliary_u16(current_page_word);
      if (current_page_fill > 0 &&
          current_page_fill < MIZU_CUDA_CONTEXT_PAGE_CAPACITY &&
          current_page_anchor + current_page_fill == new_page_anchor) {
        next_page_words[current_page_index] = pack_kv_page_word(current_page_anchor,
          current_page_fill + static_cast<int32_t>(emitted_token_count), current_page_index, 2);
        appended_to_existing_page = true;
      }
    }

    if (!appended_to_existing_page) {
      if (valid_page_count < MIZU_CUDA_CONTEXT_PAGE_COUNT) {
        current_page_index = valid_page_count;
        valid_page_count += 1;
      } else {
        for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_PAGE_COUNT - 1; ++index) {
          next_page_words[index] = next_page_words[index + 1];
        }
        next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT - 1] = 0ULL;
        current_page_index = MIZU_CUDA_CONTEXT_PAGE_COUNT - 1;
      }
      next_page_words[current_page_index] = pack_kv_page_word(new_page_anchor, emitted_token_count,
        current_page_index, 2);
    }

    if (recent_token_count < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT) {
      next_recent_tokens[recent_token_count] = token_value;
      recent_token_count += 1;
    } else {
      for (int32_t index = 0; index < MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT - 1; ++index) {
        next_recent_tokens[index] = next_recent_tokens[index + 1];
      }
      next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT - 1] = token_value;
    }
  }

  if (next_window_meta != nullptr) {
    *next_window_meta = pack_context_summary(current_page_index, valid_page_count, recent_token_count, 0);
  }
  if (next_state_image_digest != nullptr) {
    *next_state_image_digest = digest_window_state(next_page_words, next_recent_tokens,
      current_state_image_digest ^ static_cast<unsigned long long>(next_kv_tokens) ^
      static_cast<unsigned long long>(static_cast<uint32_t>(token_value)));
  }
}

__global__ void mizu_prefill_kernel(int64_t staged_tokens,
                                    const int32_t *token_values,
                                    int64_t token_count,
                                    const int8_t *modal_bytes,
                                    int64_t modal_byte_count,
                                    int32_t staged_modal_count,
                                    int64_t *consumed_token_count,
                                    unsigned long long *tensor_seed) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t resolved_count = staged_tokens > 0 ? staged_tokens : 0;
    unsigned long long seed = static_cast<unsigned long long>(staged_tokens);
    int64_t index = 0;

    if (resolved_count == 0 && staged_modal_count > 0) resolved_count = 1;
    if (token_values != nullptr && token_count > 0) {
      for (index = 0; index < token_count; ++index) {
        seed = mix_u64(seed ^ static_cast<unsigned long long>(static_cast<uint32_t>(token_values[index])));
      }
    }
    if (modal_bytes != nullptr && modal_byte_count > 0) {
      for (index = 0; index < modal_byte_count; ++index) {
        seed = mix_u64(seed ^ static_cast<unsigned long long>(static_cast<uint8_t>(modal_bytes[index])));
      }
    }
    seed = mix_u64(seed ^ (static_cast<unsigned long long>(staged_modal_count) << 33));
    *consumed_token_count = resolved_count;
    *tensor_seed = seed;
  }
}

__global__ void mizu_projector_kernel(int64_t payload_hash,
                                      int64_t modal_byte_count,
                                      int32_t placeholder_count,
                                      int64_t *embedding_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long seed = mix_u64(static_cast<unsigned long long>(payload_hash) ^
                                      (static_cast<unsigned long long>(modal_byte_count) << 1));
    int64_t resolved_count = placeholder_count > 0 ? static_cast<int64_t>(placeholder_count) : 1;
    if (modal_byte_count > 0) resolved_count += static_cast<int64_t>(seed % 2ULL);
    *embedding_count = resolved_count;
  }
}

__global__ void mizu_decode_kernel(int64_t payload_hash,
                                   int64_t kv_before,
                                   int64_t token_budget,
                                   int64_t *emitted_token_count,
                                   int32_t *token_value,
                                   int32_t *stop_reason) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long seed = static_cast<unsigned long long>(payload_hash);
    seed ^= static_cast<unsigned long long>(kv_before) * 0x9e3779b97f4a7c15ULL;
    seed ^= static_cast<unsigned long long>(token_budget) * 0xbf58476d1ce4e5b9ULL;
    seed = mix_u64(seed);

    *emitted_token_count = token_budget > 0 ? 1 : 0;
    *token_value = 1 + static_cast<int32_t>(seed % 4095ULL);
    *stop_reason = MIZU_STOP_REASON_NONE;
  }
}

inline void copy_device_name(const char *source_name, char *device_name, size_t device_name_capacity) {
  if (device_name == nullptr || device_name_capacity == 0) return;
  if (source_name == nullptr) {
    device_name[0] = '\0';
    return;
  }

  strncpy(device_name, source_name, device_name_capacity - 1);
  device_name[device_name_capacity - 1] = '\0';
}

inline void stamp_workspace_buffer(void *workspace_buffer,
                                   int64_t workspace_bytes,
                                   unsigned long long seed,
                                   unsigned char stage_tag) {
  auto *bytes = static_cast<unsigned char *>(workspace_buffer);
  if (bytes == nullptr || workspace_bytes <= 0) return;

  seed = mix_u64_host(seed ^ static_cast<unsigned long long>(stage_tag));
  for (int64_t index = 0; index < workspace_bytes && index < 32; ++index) {
    bytes[index] = static_cast<unsigned char>((seed >> ((index % 8) * 8)) & 0xffULL);
    seed = mix_u64_host(seed + static_cast<unsigned long long>(index) +
                        static_cast<unsigned long long>(stage_tag));
  }
}

inline void fill_prefill_context_bytes(unsigned long long seed,
                                       unsigned long long artifact_hash,
                                       const int32_t *token_values,
                                       int64_t token_count,
                                       int64_t modal_byte_count,
                                       int32_t staged_modal_count,
                                       int64_t consumed_token_count,
                                       int8_t *context_bytes,
                                       int32_t context_capacity,
                                       int32_t *context_byte_count) {
  auto *bytes = reinterpret_cast<unsigned char *>(context_bytes);
  uint32_t checksum = 0U;
  unsigned long long state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES] = {};
  unsigned long long summary_word = 0ULL;
  unsigned long long page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT] = {};
  int32_t recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT] = {};
  unsigned long long window_meta = 0ULL;
  unsigned long long state_image_digest = 0ULL;
  int32_t stored_count = 0;

  if (context_byte_count == nullptr) return;

  *context_byte_count = 0;
  if (context_bytes == nullptr || context_capacity <= 0) return;

  stored_count = context_capacity < MIZU_CUDA_CONTEXT_TOTAL_BYTES ? context_capacity : MIZU_CUDA_CONTEXT_TOTAL_BYTES;
  memset(bytes, 0, static_cast<size_t>(stored_count));

  if (stored_count >= 1) bytes[0] = MIZU_CUDA_CONTEXT_MAGIC_0;
  if (stored_count >= 2) bytes[1] = MIZU_CUDA_CONTEXT_MAGIC_1;
  if (stored_count >= 3) bytes[2] = MIZU_CUDA_CONTEXT_MAGIC_2;
  if (stored_count >= 4) bytes[3] = MIZU_CUDA_CONTEXT_MAGIC_3;
  if (stored_count >= 5) bytes[4] = MIZU_CUDA_CONTEXT_VERSION;
  if (stored_count >= 6) bytes[5] = MIZU_CUDA_CONTEXT_KIND_PREFILL;
  if (stored_count >= 8) {
    bytes[6] = static_cast<unsigned char>(stored_count & 0xff);
    bytes[7] = static_cast<unsigned char>((stored_count >> 8) & 0xff);
  }
  build_prefill_state_block(seed, artifact_hash, token_count, modal_byte_count, staged_modal_count,
                            consumed_token_count, state_lanes, &summary_word);
  build_prefill_window_block(token_values, token_count, consumed_token_count, page_words, recent_tokens,
                             &window_meta, &state_image_digest);
  write_context_state_block(bytes, stored_count, state_lanes, artifact_hash, summary_word);
  write_context_window_block(bytes, stored_count, page_words, recent_tokens, window_meta, state_image_digest);

  checksum = compute_context_checksum(bytes, stored_count);
  if (stored_count > 8) {
    memcpy(bytes + 8, &checksum, static_cast<size_t>(stored_count - 8 < 4 ? stored_count - 8 : 4));
  }

  *context_byte_count = stored_count;
}

inline void fill_decode_context_bytes(unsigned long long seed,
                                      unsigned long long artifact_hash,
                                      const unsigned long long next_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES],
                                      unsigned long long summary_word,
                                      const unsigned long long next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT],
                                      const int32_t next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT],
                                      unsigned long long next_window_meta,
                                      unsigned long long next_state_image_digest,
                                      int8_t *context_bytes,
                                      int32_t context_capacity,
                                      int32_t *context_byte_count) {
  auto *bytes = reinterpret_cast<unsigned char *>(context_bytes);
  uint32_t checksum = 0U;
  int32_t stored_count = 0;
  (void)seed;

  if (context_byte_count == nullptr) return;

  *context_byte_count = 0;
  if (context_bytes == nullptr || context_capacity <= 0) return;

  stored_count = context_capacity < MIZU_CUDA_CONTEXT_TOTAL_BYTES ? context_capacity : MIZU_CUDA_CONTEXT_TOTAL_BYTES;
  memset(bytes, 0, static_cast<size_t>(stored_count));

  if (stored_count >= 1) bytes[0] = MIZU_CUDA_CONTEXT_MAGIC_0;
  if (stored_count >= 2) bytes[1] = MIZU_CUDA_CONTEXT_MAGIC_1;
  if (stored_count >= 3) bytes[2] = MIZU_CUDA_CONTEXT_MAGIC_2;
  if (stored_count >= 4) bytes[3] = MIZU_CUDA_CONTEXT_MAGIC_3;
  if (stored_count >= 5) bytes[4] = MIZU_CUDA_CONTEXT_VERSION;
  if (stored_count >= 6) bytes[5] = MIZU_CUDA_CONTEXT_KIND_DECODE;
  if (stored_count >= 8) {
    bytes[6] = static_cast<unsigned char>(stored_count & 0xff);
    bytes[7] = static_cast<unsigned char>((stored_count >> 8) & 0xff);
  }
  write_context_state_block(bytes, stored_count, next_state_lanes, artifact_hash, summary_word);
  write_context_window_block(bytes, stored_count, next_page_words, next_recent_tokens, next_window_meta,
                             next_state_image_digest);

  checksum = compute_context_checksum(bytes, stored_count);
  if (stored_count > 8) {
    memcpy(bytes + 8, &checksum, static_cast<size_t>(stored_count - 8 < 4 ? stored_count - 8 : 4));
  }

  *context_byte_count = stored_count;
}

}  // namespace

extern "C" void mizu_cuda_bridge_get_device_info(int32_t *device_count,
                                                 int64_t *total_memory_bytes,
                                                 int32_t *compute_major,
                                                 int32_t *compute_minor,
                                                 int32_t *multiprocessor_count,
                                                 char *device_name,
                                                 size_t device_name_capacity,
                                                 int32_t *status_code) {
  if (device_count == nullptr || total_memory_bytes == nullptr || compute_major == nullptr ||
      compute_minor == nullptr || multiprocessor_count == nullptr || status_code == nullptr) {
    return;
  }

  *device_count = 0;
  *total_memory_bytes = 0;
  *compute_major = 0;
  *compute_minor = 0;
  *multiprocessor_count = 0;
  copy_device_name("", device_name, device_name_capacity);

  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count <= 0) {
    *status_code = MIZU_STATUS_OK;
    copy_device_name("cuda_unavailable", device_name, device_name_capacity);
    return;
  }

  cudaDeviceProp properties;
  status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess) {
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  *device_count = static_cast<int32_t>(count);
  *total_memory_bytes = static_cast<int64_t>(properties.totalGlobalMem);
  *compute_major = static_cast<int32_t>(properties.major);
  *compute_minor = static_cast<int32_t>(properties.minor);
  *multiprocessor_count = static_cast<int32_t>(properties.multiProcessorCount);
  copy_device_name(properties.name, device_name, device_name_capacity);
  *status_code = MIZU_STATUS_OK;
}

extern "C" void mizu_cuda_bridge_prefill(int64_t payload_hash,
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
  int64_t *managed_consumed_token_count = nullptr;
  int32_t *managed_token_values = nullptr;
  int8_t *managed_modal_bytes = nullptr;
  unsigned long long *managed_tensor_seed = nullptr;
  unsigned long long state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES] = {};
  unsigned long long summary_word = 0ULL;
  unsigned long long workspace_seed = 0;
  size_t token_bytes = 0;
  size_t modal_bytes_size = 0;

  if (consumed_token_count == nullptr || status_code == nullptr) return;

  *consumed_token_count = 0;
  *status_code = MIZU_STATUS_OK;

  cudaError_t status = cudaMallocManaged(&managed_consumed_token_count, sizeof(*managed_consumed_token_count));
  if (status != cudaSuccess) {
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  if (token_values != nullptr && token_count > 0) {
    token_bytes = static_cast<size_t>(token_count) * sizeof(*managed_token_values);
    status = cudaMallocManaged(&managed_token_values, token_bytes);
    if (status != cudaSuccess) {
      cudaFree(managed_consumed_token_count);
      *status_code = MIZU_STATUS_EXECUTION_ERROR;
      return;
    }
    memcpy(managed_token_values, token_values, token_bytes);
  }

  if (modal_bytes != nullptr && modal_byte_count > 0) {
    modal_bytes_size = static_cast<size_t>(modal_byte_count) * sizeof(*managed_modal_bytes);
    status = cudaMallocManaged(&managed_modal_bytes, modal_bytes_size);
    if (status != cudaSuccess) {
      cudaFree(managed_consumed_token_count);
      cudaFree(managed_token_values);
      *status_code = MIZU_STATUS_EXECUTION_ERROR;
      return;
    }
    memcpy(managed_modal_bytes, modal_bytes, modal_bytes_size);
  }

  status = cudaMallocManaged(&managed_tensor_seed, sizeof(*managed_tensor_seed));
  if (status != cudaSuccess) {
    cudaFree(managed_consumed_token_count);
    cudaFree(managed_token_values);
    cudaFree(managed_modal_bytes);
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  *managed_consumed_token_count = 0;
  *managed_tensor_seed = 0ULL;
  mizu_prefill_kernel<<<1, 1>>>(token_count, managed_token_values, token_count, managed_modal_bytes,
                                modal_byte_count, staged_modal_count, managed_consumed_token_count,
                                managed_tensor_seed);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    *consumed_token_count = *managed_consumed_token_count;
    workspace_seed = mix_u64_host(static_cast<unsigned long long>(payload_hash) ^ *managed_tensor_seed);
    build_prefill_state_block(workspace_seed, static_cast<unsigned long long>(artifact_hash), token_count,
                              modal_byte_count, staged_modal_count, *consumed_token_count, state_lanes,
                              &summary_word);
    fill_prefill_context_bytes(workspace_seed, static_cast<unsigned long long>(artifact_hash), token_values,
                               token_count, modal_byte_count, staged_modal_count, *consumed_token_count,
                               context_bytes, context_capacity, context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, state_lanes[0] ^ state_lanes[3] ^ summary_word, 3U);
  }
  *status_code = (status == cudaSuccess) ? MIZU_STATUS_OK : MIZU_STATUS_EXECUTION_ERROR;

  cudaFree(managed_consumed_token_count);
  cudaFree(managed_token_values);
  cudaFree(managed_modal_bytes);
  cudaFree(managed_tensor_seed);
}

extern "C" void mizu_cuda_bridge_projector(int64_t payload_hash,
                                           int64_t modal_byte_count,
                                           int32_t placeholder_count,
                                           void *workspace_buffer,
                                           int64_t workspace_bytes,
                                           int64_t *embedding_count,
                                           int32_t *status_code) {
  int64_t *managed_embedding_count = nullptr;
  unsigned long long workspace_seed = 0;

  if (embedding_count == nullptr || status_code == nullptr) return;

  *embedding_count = 0;
  *status_code = MIZU_STATUS_OK;

  cudaError_t status = cudaMallocManaged(&managed_embedding_count, sizeof(*managed_embedding_count));
  if (status != cudaSuccess) {
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  *managed_embedding_count = 0;
  mizu_projector_kernel<<<1, 1>>>(payload_hash, modal_byte_count, placeholder_count, managed_embedding_count);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    *embedding_count = *managed_embedding_count;
    workspace_seed = mix_u64_host(static_cast<unsigned long long>(payload_hash) ^
                                  (static_cast<unsigned long long>(modal_byte_count) << 1));
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, workspace_seed, 2U);
  }
  *status_code = (status == cudaSuccess) ? MIZU_STATUS_OK : MIZU_STATUS_EXECUTION_ERROR;

  cudaFree(managed_embedding_count);
}

extern "C" void mizu_cuda_bridge_decode(int64_t payload_hash,
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
  int64_t *managed_emitted_token_count = nullptr;
  int32_t *managed_token_value = nullptr;
  int32_t *managed_stop_reason = nullptr;
  unsigned long long current_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES] = {};
  unsigned long long next_state_lanes[MIZU_CUDA_CONTEXT_STATE_LANES] = {};
  unsigned long long current_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT] = {};
  unsigned long long next_page_words[MIZU_CUDA_CONTEXT_PAGE_COUNT] = {};
  unsigned long long summary_word = 0ULL;
  unsigned long long current_window_meta = 0ULL;
  unsigned long long next_window_meta = 0ULL;
  unsigned long long current_state_image_digest = 0ULL;
  unsigned long long next_state_image_digest = 0ULL;
  unsigned long long decode_seed = 0;
  int32_t current_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT] = {};
  int32_t next_recent_tokens[MIZU_CUDA_CONTEXT_RECENT_TOKEN_COUNT] = {};

  if (emitted_token_count == nullptr || token_value == nullptr || stop_reason == nullptr ||
      status_code == nullptr) {
    return;
  }

  *emitted_token_count = 0;
  *token_value = 0;
  *stop_reason = MIZU_STOP_REASON_NONE;

  cudaError_t status = cudaMallocManaged(&managed_emitted_token_count, sizeof(*managed_emitted_token_count));
  if (status != cudaSuccess) {
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  status = cudaMallocManaged(&managed_token_value, sizeof(*managed_token_value));
  if (status != cudaSuccess) {
    cudaFree(managed_emitted_token_count);
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  status = cudaMallocManaged(&managed_stop_reason, sizeof(*managed_stop_reason));
  if (status != cudaSuccess) {
    cudaFree(managed_emitted_token_count);
    cudaFree(managed_token_value);
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  *managed_emitted_token_count = 0;
  *managed_token_value = 0;
  *managed_stop_reason = MIZU_STOP_REASON_NONE;

  extract_context_state_block(context_bytes, context_byte_count, current_state_lanes, nullptr, &summary_word);
  extract_context_window_block(context_bytes, context_byte_count, current_page_words, current_recent_tokens,
                               &current_window_meta, &current_state_image_digest);
  decode_seed = static_cast<unsigned long long>(payload_hash);
  decode_seed = mix_u64_host(decode_seed ^ current_state_lanes[0] ^ current_state_lanes[1]);
  decode_seed = mix_u64_host(decode_seed ^ current_state_lanes[2] ^ current_state_lanes[3] ^ summary_word);
  decode_seed = mix_u64_host(decode_seed ^ current_window_meta ^ current_state_image_digest);

  mizu_decode_kernel<<<1, 1>>>(static_cast<int64_t>(decode_seed), kv_before, token_budget, managed_emitted_token_count,
                               managed_token_value, managed_stop_reason);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    *emitted_token_count = *managed_emitted_token_count;
    *token_value = *managed_token_value;
    *stop_reason = *managed_stop_reason;
    build_decode_state_block(current_state_lanes, static_cast<unsigned long long>(artifact_hash), kv_before,
                             token_budget, *emitted_token_count, *token_value, *stop_reason, next_state_lanes,
                             &summary_word);
    build_decode_window_block(current_page_words, current_recent_tokens, current_window_meta,
                              current_state_image_digest, unpack_state_kv_tokens(next_state_lanes[2]),
                              *emitted_token_count, *token_value, next_page_words, next_recent_tokens,
                              &next_window_meta, &next_state_image_digest);
    fill_decode_context_bytes(decode_seed ^ static_cast<unsigned long long>(*token_value),
                              static_cast<unsigned long long>(artifact_hash), next_state_lanes, summary_word,
                              next_page_words, next_recent_tokens, next_window_meta, next_state_image_digest,
                              updated_context_bytes, updated_context_capacity, updated_context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes,
                           next_state_lanes[1] ^ next_state_lanes[3] ^ next_state_image_digest ^ summary_word,
                           4U);
  }
  *status_code = (status == cudaSuccess) ? MIZU_STATUS_OK : MIZU_STATUS_EXECUTION_ERROR;

  cudaFree(managed_emitted_token_count);
  cudaFree(managed_token_value);
  cudaFree(managed_stop_reason);
}
