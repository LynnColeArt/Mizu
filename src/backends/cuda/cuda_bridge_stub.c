#include "mizu.h"
#include "cuda_bridge.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

static uint64_t mix_u64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
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
                              int64_t staged_tokens,
                              int32_t staged_modal_count,
                              int64_t *consumed_token_count,
                              int32_t *status_code) {
    (void)payload_hash;

    if (consumed_token_count == NULL || status_code == NULL) return;

    *consumed_token_count = staged_tokens > 0 ? staged_tokens : 0;
    if (*consumed_token_count == 0 && staged_modal_count > 0) *consumed_token_count = 1;
    *status_code = MIZU_STATUS_OK;
}

void mizu_cuda_bridge_decode(int64_t payload_hash,
                             int64_t kv_before,
                             int64_t token_budget,
                             int64_t *emitted_token_count,
                             int32_t *token_value,
                             int32_t *stop_reason,
                             int32_t *status_code) {
    uint64_t seed;

    if (emitted_token_count == NULL || token_value == NULL || stop_reason == NULL ||
        status_code == NULL) {
        return;
    }

    seed = (uint64_t)payload_hash;
    seed ^= (uint64_t)kv_before * UINT64_C(0x9e3779b97f4a7c15);
    seed ^= (uint64_t)token_budget * UINT64_C(0xbf58476d1ce4e5b9);
    seed = mix_u64(seed);

    *emitted_token_count = token_budget > 0 ? 1 : 0;
    *token_value = 1 + (int32_t)(seed % UINT64_C(4095));
    *stop_reason = MIZU_STOP_REASON_NONE;
    *status_code = MIZU_STATUS_OK;
}
