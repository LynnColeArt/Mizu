#include "mizu.h"
#include "cuda_bridge.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>
#include <string.h>

namespace {

__device__ __forceinline__ unsigned long long mix_u64(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void mizu_prefill_kernel(int64_t staged_tokens,
                                    int32_t staged_modal_count,
                                    int64_t *consumed_token_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t resolved_count = staged_tokens > 0 ? staged_tokens : 0;
    if (resolved_count == 0 && staged_modal_count > 0) resolved_count = 1;
    *consumed_token_count = resolved_count;
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
                                         int64_t staged_tokens,
                                         int32_t staged_modal_count,
                                         int64_t *consumed_token_count,
                                         int32_t *status_code) {
  int64_t *managed_consumed_token_count = nullptr;

  (void)payload_hash;

  if (consumed_token_count == nullptr || status_code == nullptr) return;

  *consumed_token_count = 0;
  *status_code = MIZU_STATUS_OK;

  cudaError_t status = cudaMallocManaged(&managed_consumed_token_count, sizeof(*managed_consumed_token_count));
  if (status != cudaSuccess) {
    *status_code = MIZU_STATUS_EXECUTION_ERROR;
    return;
  }

  *managed_consumed_token_count = 0;
  mizu_prefill_kernel<<<1, 1>>>(staged_tokens, staged_modal_count, managed_consumed_token_count);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) *consumed_token_count = *managed_consumed_token_count;
  *status_code = (status == cudaSuccess) ? MIZU_STATUS_OK : MIZU_STATUS_EXECUTION_ERROR;

  cudaFree(managed_consumed_token_count);
}

extern "C" void mizu_cuda_bridge_decode(int64_t payload_hash,
                                        int64_t kv_before,
                                        int64_t token_budget,
                                        int64_t *emitted_token_count,
                                        int32_t *token_value,
                                        int32_t *stop_reason,
                                        int32_t *status_code) {
  int64_t *managed_emitted_token_count = nullptr;
  int32_t *managed_token_value = nullptr;
  int32_t *managed_stop_reason = nullptr;

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

  mizu_decode_kernel<<<1, 1>>>(payload_hash, kv_before, token_budget, managed_emitted_token_count,
                               managed_token_value, managed_stop_reason);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    *emitted_token_count = *managed_emitted_token_count;
    *token_value = *managed_token_value;
    *stop_reason = *managed_stop_reason;
  }
  *status_code = (status == cudaSuccess) ? MIZU_STATUS_OK : MIZU_STATUS_EXECUTION_ERROR;

  cudaFree(managed_emitted_token_count);
  cudaFree(managed_token_value);
  cudaFree(managed_stop_reason);
}
