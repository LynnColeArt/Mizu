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

inline unsigned long long mix_u64_host(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
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
                                       int64_t token_count,
                                       int64_t modal_byte_count,
                                       int32_t staged_modal_count,
                                       int64_t consumed_token_count,
                                       int8_t *context_bytes,
                                       int32_t context_capacity,
                                       int32_t *context_byte_count) {
  auto *bytes = reinterpret_cast<unsigned char *>(context_bytes);
  unsigned long long seed_copy = seed;
  int32_t stored_count = 0;

  if (context_byte_count == nullptr) return;

  *context_byte_count = 0;
  if (context_bytes == nullptr || context_capacity <= 0) return;

  stored_count = context_capacity < 48 ? context_capacity : 48;
  memset(bytes, 0, static_cast<size_t>(stored_count));

  if (stored_count > 0) {
    memcpy(bytes, &seed, static_cast<size_t>(stored_count < 8 ? stored_count : 8));
  }
  if (stored_count > 8) {
    memcpy(bytes + 8, &token_count, static_cast<size_t>(stored_count - 8 < 8 ? stored_count - 8 : 8));
  }
  if (stored_count > 16) {
    memcpy(bytes + 16, &modal_byte_count, static_cast<size_t>(stored_count - 16 < 8 ? stored_count - 16 : 8));
  }
  if (stored_count > 24) {
    memcpy(bytes + 24, &staged_modal_count, static_cast<size_t>(stored_count - 24 < 4 ? stored_count - 24 : 4));
  }
  if (stored_count > 28) {
    memcpy(bytes + 28, &consumed_token_count, static_cast<size_t>(stored_count - 28 < 8 ? stored_count - 28 : 8));
  }

  for (int32_t index = 0; index < stored_count; ++index) {
    seed_copy = mix_u64_host(seed_copy ^
                             (static_cast<unsigned long long>(token_count) << 1) ^
                             (static_cast<unsigned long long>(modal_byte_count) << 9) ^
                             (static_cast<unsigned long long>(staged_modal_count) << 17) ^
                             (static_cast<unsigned long long>(consumed_token_count) << 25) ^
                             static_cast<unsigned long long>(index));
    bytes[index] ^= static_cast<unsigned char>(seed_copy & 0xffULL);
  }

  *context_byte_count = stored_count;
}

inline void fill_decode_context_bytes(unsigned long long seed,
                                      int64_t kv_after,
                                      int64_t emitted_token_count,
                                      int32_t token_value,
                                      int32_t stop_reason,
                                      int8_t *context_bytes,
                               int32_t context_capacity,
                                      int32_t *context_byte_count) {
  auto *bytes = reinterpret_cast<unsigned char *>(context_bytes);
  int32_t stored_count = 0;
  unsigned long long seed_copy = seed;

  if (context_byte_count == nullptr) return;

  *context_byte_count = 0;
  if (context_bytes == nullptr || context_capacity <= 0) return;

  stored_count = context_capacity < 48 ? context_capacity : 48;
  memset(bytes, 0, static_cast<size_t>(stored_count));

  if (stored_count > 0) {
    memcpy(bytes, &seed, static_cast<size_t>(stored_count < 8 ? stored_count : 8));
  }
  if (stored_count > 8) {
    memcpy(bytes + 8, &kv_after, static_cast<size_t>(stored_count - 8 < 8 ? stored_count - 8 : 8));
  }
  if (stored_count > 16) {
    memcpy(bytes + 16, &emitted_token_count, static_cast<size_t>(stored_count - 16 < 8 ? stored_count - 16 : 8));
  }
  if (stored_count > 24) {
    memcpy(bytes + 24, &token_value, static_cast<size_t>(stored_count - 24 < 4 ? stored_count - 24 : 4));
  }
  if (stored_count > 28) {
    memcpy(bytes + 28, &stop_reason, static_cast<size_t>(stored_count - 28 < 4 ? stored_count - 28 : 4));
  }

  for (int32_t index = 0; index < stored_count; ++index) {
    seed_copy = mix_u64_host(seed_copy ^
                             (static_cast<unsigned long long>(kv_after) << 1) ^
                             (static_cast<unsigned long long>(emitted_token_count) << 9) ^
                             (static_cast<unsigned long long>(token_value) << 17) ^
                             (static_cast<unsigned long long>(stop_reason) << 25) ^
                             static_cast<unsigned long long>(index));
    bytes[index] ^= static_cast<unsigned char>(seed_copy & 0xffULL);
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
    fill_prefill_context_bytes(workspace_seed, token_count, modal_byte_count, staged_modal_count,
                               *consumed_token_count, context_bytes, context_capacity, context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes, workspace_seed, 3U);
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
  unsigned long long decode_seed = 0;

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

  decode_seed = static_cast<unsigned long long>(payload_hash);
  if (context_bytes != nullptr && context_byte_count > 0) {
    for (int32_t index = 0; index < context_byte_count; ++index) {
      decode_seed = mix_u64_host(decode_seed ^
                                 static_cast<unsigned long long>(static_cast<unsigned char>(context_bytes[index])));
    }
  }

  mizu_decode_kernel<<<1, 1>>>(static_cast<int64_t>(decode_seed), kv_before, token_budget, managed_emitted_token_count,
                               managed_token_value, managed_stop_reason);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    *emitted_token_count = *managed_emitted_token_count;
    *token_value = *managed_token_value;
    *stop_reason = *managed_stop_reason;
    fill_decode_context_bytes(decode_seed ^ static_cast<unsigned long long>(*token_value),
                              kv_before + *emitted_token_count, *emitted_token_count, *token_value,
                              *stop_reason, updated_context_bytes, updated_context_capacity,
                              updated_context_byte_count);
    stamp_workspace_buffer(workspace_buffer, workspace_bytes,
                           mix_u64_host(decode_seed ^
                                        static_cast<unsigned long long>(kv_before)),
                           4U);
  }
  *status_code = (status == cudaSuccess) ? MIZU_STATUS_OK : MIZU_STATUS_EXECUTION_ERROR;

  cudaFree(managed_emitted_token_count);
  cudaFree(managed_token_value);
  cudaFree(managed_stop_reason);
}
