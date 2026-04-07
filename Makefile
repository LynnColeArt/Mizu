FC := gfortran
CC ?= gcc
CXX ?= g++
NVCC ?= nvcc

FFLAGS ?= -std=f2018 -Wall -Wextra
CFLAGS ?= -std=c11 -Wall -Wextra
CXXFLAGS ?= -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -std=c++17 -Iinclude -Isrc/backends/cuda

HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)

BUILD_DIR := build
TEST_DIR := $(BUILD_DIR)/tests
CUDA_BRIDGE_OBJ := $(BUILD_DIR)/cuda_bridge.o

COMMON_F90 := \
	src/common/mod_kinds.f90 \
	src/common/mod_status.f90 \
	src/common/mod_types.f90 \
	src/common/mod_errors.f90

MODEL_F90 := \
	src/model/mod_model_manifest.f90 \
	src/model/mod_model_loader.f90

CACHE_F90 := \
	src/cache/mod_cache_keys.f90 \
	src/cache/mod_cache_store.f90

RUNTIME_F90 := \
	src/runtime/mod_request.f90 \
	src/runtime/mod_workspace.f90 \
	src/runtime/mod_scheduler.f90 \
	src/runtime/mod_runtime.f90 \
	src/runtime/mod_session.f90 \
	src/runtime/mod_optimization_store.f90

BACKEND_F90 := \
	src/backends/mod_backend_contract.f90 \
	src/backends/mod_backend_probe_support.f90 \
	src/backends/apple/mod_apple_capability.f90 \
	src/backends/cuda/mod_cuda_bridge.f90 \
	src/backends/cuda/mod_cuda_capability.f90 \
	src/backends/cuda/mod_cuda_planner.f90 \
	src/backends/cuda/mod_cuda_executor.f90 \
	src/backends/mod_backend_registry.f90

CAPI_F90 := src/c_api/mod_c_api.f90

UNIT_BINS := \
	$(TEST_DIR)/test_model_manifest_loader \
	$(TEST_DIR)/test_cache_keys \
	$(TEST_DIR)/test_cache_store \
	$(TEST_DIR)/test_optimization_store \
	$(TEST_DIR)/test_backend_registry \
	$(TEST_DIR)/test_runtime_workspace \
	$(TEST_DIR)/test_session_staging \
	$(TEST_DIR)/test_cuda_planner \
	$(TEST_DIR)/test_cuda_executor

CONTRACT_SMOKES := \
	$(TEST_DIR)/test_header_c_smoke.o \
	$(TEST_DIR)/test_header_cpp_smoke.o

CONTRACT_BINS := \
	$(TEST_DIR)/test_opaque_handles \
	$(TEST_DIR)/test_cuda_artifacts \
	$(TEST_DIR)/test_stage_reports

.PHONY: all test unit-tests contract-tests contract-smokes clean

all: test

test: unit-tests contract-tests

unit-tests: $(UNIT_BINS)
	@for test_bin in $(UNIT_BINS); do $$test_bin; done

contract-tests: contract-smokes $(CONTRACT_BINS)
	@for test_bin in $(CONTRACT_BINS); do $$test_bin; done

contract-smokes: $(CONTRACT_SMOKES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_DIR):
	mkdir -p $(TEST_DIR)

ifeq ($(HAVE_NVCC),1)
CUDA_BRIDGE_SRC := src/backends/cuda/cuda_bridge.cu
CUDA_BRIDGE_LINK_LIBS := -lcudart -lstdc++

$(CUDA_BRIDGE_OBJ): $(CUDA_BRIDGE_SRC) src/backends/cuda/cuda_bridge.h include/mizu.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
CUDA_BRIDGE_SRC := src/backends/cuda/cuda_bridge_stub.c
CUDA_BRIDGE_LINK_LIBS :=

$(CUDA_BRIDGE_OBJ): $(CUDA_BRIDGE_SRC) src/backends/cuda/cuda_bridge.h include/mizu.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -Isrc/backends/cuda -c $< -o $@
endif

$(TEST_DIR)/test_model_manifest_loader: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/loader_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/loader_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/model/mod_model_manifest.f90 \
		src/model/mod_model_loader.f90 \
		tests/unit/test_model_manifest_loader.f90

$(TEST_DIR)/test_cache_keys: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/cache_keys_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/cache_keys_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/model/mod_model_manifest.f90 \
		src/model/mod_model_loader.f90 \
		src/cache/mod_cache_keys.f90 \
		tests/unit/test_cache_keys.f90

$(TEST_DIR)/test_cache_store: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/cache_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/cache_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/model/mod_model_manifest.f90 \
		src/cache/mod_cache_keys.f90 \
		src/cache/mod_cache_store.f90 \
		tests/unit/test_cache_store.f90

$(TEST_DIR)/test_optimization_store: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/optimization_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/optimization_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/model/mod_model_manifest.f90 \
		src/cache/mod_cache_keys.f90 \
		src/runtime/mod_optimization_store.f90 \
		tests/unit/test_optimization_store.f90

$(TEST_DIR)/test_backend_registry: $(TEST_DIR) $(CUDA_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/backend_registry_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/backend_registry_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/runtime/mod_workspace.f90 \
		src/runtime/mod_runtime.f90 \
		src/backends/mod_backend_contract.f90 \
		src/backends/mod_backend_probe_support.f90 \
		src/backends/apple/mod_apple_capability.f90 \
		src/backends/cuda/mod_cuda_bridge.f90 \
		src/backends/cuda/mod_cuda_capability.f90 \
		src/backends/mod_backend_registry.f90 \
		tests/unit/test_backend_registry.f90 \
		$(CUDA_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_runtime_workspace: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/runtime_workspace_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/runtime_workspace_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/runtime/mod_workspace.f90 \
		src/runtime/mod_runtime.f90 \
		tests/unit/test_runtime_workspace.f90

$(TEST_DIR)/test_session_staging: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/session_staging_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/session_staging_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/runtime/mod_session.f90 \
		tests/unit/test_session_staging.f90

$(TEST_DIR)/test_cuda_planner: $(TEST_DIR)
	mkdir -p $(TEST_DIR)/cuda_planner_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/cuda_planner_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/backends/mod_backend_contract.f90 \
		src/backends/cuda/mod_cuda_planner.f90 \
		tests/unit/test_cuda_planner.f90

$(TEST_DIR)/test_cuda_executor: $(TEST_DIR) $(CUDA_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/cuda_executor_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/cuda_executor_mods -o $@ \
		src/common/mod_kinds.f90 \
		src/common/mod_status.f90 \
		src/common/mod_types.f90 \
		src/runtime/mod_workspace.f90 \
		src/model/mod_model_manifest.f90 \
		src/backends/cuda/mod_cuda_bridge.f90 \
		src/backends/cuda/mod_cuda_executor.f90 \
		tests/unit/test_cuda_executor.f90 \
		$(CUDA_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_header_c_smoke.o: tests/contract/test_header_c_smoke.c | $(TEST_DIR)
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(TEST_DIR)/test_header_cpp_smoke.o: tests/contract/test_header_cpp_smoke.cpp | $(TEST_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

$(TEST_DIR)/test_opaque_handles: $(TEST_DIR)
	$(CC) $(CFLAGS) -Iinclude tests/contract/test_opaque_handles.c -o $@

$(TEST_DIR)/test_cuda_artifacts.o: tests/contract/test_cuda_artifacts.c | $(TEST_DIR)
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(TEST_DIR)/test_cuda_artifacts: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_cuda_artifacts.o $(CUDA_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/cuda_contract_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/cuda_contract_mods -I $(TEST_DIR)/cuda_contract_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_cuda_artifacts.o \
		$(CUDA_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_LINK_LIBS)

$(TEST_DIR)/test_stage_reports.o: tests/contract/test_stage_reports.c | $(TEST_DIR)
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(TEST_DIR)/test_stage_reports: $(COMMON_F90) $(MODEL_F90) $(CACHE_F90) $(RUNTIME_F90) $(BACKEND_F90) \
	$(CAPI_F90) $(TEST_DIR)/test_stage_reports.o $(CUDA_BRIDGE_OBJ)
	mkdir -p $(TEST_DIR)/contract_mods
	$(FC) $(FFLAGS) -J $(TEST_DIR)/contract_mods -I $(TEST_DIR)/contract_mods -o $@ \
		$(COMMON_F90) \
		$(MODEL_F90) \
		$(CACHE_F90) \
		$(RUNTIME_F90) \
		$(BACKEND_F90) \
		$(CAPI_F90) \
		$(TEST_DIR)/test_stage_reports.o \
		$(CUDA_BRIDGE_OBJ) \
		$(CUDA_BRIDGE_LINK_LIBS)

clean:
	rm -rf $(BUILD_DIR) ./*.mod
