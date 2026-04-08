module mod_c_api
  use iso_c_binding, only: c_ptr, c_null_ptr, c_associated, c_f_pointer, c_loc, &
                           c_size_t, c_int32_t, c_int64_t, c_char, c_float, &
                           c_null_char, c_sizeof
  use mod_kinds,     only: i8, i32, i64, c_i8, r32, MAX_NAME_LEN, MAX_PATH_LEN
  use mod_status,    only: MIZU_STATUS_OK, MIZU_STATUS_END_OF_SEQUENCE, &
                           MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_INVALID_STATE, &
                           MIZU_STATUS_BUFFER_TOO_SMALL, MIZU_STATUS_ABI_MISMATCH, &
                           MIZU_STATUS_BUSY, MIZU_STATUS_UNSUPPORTED_MODEL, &
                           MIZU_STATUS_UNSUPPORTED_MODALITY, MIZU_STATUS_NO_VALID_PLAN, &
                           MIZU_STATUS_SESSION_EVICTED
  use mod_types,     only: MIZU_ABI_VERSION, MIZU_OPTIMIZATION_MODE_DISABLED, &
                           MIZU_OPTIMIZATION_MODE_MEASURE_ONLY, &
                           MIZU_OPTIMIZATION_MODE_LEARN_AND_REUSE, &
                           MIZU_BACKEND_MASK_NONE, MIZU_BACKEND_MASK_APPLE_ANE, &
                           MIZU_BACKEND_MASK_APPLE_METAL, MIZU_BACKEND_MASK_CUDA, &
                           MIZU_MODEL_FLAG_NONE, MIZU_SESSION_FLAG_NONE, &
                           MIZU_ATTACH_FLAG_NONE, MIZU_MODEL_FEATURE_MULTIMODAL, &
                           MIZU_MODEL_FEATURE_PROJECTOR, MIZU_MODEL_FAMILY_UNKNOWN, &
                           MIZU_MODEL_FAMILY_QWEN3_5, MIZU_MODEL_FAMILY_GEMMA4, &
                           MIZU_STAGE_NONE, MIZU_STAGE_MODEL_LOAD, MIZU_STAGE_PROJECTOR, &
                           MIZU_STAGE_PREFILL, MIZU_STAGE_DECODE, MIZU_STAGE_PARK, &
                           MIZU_STAGE_RESUME, MIZU_SELECTION_MODE_NONE, &
                           MIZU_SELECTION_MODE_DIRECT, MIZU_SELECTION_MODE_EXPLORATORY, &
                           MIZU_SELECTION_MODE_REUSE, MIZU_COLD_STATE_UNKNOWN, &
                           MIZU_COLD_STATE_COLD, MIZU_COLD_STATE_WARM, &
                           MIZU_FALLBACK_REASON_NONE, MIZU_OUTPUT_KIND_TOKEN_IDS, &
                           MIZU_STOP_REASON_NONE, MIZU_STOP_REASON_TOKEN_BUDGET, &
                           MIZU_BACKEND_FAMILY_NONE, &
                           MIZU_BACKEND_FAMILY_APPLE, MIZU_BACKEND_FAMILY_CUDA, &
                           MIZU_EXEC_ROUTE_NONE, MIZU_EXEC_ROUTE_ANE, &
                           MIZU_EXEC_ROUTE_METAL, MIZU_EXEC_ROUTE_CUDA, &
                           MIZU_CACHE_FLAG_NONE, MIZU_CACHE_FLAG_WEIGHT_HIT, &
                           MIZU_CACHE_FLAG_PLAN_HIT, MIZU_CACHE_FLAG_SESSION_HIT, &
                           MIZU_CACHE_FLAG_MM_HIT, MIZU_CACHE_FLAG_WINNER_REUSED, &
                           MIZU_MODALITY_KIND_IMAGE, &
                           MIZU_DTYPE_U8, MIZU_DTYPE_BF16, runtime_handle, model_handle, &
                           session_handle, runtime_config, model_open_config, session_config, &
                           model_info, session_info, execution_report, runtime_state, &
                           model_state, session_state, MAX_RECENT_OUTPUT_TOKENS, MAX_LIVE_CONTEXT_BYTES
  use mod_runtime,   only: initialize_runtime_state, reset_runtime_state, &
                           initialize_model_state, reset_model_state, register_model, &
                           unregister_model, register_session, unregister_session, &
                           validate_runtime_destroy, validate_model_close, &
                           set_runtime_error
  use mod_workspace, only: reserve_workspace_bytes, release_workspace_bytes
  use mod_session,   only: initialize_session_state, reset_session_state, &
                           stage_tokens, stage_modal_input, clear_pending_inputs, &
                           complete_prefill, complete_decode, park_session_state, &
                           resume_session_state, evict_parked_session, &
                           build_session_info, validate_read_output, store_live_context_record, &
                           update_live_context_record, offload_live_context_record
  use mod_optimization_store, only: runtime_optimization_store, &
                                     initialize_runtime_optimization_store, &
                                     reset_runtime_optimization_store, &
                                     record_execution_sample, lookup_winner_candidate, &
                                     lookup_optimization_entry_stats, &
                                     load_runtime_optimization_store, &
                                     save_runtime_optimization_store
  use mod_backend_registry, only: runtime_backend_registry, initialize_runtime_backend_registry, &
                                  probe_runtime_backend_registry, apply_backend_registry_to_runtime
  use mod_backend_contract, only: plan_request, planner_result, initialize_plan_request, &
                                  planner_result_is_success, OP_FAMILY_NONE, OP_FAMILY_PROJECTOR, &
                                  OP_FAMILY_PREFILL, OP_FAMILY_DECODE
  use mod_model_manifest, only: model_manifest, populate_model_info_from_manifest, &
                                manifest_tensor_count, manifest_modality_count, &
                                hash_text64
  use mod_model_loader,   only: load_model_manifest_from_root
  use mod_apple_planner,  only: APPLE_ARTIFACT_PAYLOAD_LEN, plan_apple_stage, &
                                build_apple_artifact_payload_text
  use mod_apple_executor, only: execute_apple_projector, execute_apple_prefill, execute_apple_decode, &
                                apple_context_bytes_are_valid, extract_apple_context_lineage
  use mod_cuda_planner,   only: CUDA_ARTIFACT_PAYLOAD_LEN, plan_cuda_stage, &
                                build_cuda_artifact_payload_text
  use mod_cuda_executor,  only: execute_cuda_projector, execute_cuda_prefill, execute_cuda_decode, &
                                cuda_context_bytes_are_valid, extract_cuda_context_lineage
  use mod_cache_keys,     only: MAX_CACHE_KEY_LEN, plan_cache_key, weight_cache_key, &
                                session_cache_key, multimodal_cache_key, build_plan_cache_key, &
                                build_weight_cache_key, build_session_cache_key, &
                                build_multimodal_cache_key
  use mod_cache_store,    only: artifact_metadata_record, runtime_cache_bundle, &
                                initialize_runtime_cache_bundle, reset_runtime_cache_bundle, &
                                touch_weight_cache_key, touch_plan_cache_key, &
                                touch_session_cache_key, touch_multimodal_cache_key, &
                                record_weight_artifact_metadata, record_plan_artifact_metadata, &
                                record_session_artifact_metadata, record_multimodal_artifact_metadata, &
                                lookup_session_artifact_metadata, load_runtime_cache_bundle, &
                                save_runtime_cache_bundle

  implicit none

  private
  public :: mizu_get_abi_version
  public :: mizu_runtime_create, mizu_runtime_destroy, mizu_runtime_copy_last_error
  public :: mizu_model_open, mizu_model_close, mizu_model_get_info
  public :: mizu_model_get_last_report
  public :: mizu_session_open, mizu_session_close, mizu_session_park
  public :: mizu_session_resume, mizu_session_get_info
  public :: mizu_session_attach_tokens, mizu_session_attach_modal_input
  public :: mizu_session_clear_pending_inputs, mizu_session_prefill
  public :: mizu_session_decode_step, mizu_session_read_output
  public :: mizu_session_get_last_report

  integer(i64), parameter :: INITIAL_REGISTRY_CAPACITY = 8_i64

  type, bind(c) :: c_runtime_config
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: abi_version
    type(c_ptr)        :: cache_root_z
    integer(c_int32_t) :: optimization_mode
    integer(c_int32_t) :: exploration_budget
    integer(c_int64_t) :: runtime_flags
  end type c_runtime_config

  type, bind(c) :: c_model_open_config
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: abi_version
    type(c_ptr)        :: model_root_z
    integer(c_int64_t) :: allowed_backend_mask
    integer(c_int64_t) :: model_flags
  end type c_model_open_config

  type, bind(c) :: c_session_config
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: abi_version
    integer(c_int64_t) :: max_context_tokens
    integer(c_int64_t) :: max_decode_tokens
    integer(c_int32_t) :: sampler_kind
    integer(c_int64_t) :: seed
    real(c_float)      :: temperature
    integer(c_int32_t) :: top_k
    real(c_float)      :: top_p
    integer(c_int64_t) :: session_flags
  end type c_session_config

  type, bind(c) :: c_model_info
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: model_family
    integer(c_int64_t) :: allowed_backend_mask
    integer(c_int64_t) :: model_features
    integer(c_int32_t) :: projector_slot_count
    integer(c_int32_t) :: reserved_u32
  end type c_model_info

  type, bind(c) :: c_session_info
    integer(c_size_t)  :: struct_size
    integer(c_int64_t) :: session_state_flags
    integer(c_int64_t) :: kv_token_count
    integer(c_int64_t) :: staged_token_count
    integer(c_int32_t) :: staged_modal_count
    integer(c_int32_t) :: reserved_u32
  end type c_session_info

  type, bind(c) :: c_modal_input_desc
    integer(c_size_t)  :: struct_size
    type(c_ptr)        :: slot_name_z
    integer(c_int32_t) :: placeholder_ordinal
    integer(c_int32_t) :: modality_kind
    integer(c_int32_t) :: storage_kind
    integer(c_int32_t) :: dtype
    integer(c_int32_t) :: rank
    type(c_ptr)        :: shape
    type(c_ptr)        :: data
    integer(c_size_t)  :: byte_count
    integer(c_int32_t) :: lifetime_policy
    integer(c_int64_t) :: input_flags
  end type c_modal_input_desc

  type, bind(c) :: c_decode_options
    integer(c_size_t)  :: struct_size
    integer(c_int64_t) :: token_budget
    integer(c_int64_t) :: stop_flags
    integer(c_int64_t) :: decode_flags
  end type c_decode_options

  type, bind(c) :: c_decode_result
    integer(c_size_t)  :: struct_size
    type(c_ptr)        :: token_buffer
    integer(c_size_t)  :: token_capacity
    integer(c_size_t)  :: token_count
    integer(c_int32_t) :: stop_reason
    integer(c_int64_t) :: result_flags
  end type c_decode_result

  type, bind(c) :: c_output_buffer
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: output_kind
    type(c_ptr)        :: data
    integer(c_size_t)  :: byte_capacity
    integer(c_size_t)  :: bytes_written
    integer(c_int64_t) :: output_flags
  end type c_output_buffer

  type, bind(c) :: c_execution_report
    integer(c_size_t)  :: struct_size
    integer(c_int32_t) :: stage_kind
    integer(c_int32_t) :: backend_family
    integer(c_int32_t) :: execution_route
    integer(c_int64_t) :: plan_id
    integer(c_int32_t) :: selection_mode
    integer(c_int32_t) :: cold_state
    integer(c_int32_t) :: fallback_reason
    integer(c_int64_t) :: cache_flags
    integer(c_int64_t) :: elapsed_us
  end type c_execution_report

  type, bind(c) :: c_report_buffer
    integer(c_size_t)  :: struct_size
    type(c_ptr)        :: reports
    integer(c_size_t)  :: report_capacity
    integer(c_size_t)  :: report_count
  end type c_report_buffer

  type, bind(c) :: runtime_box
    integer(c_int64_t) :: id = 0_c_int64_t
  end type runtime_box

  type, bind(c) :: model_box
    integer(c_int64_t) :: id = 0_c_int64_t
  end type model_box

  type, bind(c) :: session_box
    integer(c_int64_t) :: id = 0_c_int64_t
  end type session_box

  type(runtime_state), allocatable, target, save :: runtime_registry(:)
  type(runtime_cache_bundle), allocatable, target, save :: runtime_cache_registry(:)
  type(runtime_optimization_store), allocatable, target, save :: runtime_optimization_registry(:)
  type(model_state), allocatable, target, save   :: model_registry(:)
  type(session_state), allocatable, target, save :: session_registry(:)

  logical, allocatable, save :: runtime_used(:)
  logical, allocatable, save :: model_used(:)
  logical, allocatable, save :: session_used(:)

  interface
    function c_strlen(str) bind(c, name="strlen") result(length)
      import c_ptr, c_size_t
      type(c_ptr), value :: str
      integer(c_size_t)  :: length
    end function c_strlen
  end interface

contains

  integer(c_int32_t) function mizu_get_abi_version() bind(c, name="mizu_get_abi_version")
    mizu_get_abi_version = int(MIZU_ABI_VERSION, kind=c_int32_t)
  end function mizu_get_abi_version

  integer(c_int32_t) function mizu_runtime_create(config_ptr, out_runtime_ptr) &
      bind(c, name="mizu_runtime_create")
    type(c_ptr), value :: config_ptr
    type(c_ptr)        :: out_runtime_ptr
    type(c_runtime_config), pointer :: c_config
    type(runtime_box), pointer      :: box
    type(runtime_config)            :: config
    type(runtime_backend_registry)  :: backend_registry
    integer(i64)                    :: slot_id

    out_runtime_ptr = c_null_ptr

    if (.not. c_associated(config_ptr)) then
      mizu_runtime_create = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(config_ptr, c_config)
    if (.not. associated(c_config)) then
      mizu_runtime_create = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (int(c_config%abi_version, kind=i32) /= MIZU_ABI_VERSION) then
      mizu_runtime_create = int(MIZU_STATUS_ABI_MISMATCH, kind=c_int32_t)
      return
    end if

    config%abi_version        = int(c_config%abi_version, kind=i32)
    config%optimization_mode  = int(c_config%optimization_mode, kind=i32)
    config%exploration_budget = int(c_config%exploration_budget, kind=i32)
    config%runtime_flags      = int(c_config%runtime_flags, kind=i64)
    call copy_c_string_ptr_to_fortran(c_config%cache_root_z, config%cache_root)

    slot_id = acquire_runtime_slot()
    call initialize_runtime_state(runtime_registry(slot_id), config)
    call initialize_runtime_cache_bundle(runtime_cache_registry(slot_id))
    call initialize_runtime_optimization_store(runtime_optimization_registry(slot_id))
    call initialize_runtime_backend_registry(backend_registry)
    call probe_runtime_backend_registry(backend_registry)
    call apply_backend_registry_to_runtime(backend_registry, runtime_registry(slot_id))
    runtime_registry(slot_id)%handle%value = slot_id
    call hydrate_runtime_cache_state(runtime_registry(slot_id), runtime_cache_registry(slot_id))
    call hydrate_runtime_optimization_state(runtime_registry(slot_id), runtime_optimization_registry(slot_id))

    allocate(box)
    box%id = int(slot_id, kind=c_int64_t)
    out_runtime_ptr = c_loc(box)

    mizu_runtime_create = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_runtime_create

  integer(c_int32_t) function mizu_runtime_destroy(runtime_ptr) bind(c, name="mizu_runtime_destroy")
    type(c_ptr), value :: runtime_ptr
    type(runtime_box), pointer  :: box
    type(runtime_state), pointer :: runtime
    integer(i32) :: status_code

    if (.not. c_associated(runtime_ptr)) then
      mizu_runtime_destroy = int(MIZU_STATUS_OK, kind=c_int32_t)
      return
    end if

    call resolve_runtime_handle(runtime_ptr, box, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_destroy = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_runtime_destroy(runtime)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_destroy = int(status_code, kind=c_int32_t)
      return
    end if

    call persist_runtime_cache_state(runtime, runtime_cache_registry(int(box%id, kind=i64)))
    call persist_runtime_optimization_state(runtime, runtime_optimization_registry(int(box%id, kind=i64)))
    call reset_runtime_state(runtime)
    call reset_runtime_cache_bundle(runtime_cache_registry(int(box%id, kind=i64)))
    call reset_runtime_optimization_store(runtime_optimization_registry(int(box%id, kind=i64)))
    runtime_used(int(box%id, kind=i64)) = .false.
    deallocate(box)

    mizu_runtime_destroy = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_runtime_destroy

  integer(c_int32_t) function mizu_runtime_copy_last_error(runtime_ptr, buffer_ptr, capacity, &
                                                           out_required_ptr) &
      bind(c, name="mizu_runtime_copy_last_error")
    type(c_ptr), value         :: runtime_ptr
    type(c_ptr), value         :: buffer_ptr
    integer(c_size_t), value   :: capacity
    type(c_ptr), value         :: out_required_ptr
    type(runtime_box), pointer  :: box
    type(runtime_state), pointer :: runtime
    integer(i32)                :: status_code
    integer(i64)                :: required_len

    call resolve_runtime_handle(runtime_ptr, box, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_runtime_copy_last_error = int(status_code, kind=c_int32_t)
      return
    end if

    required_len = int(len_trim(runtime%last_error_message), kind=i64) + 1_i64
    call write_size_t_pointer(out_required_ptr, required_len)
    call copy_fortran_string_to_c(runtime%last_error_message, buffer_ptr, capacity)

    mizu_runtime_copy_last_error = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_runtime_copy_last_error

  integer(c_int32_t) function mizu_model_open(runtime_ptr, config_ptr, out_model_ptr) &
      bind(c, name="mizu_model_open")
    type(c_ptr), value :: runtime_ptr
    type(c_ptr), value :: config_ptr
    type(c_ptr)        :: out_model_ptr
    type(runtime_box), pointer      :: runtime_box_ptr
    type(runtime_state), pointer    :: runtime
    type(c_model_open_config), pointer :: c_config
    type(model_box), pointer        :: box
    type(model_open_config)         :: config
    type(model_manifest)            :: manifest
    type(model_info)                :: info
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i64)                    :: slot_id
    integer(i64)                    :: load_cache_flags
    integer(i64)                    :: model_plan_id
    integer(i64)                    :: load_elapsed_us
    integer(i32)                    :: selection_mode
    integer(i32)                    :: report_backend_family
    integer(i32)                    :: report_route
    integer(i32)                    :: status_code
    integer(i64)                    :: stage_started_us

    out_model_ptr = c_null_ptr

    call resolve_runtime_handle(runtime_ptr, runtime_box_ptr, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_open = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(config_ptr)) then
      call set_runtime_error(runtime, MIZU_STATUS_INVALID_ARGUMENT, "model config pointer is null")
      mizu_model_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(config_ptr, c_config)
    if (.not. associated(c_config)) then
      call set_runtime_error(runtime, MIZU_STATUS_INVALID_ARGUMENT, "model config pointer is invalid")
      mizu_model_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (int(c_config%abi_version, kind=i32) /= MIZU_ABI_VERSION) then
      call set_runtime_error(runtime, MIZU_STATUS_ABI_MISMATCH, "model config ABI version mismatch")
      mizu_model_open = int(MIZU_STATUS_ABI_MISMATCH, kind=c_int32_t)
      return
    end if

    config%abi_version         = int(c_config%abi_version, kind=i32)
    config%allowed_backend_mask = int(c_config%allowed_backend_mask, kind=i64)
    config%model_flags         = int(c_config%model_flags, kind=i64)
    call copy_c_string_ptr_to_fortran(c_config%model_root_z, config%model_root)

    stage_started_us = monotonic_timestamp_us()
    status_code = build_model_info(config, manifest, info)
    if (status_code /= MIZU_STATUS_OK) then
      call set_runtime_error(runtime, status_code, "model manifest load failed")
      mizu_model_open = int(status_code, kind=c_int32_t)
      return
    end if

    slot_id = acquire_model_slot()
    call initialize_model_state(model_registry(slot_id), config, info)
    model_registry(slot_id)%handle%value = slot_id
    model_registry(slot_id)%runtime_owner%value = runtime%handle%value
    model_registry(slot_id)%source_format = manifest%provenance%source_format
    model_registry(slot_id)%logical_model_hash = manifest%logical_model_hash
    model_registry(slot_id)%projector_revision = manifest%projector%revision_identity
    model_registry(slot_id)%tensor_count = manifest_tensor_count(manifest)
    model_registry(slot_id)%modality_count = manifest_modality_count(manifest)
    model_registry(slot_id)%source_model_id = manifest%provenance%source_model_id
    runtime_cache => runtime_cache_registry(runtime%handle%value)
    optimization_store => runtime_optimization_registry(runtime%handle%value)
    load_elapsed_us = elapsed_since_us(stage_started_us)
    load_cache_flags = resolve_weight_cache_flags(runtime, runtime_cache, optimization_store, &
      manifest, info%allowed_backend_mask, load_elapsed_us, model_plan_id, selection_mode, &
      report_backend_family, report_route)
    model_registry(slot_id)%last_report = make_stage_report(MIZU_STAGE_MODEL_LOAD, report_backend_family, &
      report_route, MIZU_FALLBACK_REASON_NONE, selection_mode, &
      MIZU_COLD_STATE_COLD, load_cache_flags, model_plan_id, load_elapsed_us)

    call register_model(runtime)

    allocate(box)
    box%id = int(slot_id, kind=c_int64_t)
    out_model_ptr = c_loc(box)

    mizu_model_open = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_open

  integer(c_int32_t) function mizu_model_close(model_ptr) bind(c, name="mizu_model_close")
    type(c_ptr), value :: model_ptr
    type(model_box), pointer   :: box
    type(model_state), pointer :: model
    type(runtime_state), pointer :: runtime
    integer(i32) :: status_code
    integer(i64) :: runtime_id

    if (.not. c_associated(model_ptr)) then
      mizu_model_close = int(MIZU_STATUS_OK, kind=c_int32_t)
      return
    end if

    call resolve_model_handle(model_ptr, box, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_close = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = validate_model_close(model)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_close = int(status_code, kind=c_int32_t)
      return
    end if

    runtime_id = model%runtime_owner%value
    if (is_runtime_slot_valid(runtime_id)) then
      runtime => runtime_registry(runtime_id)
      call unregister_model(runtime)
    end if

    call reset_model_state(model)
    model_used(int(box%id, kind=i64)) = .false.
    deallocate(box)

    mizu_model_close = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_close

  integer(c_int32_t) function mizu_model_get_info(model_ptr, out_info_ptr) bind(c, name="mizu_model_get_info")
    type(c_ptr), value :: model_ptr
    type(c_ptr), value :: out_info_ptr
    type(model_box), pointer   :: box
    type(model_state), pointer :: model
    type(c_model_info), pointer :: c_info
    integer(i32) :: status_code

    call resolve_model_handle(model_ptr, box, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_get_info = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_info_ptr)) then
      mizu_model_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_info_ptr, c_info)
    if (.not. associated(c_info)) then
      mizu_model_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    c_info%model_family         = int(model%info%model_family, kind=c_int32_t)
    c_info%allowed_backend_mask = int(model%info%allowed_backend_mask, kind=c_int64_t)
    c_info%model_features       = int(model%info%model_features, kind=c_int64_t)
    c_info%projector_slot_count = int(model%info%projector_slot_count, kind=c_int32_t)
    c_info%reserved_u32         = 0_c_int32_t

    mizu_model_get_info = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_get_info

  integer(c_int32_t) function mizu_model_get_last_report(model_ptr, out_report_ptr) &
      bind(c, name="mizu_model_get_last_report")
    type(c_ptr), value :: model_ptr
    type(c_ptr), value :: out_report_ptr
    type(model_box), pointer    :: box
    type(model_state), pointer  :: model
    type(c_execution_report), pointer :: c_report
    integer(i32) :: status_code

    call resolve_model_handle(model_ptr, box, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_model_get_last_report = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_report_ptr)) then
      mizu_model_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_report_ptr, c_report)
    if (.not. associated(c_report)) then
      mizu_model_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call copy_internal_report_to_c(model%last_report, c_report)
    mizu_model_get_last_report = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_model_get_last_report

  integer(c_int32_t) function mizu_session_open(model_ptr, config_ptr, out_session_ptr) &
      bind(c, name="mizu_session_open")
    type(c_ptr), value :: model_ptr
    type(c_ptr), value :: config_ptr
    type(c_ptr)        :: out_session_ptr
    type(model_box), pointer     :: model_box_ptr
    type(model_state), pointer   :: model
    type(c_session_config), pointer :: c_config
    type(session_box), pointer   :: box
    type(session_config)         :: config
    integer(i64)                 :: slot_id
    integer(i32)                 :: status_code

    out_session_ptr = c_null_ptr

    call resolve_model_handle(model_ptr, model_box_ptr, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_open = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(config_ptr)) then
      mizu_session_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(config_ptr, c_config)
    if (.not. associated(c_config)) then
      mizu_session_open = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (int(c_config%abi_version, kind=i32) /= MIZU_ABI_VERSION) then
      mizu_session_open = int(MIZU_STATUS_ABI_MISMATCH, kind=c_int32_t)
      return
    end if

    config%abi_version        = int(c_config%abi_version, kind=i32)
    config%max_context_tokens = int(c_config%max_context_tokens, kind=i64)
    config%max_decode_tokens  = int(c_config%max_decode_tokens, kind=i64)
    config%sampler_kind       = int(c_config%sampler_kind, kind=i32)
    config%seed               = int(c_config%seed, kind=i64)
    config%temperature        = real(c_config%temperature, kind=r32)
    config%top_k              = int(c_config%top_k, kind=i32)
    config%top_p              = real(c_config%top_p, kind=r32)
    config%session_flags      = int(c_config%session_flags, kind=i64)

    slot_id = acquire_session_slot()
    call initialize_session_state(session_registry(slot_id), config)
    session_registry(slot_id)%handle%value      = slot_id
    session_registry(slot_id)%model_owner%value = model%handle%value

    call register_session(model)

    allocate(box)
    box%id = int(slot_id, kind=c_int64_t)
    out_session_ptr = c_loc(box)

    mizu_session_open = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_open

  integer(c_int32_t) function mizu_session_close(session_ptr) bind(c, name="mizu_session_close")
    type(c_ptr), value :: session_ptr
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    type(model_state), pointer   :: model
    integer(i32) :: status_code
    integer(i64) :: model_id

    if (.not. c_associated(session_ptr)) then
      mizu_session_close = int(MIZU_STATUS_OK, kind=c_int32_t)
      return
    end if

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_close = int(status_code, kind=c_int32_t)
      return
    end if

    model_id = session%model_owner%value
    if (is_model_slot_valid(model_id)) then
      model => model_registry(model_id)
      call unregister_session(model)
    end if

    call reset_session_state(session)
    session_used(int(box%id, kind=i64)) = .false.
    deallocate(box)

    mizu_session_close = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_close

  integer(c_int32_t) function mizu_session_park(session_ptr, out_reports_ptr) &
      bind(c, name="mizu_session_park")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer    :: box
    type(session_state), pointer  :: session
    type(model_state), pointer    :: model
    type(runtime_state), pointer  :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i64) :: cache_flags
    integer(i64) :: stage_plan_id
    integer(i64) :: stage_elapsed_us
    integer(i64) :: stage_started_us
    integer(i32) :: selection_mode
    integer(i32) :: report_backend_family
    integer(i32) :: report_route
    logical      :: checkpoint_offloaded
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = prepare_report_buffer(out_reports_ptr, 1_i64)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    call park_session_state(session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_park = int(status_code, kind=c_int32_t)
      return
    end if
    call persist_session_checkpoint(runtime, runtime_cache, model, session, checkpoint_offloaded)
    if (checkpoint_offloaded) call offload_live_context_record(session)

    stage_elapsed_us = elapsed_since_us(stage_started_us)
    call resolve_session_stage_cache(runtime, runtime_cache, optimization_store, model, session, &
      stage_elapsed_us, stage_plan_id, selection_mode, report_backend_family, report_route, cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_PARK, report_backend_family, report_route, &
      MIZU_FALLBACK_REASON_NONE, selection_mode, MIZU_COLD_STATE_WARM, &
      cache_flags, stage_plan_id, stage_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())

    mizu_session_park = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_park

  integer(c_int32_t) function mizu_session_resume(session_ptr, out_reports_ptr) &
      bind(c, name="mizu_session_resume")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer    :: box
    type(session_state), pointer  :: session
    type(model_state), pointer    :: model
    type(runtime_state), pointer  :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i64) :: cache_flags
    integer(i64) :: stage_plan_id
    integer(i64) :: stage_elapsed_us
    integer(i64) :: stage_started_us
    integer(i32) :: selection_mode
    integer(i32) :: report_backend_family
    integer(i32) :: report_route
    logical      :: restored_ok
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    status_code = prepare_report_buffer(out_reports_ptr, 1_i64)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    restored_ok = .true.
    if (session%live_context_byte_count > 0_i32 .and. .not. session%has_resident_live_context) then
      call restore_session_checkpoint(runtime%config%cache_root, runtime_cache, model, session, restored_ok)
      if (.not. restored_ok) then
        mizu_session_resume = int(MIZU_STATUS_INVALID_STATE, kind=c_int32_t)
        return
      end if
    end if
    call resume_session_state(session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_resume = int(status_code, kind=c_int32_t)
      return
    end if

    stage_elapsed_us = elapsed_since_us(stage_started_us)
    call resolve_session_stage_cache(runtime, runtime_cache, optimization_store, model, session, &
      stage_elapsed_us, stage_plan_id, selection_mode, report_backend_family, report_route, cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_RESUME, report_backend_family, report_route, &
      MIZU_FALLBACK_REASON_NONE, selection_mode, MIZU_COLD_STATE_WARM, &
      cache_flags, stage_plan_id, stage_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())

    mizu_session_resume = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_resume

  integer(c_int32_t) function mizu_session_get_info(session_ptr, out_info_ptr) &
      bind(c, name="mizu_session_get_info")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_info_ptr
    type(session_box), pointer    :: box
    type(session_state), pointer  :: session
    type(c_session_info), pointer :: c_info
    type(session_info)            :: info
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_get_info = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_info_ptr)) then
      mizu_session_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_info_ptr, c_info)
    if (.not. associated(c_info)) then
      mizu_session_get_info = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    info = build_session_info(session)
    c_info%session_state_flags = int(info%session_state_flags, kind=c_int64_t)
    c_info%kv_token_count      = int(info%kv_token_count, kind=c_int64_t)
    c_info%staged_token_count  = int(info%staged_token_count, kind=c_int64_t)
    c_info%staged_modal_count  = int(info%staged_modal_count, kind=c_int32_t)
    c_info%reserved_u32        = 0_c_int32_t

    mizu_session_get_info = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_get_info

  integer(c_int32_t) function mizu_session_attach_tokens(session_ptr, tokens_ptr, token_count, &
                                                         attach_flags) &
      bind(c, name="mizu_session_attach_tokens")
    type(c_ptr), value       :: session_ptr
    type(c_ptr), value       :: tokens_ptr
    integer(c_size_t), value :: token_count
    integer(c_int32_t), value :: attach_flags
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    integer(c_int32_t), pointer  :: token_values(:)
    integer(i32) :: status_code

    if (attach_flags /= int(MIZU_ATTACH_FLAG_NONE, kind=c_int32_t)) then
      ! Reserved for future policies.
    end if

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_attach_tokens = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(tokens_ptr)) then
      mizu_session_attach_tokens = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(tokens_ptr, token_values, [int(token_count)])
    if (.not. associated(token_values)) then
      mizu_session_attach_tokens = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call stage_tokens(session, int(token_count, kind=i64), status_code, int(token_values, kind=i32))
    mizu_session_attach_tokens = int(status_code, kind=c_int32_t)
  end function mizu_session_attach_tokens

  integer(c_int32_t) function mizu_session_attach_modal_input(session_ptr, input_ptr) &
      bind(c, name="mizu_session_attach_modal_input")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: input_ptr
    type(session_box), pointer        :: box
    type(session_state), pointer      :: session
    type(c_modal_input_desc), pointer :: input
    integer(c_i8), pointer            :: modal_bytes(:)
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_attach_modal_input = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(input_ptr)) then
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(input_ptr, input)
    if (.not. associated(input)) then
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(input%data) .and. input%byte_count > 0_c_size_t) then
      mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (input%byte_count > 0_c_size_t) then
      call c_f_pointer(input%data, modal_bytes, [int(input%byte_count)])
      if (.not. associated(modal_bytes)) then
        mizu_session_attach_modal_input = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
        return
      end if

      call stage_modal_input(session, status_code, int(input%byte_count, kind=i64), &
        int(input%modality_kind, kind=i32), int(input%dtype, kind=i32), &
        copy_c_string_ptr(input%slot_name_z, "image"), int(modal_bytes, kind=i8))
    else
      call stage_modal_input(session, status_code, int(input%byte_count, kind=i64), &
        int(input%modality_kind, kind=i32), int(input%dtype, kind=i32), &
        copy_c_string_ptr(input%slot_name_z, "image"))
    end if
    mizu_session_attach_modal_input = int(status_code, kind=c_int32_t)
  end function mizu_session_attach_modal_input

  integer(c_int32_t) function mizu_session_clear_pending_inputs(session_ptr) &
      bind(c, name="mizu_session_clear_pending_inputs")
    type(c_ptr), value :: session_ptr
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_clear_pending_inputs = int(status_code, kind=c_int32_t)
      return
    end if

    call clear_pending_inputs(session, status_code)
    mizu_session_clear_pending_inputs = int(status_code, kind=c_int32_t)
  end function mizu_session_clear_pending_inputs

  integer(c_int32_t) function mizu_session_prefill(session_ptr, out_reports_ptr) &
      bind(c, name="mizu_session_prefill")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer   :: box
    type(session_state), pointer :: session
    type(model_state), pointer   :: model
    type(runtime_state), pointer :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    integer(i32) :: status_code
    integer(i64) :: required_reports
    integer(i64) :: kv_before
    integer(i64) :: staged_tokens_before
    integer(i64) :: staged_token_hash_before
    integer(i64) :: staged_modal_byte_count_before
    integer(i64) :: staged_modal_hash_before
    integer(i32) :: staged_modal_before
    integer(i32) :: staged_modal_kind_before
    integer(i32) :: staged_modal_dtype_before
    integer(i64) :: projector_cache_flags
    integer(i64) :: prefill_cache_flags
    integer(i64) :: projector_plan_id
    integer(i64) :: prefill_plan_id
    integer(i64) :: projector_elapsed_us
    integer(i64) :: prefill_elapsed_us
    integer(i64) :: stage_started_us
    integer(i64) :: projector_embedding_count
    integer(i64) :: consumed_token_count
    integer(i32) :: prefill_cold_state
    integer(i32) :: projector_selection_mode
    integer(i32) :: prefill_selection_mode
    integer(i32) :: projector_backend_family
    integer(i32) :: projector_route
    integer(i32) :: projector_placeholder_count
    integer(i32) :: prefill_context_byte_count
    integer(i8)  :: prefill_context_bytes(MAX_LIVE_CONTEXT_BYTES)
    integer(i64) :: prefill_context_artifact_hash
    integer(i32) :: prefill_backend_family
    integer(i32) :: prefill_route
    character(len=MAX_PATH_LEN) :: staged_modal_slot_name_before
    character(len=MAX_CACHE_KEY_LEN) :: projector_optimization_key_text
    character(len=MAX_CACHE_KEY_LEN) :: projector_candidate_key_text
    character(len=MAX_CACHE_KEY_LEN) :: prefill_optimization_key_text
    character(len=MAX_CACHE_KEY_LEN) :: prefill_candidate_key_text
    logical      :: has_modal_inputs
    logical      :: projector_workspace_reserved
    logical      :: prefill_workspace_reserved
    type(execution_report) :: projector_report
    type(artifact_metadata_record) :: projector_artifact_metadata
    type(artifact_metadata_record) :: prefill_artifact_metadata

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    has_modal_inputs = (session%staged_modal_count > 0_i32)
    required_reports = merge(2_i64, 1_i64, has_modal_inputs)
    kv_before = session%kv_token_count
    staged_tokens_before = session%staged_token_count
    staged_token_hash_before = session%staged_token_hash
    staged_modal_before = session%staged_modal_count
    staged_modal_byte_count_before = session%staged_modal_byte_count
    staged_modal_hash_before = session%staged_modal_hash
    staged_modal_kind_before = session%staged_modal_kind
    staged_modal_dtype_before = session%staged_modal_dtype
    staged_modal_slot_name_before = session%staged_modal_slot_name
    prefill_cold_state = merge(MIZU_COLD_STATE_WARM, MIZU_COLD_STATE_COLD, session%has_live_context)

    status_code = prepare_report_buffer(out_reports_ptr, required_reports)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    if (has_modal_inputs) then
      call prepare_projector_stage_candidate(runtime, optimization_store, model, staged_modal_byte_count_before, &
        staged_modal_kind_before, staged_modal_dtype_before, staged_modal_slot_name_before, &
        projector_optimization_key_text, projector_candidate_key_text, projector_plan_id, &
        projector_selection_mode, projector_backend_family, projector_route, projector_artifact_metadata, &
        projector_placeholder_count)

      call reserve_stage_workspace(runtime, projector_artifact_metadata, projector_workspace_reserved, status_code)
      if (status_code /= MIZU_STATUS_OK) then
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if

      stage_started_us = monotonic_timestamp_us()
      projector_embedding_count = 0_i64
      if (projector_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
          (projector_route == MIZU_EXEC_ROUTE_ANE .or. projector_route == MIZU_EXEC_ROUTE_METAL)) then
        call execute_apple_projector(runtime%config%cache_root, trim(projector_artifact_metadata%payload_path), &
          projector_route, staged_modal_byte_count_before, projector_placeholder_count, staged_modal_hash_before, &
          projector_embedding_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use)
        if (status_code /= MIZU_STATUS_OK) then
          call release_stage_workspace(runtime, projector_workspace_reserved)
          mizu_session_prefill = int(status_code, kind=c_int32_t)
          return
        end if
      else if (projector_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. projector_route == MIZU_EXEC_ROUTE_CUDA) then
        call execute_cuda_projector(runtime%config%cache_root, trim(projector_artifact_metadata%payload_path), &
          staged_modal_byte_count_before, projector_placeholder_count, staged_modal_hash_before, &
          projector_embedding_count, status_code, runtime%workspace%host_buffer, &
          runtime%workspace%bytes_in_use)
        if (status_code /= MIZU_STATUS_OK) then
          call release_stage_workspace(runtime, projector_workspace_reserved)
          mizu_session_prefill = int(status_code, kind=c_int32_t)
          return
        end if
      end if
      call release_stage_workspace(runtime, projector_workspace_reserved)
      projector_elapsed_us = elapsed_since_us(stage_started_us)
      call finalize_projector_stage_cache(runtime_cache, optimization_store, trim(projector_optimization_key_text), &
        trim(projector_candidate_key_text), projector_plan_id, projector_selection_mode, projector_elapsed_us, &
        projector_artifact_metadata, projector_cache_flags)
      projector_report = make_stage_report(MIZU_STAGE_PROJECTOR, projector_backend_family, &
        projector_route, MIZU_FALLBACK_REASON_NONE, projector_selection_mode, prefill_cold_state, &
        projector_cache_flags, projector_plan_id, projector_elapsed_us)
    else
      projector_report = execution_report()
    end if

    call prepare_plan_stage_candidate(runtime, optimization_store, model, MIZU_STAGE_PREFILL, &
      OP_FAMILY_PREFILL, [max(0_i64, kv_before), max(0_i64, staged_tokens_before), &
      max(0_i64, int(staged_modal_before, kind=i64))], max(0_i64, staged_tokens_before), &
      prefill_optimization_key_text, prefill_candidate_key_text, prefill_plan_id, &
      prefill_selection_mode, prefill_backend_family, prefill_route, prefill_artifact_metadata)

    call reserve_stage_workspace(runtime, prefill_artifact_metadata, prefill_workspace_reserved, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    consumed_token_count = staged_tokens_before
    prefill_context_byte_count = 0_i32
    prefill_context_bytes = 0_i8
    prefill_context_artifact_hash = 0_i64
    if (prefill_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
        (prefill_route == MIZU_EXEC_ROUTE_ANE .or. prefill_route == MIZU_EXEC_ROUTE_METAL)) then
      if (allocated(session%staged_tokens)) then
        if (allocated(session%staged_modal_bytes)) then
          call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
            staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
            runtime%workspace%bytes_in_use, token_values=session%staged_tokens, &
            modal_bytes=session%staged_modal_bytes, context_bytes=prefill_context_bytes, &
            context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
        else
          call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
            staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
            runtime%workspace%bytes_in_use, token_values=session%staged_tokens, &
            context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
            context_artifact_hash=prefill_context_artifact_hash)
        end if
      else if (allocated(session%staged_modal_bytes)) then
        call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
          staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
          runtime%workspace%bytes_in_use, modal_bytes=session%staged_modal_bytes, &
          context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
          context_artifact_hash=prefill_context_artifact_hash)
      else
        call execute_apple_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          prefill_route, staged_tokens_before, staged_modal_before, staged_token_hash_before, &
          staged_modal_hash_before, consumed_token_count, status_code, runtime%workspace%host_buffer, &
          runtime%workspace%bytes_in_use, context_bytes=prefill_context_bytes, &
          context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
      end if
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, prefill_workspace_reserved)
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if
    else if (prefill_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. prefill_route == MIZU_EXEC_ROUTE_CUDA) then
      if (allocated(session%staged_tokens)) then
        if (allocated(session%staged_modal_bytes)) then
          call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
            consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
            token_values=session%staged_tokens, modal_bytes=session%staged_modal_bytes, &
            context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
            context_artifact_hash=prefill_context_artifact_hash)
        else
          call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
            staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
            consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
            token_values=session%staged_tokens, context_bytes=prefill_context_bytes, &
            context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
        end if
      else if (allocated(session%staged_modal_bytes)) then
        call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
          consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
          modal_bytes=session%staged_modal_bytes, context_bytes=prefill_context_bytes, &
          context_byte_count=prefill_context_byte_count, context_artifact_hash=prefill_context_artifact_hash)
      else
        call execute_cuda_prefill(runtime%config%cache_root, trim(prefill_artifact_metadata%payload_path), &
          staged_tokens_before, staged_modal_before, staged_token_hash_before, staged_modal_hash_before, &
          consumed_token_count, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
          context_bytes=prefill_context_bytes, context_byte_count=prefill_context_byte_count, &
          context_artifact_hash=prefill_context_artifact_hash)
      end if
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, prefill_workspace_reserved)
        mizu_session_prefill = int(status_code, kind=c_int32_t)
        return
      end if
    end if
    call release_stage_workspace(runtime, prefill_workspace_reserved)
    call complete_prefill(session, consumed_token_count=consumed_token_count, status_code=status_code, &
      token_content_hash=staged_token_hash_before, modal_content_hash=staged_modal_hash_before, &
      projector_embedding_count=projector_embedding_count)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_prefill = int(status_code, kind=c_int32_t)
      return
    end if
    call store_live_context_record(session, prefill_backend_family, prefill_route, prefill_context_bytes, &
      prefill_context_byte_count, producer_stage=MIZU_STAGE_PREFILL, artifact_hash=prefill_context_artifact_hash)
    prefill_elapsed_us = elapsed_since_us(stage_started_us)

    call finalize_plan_stage_cache(runtime_cache, optimization_store, trim(prefill_optimization_key_text), &
      trim(prefill_candidate_key_text), prefill_plan_id, prefill_selection_mode, prefill_elapsed_us, &
      prefill_artifact_metadata, prefill_cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_PREFILL, prefill_backend_family, prefill_route, &
      MIZU_FALLBACK_REASON_NONE, prefill_selection_mode, prefill_cold_state, &
      prefill_cache_flags, prefill_plan_id, prefill_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, projector_report)

    mizu_session_prefill = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_prefill

  integer(c_int32_t) function mizu_session_decode_step(session_ptr, options_ptr, out_result_ptr, &
                                                       out_reports_ptr) &
      bind(c, name="mizu_session_decode_step")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: options_ptr
    type(c_ptr), value :: out_result_ptr
    type(c_ptr), value :: out_reports_ptr
    type(session_box), pointer       :: box
    type(session_state), pointer     :: session
    type(model_state), pointer       :: model
    type(runtime_state), pointer     :: runtime
    type(runtime_cache_bundle), pointer :: runtime_cache
    type(runtime_optimization_store), pointer :: optimization_store
    type(c_decode_options), pointer  :: options
    type(c_decode_result), pointer   :: result
    integer(c_int32_t), pointer      :: token_buffer(:)
    integer(c_int32_t)               :: token_value
    integer(i32) :: status_code
    integer(i64) :: emitted_token_count
    integer(i64) :: kv_before
    integer(i64) :: decode_cache_flags
    integer(i64) :: decode_plan_id
    integer(i64) :: decode_elapsed_us
    integer(i64) :: stage_started_us
    integer(i32) :: decode_stop_reason
    integer(i32) :: updated_context_byte_count
    integer(i8)  :: updated_context_bytes(MAX_LIVE_CONTEXT_BYTES)
    integer(i64) :: decode_context_artifact_hash
    integer(i32) :: selection_mode
    integer(i32) :: report_backend_family
    integer(i32) :: report_route
    integer(i32) :: emitted_tokens_local(MAX_RECENT_OUTPUT_TOKENS)
    character(len=MAX_CACHE_KEY_LEN) :: decode_optimization_key_text
    character(len=MAX_CACHE_KEY_LEN) :: decode_candidate_key_text
    logical      :: decode_workspace_reserved
    type(artifact_metadata_record) :: decode_artifact_metadata

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    call resolve_session_owner_model(session, model, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_runtime(model, runtime, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_cache(model, runtime_cache, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    call resolve_model_owner_optimizer(model, optimization_store, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(options_ptr) .or. .not. c_associated(out_result_ptr)) then
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(options_ptr, options)
    call c_f_pointer(out_result_ptr, result)
    if ((.not. associated(options)) .or. (.not. associated(result))) then
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = prepare_report_buffer(out_reports_ptr, 1_i64)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    if (options%token_budget <= 0_c_int64_t) then
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    kv_before = session%kv_token_count
    emitted_tokens_local = 0_i32
    updated_context_byte_count = 0_i32
    updated_context_bytes = 0_i8
    decode_context_artifact_hash = 0_i64
    call prepare_plan_stage_candidate(runtime, optimization_store, model, MIZU_STAGE_DECODE, &
      OP_FAMILY_DECODE, [max(0_i64, kv_before), max(0_i64, int(options%token_budget, kind=i64)), 1_i64], &
      max(0_i64, int(options%token_budget, kind=i64)), decode_optimization_key_text, &
      decode_candidate_key_text, decode_plan_id, selection_mode, report_backend_family, report_route, &
      decode_artifact_metadata)

    if (session%has_live_context .and. session%live_context_producer_stage == MIZU_STAGE_DECODE) then
      if (report_backend_family /= session%live_context_backend_family .or. &
          report_route /= session%live_context_execution_route) then
        mizu_session_decode_step = int(MIZU_STATUS_INVALID_STATE, kind=c_int32_t)
        return
      end if
    end if

    call reserve_stage_workspace(runtime, decode_artifact_metadata, decode_workspace_reserved, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if

    stage_started_us = monotonic_timestamp_us()
    emitted_token_count = min(int(options%token_budget, kind=i64), 1_i64)
    decode_stop_reason = MIZU_STOP_REASON_NONE
    token_value = int(mod(session%kv_token_count, 4096_i64), kind=c_int32_t)
    if (token_value == 0_c_int32_t) token_value = 1_c_int32_t
    if (report_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
        (report_route == MIZU_EXEC_ROUTE_ANE .or. report_route == MIZU_EXEC_ROUTE_METAL)) then
      call execute_apple_decode(runtime%config%cache_root, trim(decode_artifact_metadata%payload_path), &
        report_route, kv_before, int(options%token_budget, kind=i64), emitted_token_count, token_value, &
        decode_stop_reason, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
        session%live_context_bytes, session%live_context_byte_count, updated_context_bytes, &
        updated_context_byte_count, context_artifact_hash=decode_context_artifact_hash)
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, decode_workspace_reserved)
        mizu_session_decode_step = int(status_code, kind=c_int32_t)
        return
      end if
    else if (report_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. report_route == MIZU_EXEC_ROUTE_CUDA) then
      call execute_cuda_decode(runtime%config%cache_root, trim(decode_artifact_metadata%payload_path), &
        kv_before, int(options%token_budget, kind=i64), emitted_token_count, token_value, &
        decode_stop_reason, status_code, runtime%workspace%host_buffer, runtime%workspace%bytes_in_use, &
        session%live_context_bytes, session%live_context_byte_count, updated_context_bytes, &
        updated_context_byte_count, context_artifact_hash=decode_context_artifact_hash)
      if (status_code /= MIZU_STATUS_OK) then
        call release_stage_workspace(runtime, decode_workspace_reserved)
        mizu_session_decode_step = int(status_code, kind=c_int32_t)
        return
      end if
    end if
    call release_stage_workspace(runtime, decode_workspace_reserved)

    result%token_count  = int(emitted_token_count, kind=c_size_t)
    result%stop_reason  = int(decode_stop_reason, kind=c_int32_t)
    result%result_flags = 0_c_int64_t

    if (result%token_capacity < int(emitted_token_count, kind=c_size_t)) then
      mizu_session_decode_step = int(MIZU_STATUS_BUFFER_TOO_SMALL, kind=c_int32_t)
      return
    end if

    if (emitted_token_count > 0_i64 .and. .not. c_associated(result%token_buffer)) then
      mizu_session_decode_step = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (emitted_token_count > 0_i64) emitted_tokens_local(1) = int(token_value, kind=i32)
    call complete_decode(session, emitted_token_count, decode_stop_reason, status_code, emitted_tokens_local)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_decode_step = int(status_code, kind=c_int32_t)
      return
    end if
    if ((report_backend_family == MIZU_BACKEND_FAMILY_APPLE .and. &
         (report_route == MIZU_EXEC_ROUTE_ANE .or. report_route == MIZU_EXEC_ROUTE_METAL)) .or. &
        (report_backend_family == MIZU_BACKEND_FAMILY_CUDA .and. report_route == MIZU_EXEC_ROUTE_CUDA)) then
      call update_live_context_record(session, updated_context_bytes, updated_context_byte_count, &
        producer_stage=MIZU_STAGE_DECODE, artifact_hash=decode_context_artifact_hash, &
        backend_family=report_backend_family, execution_route=report_route)
    end if
    decode_elapsed_us = elapsed_since_us(stage_started_us)

    if (emitted_token_count > 0_i64) then
      call c_f_pointer(result%token_buffer, token_buffer, [int(emitted_token_count)])
      token_buffer(1) = token_value
    end if

    call finalize_plan_stage_cache(runtime_cache, optimization_store, trim(decode_optimization_key_text), &
      trim(decode_candidate_key_text), decode_plan_id, selection_mode, decode_elapsed_us, &
      decode_artifact_metadata, decode_cache_flags)
    session%last_report = make_stage_report(MIZU_STAGE_DECODE, report_backend_family, report_route, &
      MIZU_FALLBACK_REASON_NONE, selection_mode, MIZU_COLD_STATE_WARM, &
      decode_cache_flags, decode_plan_id, decode_elapsed_us)
    call fill_report_buffer(out_reports_ptr, session%last_report, execution_report())

    mizu_session_decode_step = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_decode_step

  integer(c_int32_t) function mizu_session_read_output(session_ptr, out_output_ptr) &
      bind(c, name="mizu_session_read_output")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_output_ptr
    type(session_box), pointer      :: box
    type(session_state), pointer    :: session
    type(c_output_buffer), pointer  :: output
    integer(c_int32_t), pointer     :: token_buffer(:)
    integer(c_int32_t)              :: token_example
    integer(i32) :: status_code
    integer(i64) :: bytes_required

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_read_output = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_output_ptr)) then
      mizu_session_read_output = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_output_ptr, output)
    if (.not. associated(output)) then
      mizu_session_read_output = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    status_code = validate_read_output(session)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_read_output = int(status_code, kind=c_int32_t)
      return
    end if

    if (int(output%output_kind, kind=i32) /= MIZU_OUTPUT_KIND_TOKEN_IDS) then
      mizu_session_read_output = int(MIZU_STATUS_UNSUPPORTED_MODALITY, kind=c_int32_t)
      return
    end if

    bytes_required = session%last_output_token_count * int(c_sizeof(token_example), kind=i64)
    output%bytes_written = int(bytes_required, kind=c_size_t)

    if (output%byte_capacity < int(bytes_required, kind=c_size_t)) then
      mizu_session_read_output = int(MIZU_STATUS_BUFFER_TOO_SMALL, kind=c_int32_t)
      return
    end if

    if (bytes_required > 0_i64 .and. .not. c_associated(output%data)) then
      mizu_session_read_output = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    if (session%last_output_token_count > 0_i64) then
      call c_f_pointer(output%data, token_buffer, [int(session%last_output_token_count)])
      token_buffer(1:int(session%last_output_token_count)) = int( &
        session%last_output_tokens(1:int(session%last_output_token_count)), kind=c_int32_t)
    end if

    mizu_session_read_output = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_read_output

  integer(c_int32_t) function mizu_session_get_last_report(session_ptr, out_report_ptr) &
      bind(c, name="mizu_session_get_last_report")
    type(c_ptr), value :: session_ptr
    type(c_ptr), value :: out_report_ptr
    type(session_box), pointer      :: box
    type(session_state), pointer    :: session
    type(c_execution_report), pointer :: c_report
    integer(i32) :: status_code

    call resolve_session_handle(session_ptr, box, session, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      mizu_session_get_last_report = int(status_code, kind=c_int32_t)
      return
    end if

    if (.not. c_associated(out_report_ptr)) then
      mizu_session_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call c_f_pointer(out_report_ptr, c_report)
    if (.not. associated(c_report)) then
      mizu_session_get_last_report = int(MIZU_STATUS_INVALID_ARGUMENT, kind=c_int32_t)
      return
    end if

    call copy_internal_report_to_c(session%last_report, c_report)
    mizu_session_get_last_report = int(MIZU_STATUS_OK, kind=c_int32_t)
  end function mizu_session_get_last_report

  subroutine ensure_runtime_registry_capacity(required_capacity)
    integer(i64), intent(in) :: required_capacity
    type(runtime_state), allocatable :: new_registry(:)
    type(runtime_cache_bundle), allocatable :: new_cache_registry(:)
    type(runtime_optimization_store), allocatable :: new_optimization_registry(:)
    logical, allocatable             :: new_used(:)
    integer(i64)                     :: current_capacity, new_capacity

    if (.not. allocated(runtime_registry)) then
      new_capacity = max(INITIAL_REGISTRY_CAPACITY, required_capacity)
      allocate(runtime_registry(new_capacity), runtime_cache_registry(new_capacity), &
               runtime_optimization_registry(new_capacity), runtime_used(new_capacity))
      runtime_registry = runtime_state()
      runtime_cache_registry = runtime_cache_bundle()
      runtime_optimization_registry = runtime_optimization_store()
      runtime_used     = .false.
      return
    end if

    current_capacity = int(size(runtime_registry), kind=i64)
    if (required_capacity <= current_capacity) return

    new_capacity = max(required_capacity, 2_i64 * current_capacity)
    allocate(new_registry(new_capacity), new_cache_registry(new_capacity), &
             new_optimization_registry(new_capacity), new_used(new_capacity))
    new_registry = runtime_state()
    new_cache_registry = runtime_cache_bundle()
    new_optimization_registry = runtime_optimization_store()
    new_used     = .false.
    new_registry(1:current_capacity) = runtime_registry
    new_cache_registry(1:current_capacity) = runtime_cache_registry
    new_optimization_registry(1:current_capacity) = runtime_optimization_registry
    new_used(1:current_capacity)     = runtime_used
    call move_alloc(new_registry, runtime_registry)
    call move_alloc(new_cache_registry, runtime_cache_registry)
    call move_alloc(new_optimization_registry, runtime_optimization_registry)
    call move_alloc(new_used, runtime_used)
  end subroutine ensure_runtime_registry_capacity

  subroutine ensure_model_registry_capacity(required_capacity)
    integer(i64), intent(in) :: required_capacity
    type(model_state), allocatable :: new_registry(:)
    logical, allocatable           :: new_used(:)
    integer(i64)                   :: current_capacity, new_capacity

    if (.not. allocated(model_registry)) then
      new_capacity = max(INITIAL_REGISTRY_CAPACITY, required_capacity)
      allocate(model_registry(new_capacity), model_used(new_capacity))
      model_registry = model_state()
      model_used     = .false.
      return
    end if

    current_capacity = int(size(model_registry), kind=i64)
    if (required_capacity <= current_capacity) return

    new_capacity = max(required_capacity, 2_i64 * current_capacity)
    allocate(new_registry(new_capacity), new_used(new_capacity))
    new_registry = model_state()
    new_used     = .false.
    new_registry(1:current_capacity) = model_registry
    new_used(1:current_capacity)     = model_used
    call move_alloc(new_registry, model_registry)
    call move_alloc(new_used, model_used)
  end subroutine ensure_model_registry_capacity

  subroutine ensure_session_registry_capacity(required_capacity)
    integer(i64), intent(in) :: required_capacity
    type(session_state), allocatable :: new_registry(:)
    logical, allocatable             :: new_used(:)
    integer(i64)                     :: current_capacity, new_capacity

    if (.not. allocated(session_registry)) then
      new_capacity = max(INITIAL_REGISTRY_CAPACITY, required_capacity)
      allocate(session_registry(new_capacity), session_used(new_capacity))
      session_registry = session_state()
      session_used     = .false.
      return
    end if

    current_capacity = int(size(session_registry), kind=i64)
    if (required_capacity <= current_capacity) return

    new_capacity = max(required_capacity, 2_i64 * current_capacity)
    allocate(new_registry(new_capacity), new_used(new_capacity))
    new_registry = session_state()
    new_used     = .false.
    new_registry(1:current_capacity) = session_registry
    new_used(1:current_capacity)     = session_used
    call move_alloc(new_registry, session_registry)
    call move_alloc(new_used, session_used)
  end subroutine ensure_session_registry_capacity

  integer(i64) function acquire_runtime_slot() result(slot_id)
    integer(i64) :: index

    call ensure_runtime_registry_capacity(INITIAL_REGISTRY_CAPACITY)
    do index = 1_i64, int(size(runtime_used), kind=i64)
      if (.not. runtime_used(index)) then
        runtime_used(index) = .true.
        slot_id = index
        return
      end if
    end do

    index = int(size(runtime_used), kind=i64) + 1_i64
    call ensure_runtime_registry_capacity(index)
    runtime_used(index) = .true.
    slot_id = index
  end function acquire_runtime_slot

  integer(i64) function acquire_model_slot() result(slot_id)
    integer(i64) :: index

    call ensure_model_registry_capacity(INITIAL_REGISTRY_CAPACITY)
    do index = 1_i64, int(size(model_used), kind=i64)
      if (.not. model_used(index)) then
        model_used(index) = .true.
        slot_id = index
        return
      end if
    end do

    index = int(size(model_used), kind=i64) + 1_i64
    call ensure_model_registry_capacity(index)
    model_used(index) = .true.
    slot_id = index
  end function acquire_model_slot

  integer(i64) function acquire_session_slot() result(slot_id)
    integer(i64) :: index

    call ensure_session_registry_capacity(INITIAL_REGISTRY_CAPACITY)
    do index = 1_i64, int(size(session_used), kind=i64)
      if (.not. session_used(index)) then
        session_used(index) = .true.
        slot_id = index
        return
      end if
    end do

    index = int(size(session_used), kind=i64) + 1_i64
    call ensure_session_registry_capacity(index)
    session_used(index) = .true.
    slot_id = index
  end function acquire_session_slot

  subroutine resolve_runtime_handle(runtime_ptr, box, runtime, status_code)
    type(c_ptr), value             :: runtime_ptr
    type(runtime_box), pointer     :: box
    type(runtime_state), pointer   :: runtime
    integer(i32), intent(out)      :: status_code
    integer(i64)                   :: slot_id

    if (.not. c_associated(runtime_ptr)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call c_f_pointer(runtime_ptr, box)
    if (.not. associated(box)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    slot_id = int(box%id, kind=i64)
    if (.not. is_runtime_slot_valid(slot_id)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    runtime => runtime_registry(slot_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_runtime_handle

  subroutine resolve_model_handle(model_ptr, box, model, status_code)
    type(c_ptr), value           :: model_ptr
    type(model_box), pointer     :: box
    type(model_state), pointer   :: model
    integer(i32), intent(out)    :: status_code
    integer(i64)                 :: slot_id

    if (.not. c_associated(model_ptr)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call c_f_pointer(model_ptr, box)
    if (.not. associated(box)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    slot_id = int(box%id, kind=i64)
    if (.not. is_model_slot_valid(slot_id)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    model => model_registry(slot_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_handle

  subroutine resolve_session_handle(session_ptr, box, session, status_code)
    type(c_ptr), value             :: session_ptr
    type(session_box), pointer     :: box
    type(session_state), pointer   :: session
    integer(i32), intent(out)      :: status_code
    integer(i64)                   :: slot_id

    if (.not. c_associated(session_ptr)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call c_f_pointer(session_ptr, box)
    if (.not. associated(box)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    slot_id = int(box%id, kind=i64)
    if (.not. is_session_slot_valid(slot_id)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    session => session_registry(slot_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_session_handle

  pure logical function is_runtime_slot_valid(slot_id) result(is_valid)
    integer(i64), intent(in) :: slot_id

    is_valid = allocated(runtime_used) .and. slot_id >= 1_i64 .and. &
      slot_id <= int(size(runtime_used), kind=i64) .and. runtime_used(slot_id)
  end function is_runtime_slot_valid

  pure logical function is_model_slot_valid(slot_id) result(is_valid)
    integer(i64), intent(in) :: slot_id

    is_valid = allocated(model_used) .and. slot_id >= 1_i64 .and. &
      slot_id <= int(size(model_used), kind=i64) .and. model_used(slot_id)
  end function is_model_slot_valid

  pure logical function is_session_slot_valid(slot_id) result(is_valid)
    integer(i64), intent(in) :: slot_id

    is_valid = allocated(session_used) .and. slot_id >= 1_i64 .and. &
      slot_id <= int(size(session_used), kind=i64) .and. session_used(slot_id)
  end function is_session_slot_valid

  integer(i32) function build_model_info(config, manifest, info) result(status_code)
    type(model_open_config), intent(in) :: config
    type(model_manifest), intent(out)   :: manifest
    type(model_info), intent(out)       :: info

    info = model_info()

    if (config%allowed_backend_mask == MIZU_BACKEND_MASK_NONE) then
      status_code = MIZU_STATUS_NO_VALID_PLAN
      return
    end if

    status_code = load_model_manifest_from_root(config%model_root, manifest)
    if (status_code /= MIZU_STATUS_OK) return

    call populate_model_info_from_manifest(manifest, info)
    info%allowed_backend_mask = config%allowed_backend_mask
    status_code = MIZU_STATUS_OK
  end function build_model_info

  subroutine resolve_session_owner_model(session, model, status_code)
    type(session_state), intent(in)   :: session
    type(model_state), pointer        :: model
    integer(i32), intent(out)         :: status_code
    integer(i64)                      :: model_id

    nullify(model)
    model_id = session%model_owner%value
    if (.not. is_model_slot_valid(model_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    model => model_registry(model_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_session_owner_model

  subroutine resolve_model_owner_runtime(model, runtime, status_code)
    type(model_state), intent(in) :: model
    type(runtime_state), pointer  :: runtime
    integer(i32), intent(out)     :: status_code
    integer(i64)                  :: runtime_id

    nullify(runtime)
    runtime_id = model%runtime_owner%value
    if (.not. is_runtime_slot_valid(runtime_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    runtime => runtime_registry(runtime_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_owner_runtime

  subroutine resolve_model_owner_cache(model, runtime_cache, status_code)
    type(model_state), intent(in)            :: model
    type(runtime_cache_bundle), pointer      :: runtime_cache
    integer(i32), intent(out)                :: status_code
    integer(i64)                             :: runtime_id

    nullify(runtime_cache)
    runtime_id = model%runtime_owner%value
    if (.not. is_runtime_slot_valid(runtime_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    runtime_cache => runtime_cache_registry(runtime_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_owner_cache

  subroutine resolve_model_owner_optimizer(model, optimization_store, status_code)
    type(model_state), intent(in)                 :: model
    type(runtime_optimization_store), pointer     :: optimization_store
    integer(i32), intent(out)                     :: status_code
    integer(i64)                                  :: runtime_id

    nullify(optimization_store)
    runtime_id = model%runtime_owner%value
    if (.not. is_runtime_slot_valid(runtime_id)) then
      status_code = MIZU_STATUS_INVALID_STATE
      return
    end if

    optimization_store => runtime_optimization_registry(runtime_id)
    status_code = MIZU_STATUS_OK
  end subroutine resolve_model_owner_optimizer

  pure subroutine populate_manifest_identity(model, manifest)
    type(model_state), intent(in)   :: model
    type(model_manifest), intent(out) :: manifest

    manifest = model_manifest()
    manifest%model_family = model%info%model_family
    manifest%model_features = model%info%model_features
    manifest%logical_model_hash = model%logical_model_hash
    manifest%provenance%source_model_id = model%source_model_id
    manifest%projector%is_present = (model%projector_revision /= 0_i64)
    manifest%projector%revision_identity = model%projector_revision
    if (manifest%projector%is_present) then
      manifest%projector%slot_name = "image"
      manifest%projector%placeholder_count = 1_i32
      manifest%projector%input_dtype = MIZU_DTYPE_U8
      manifest%projector%embedding_dtype = MIZU_DTYPE_BF16
    end if
  end subroutine populate_manifest_identity

  integer(i64) function resolve_weight_cache_flags(runtime, runtime_cache, optimization_store, manifest, &
                                                   allowed_backend_mask, elapsed_us, plan_id, &
                                                   selection_mode, backend_family, execution_route) &
      result(cache_flags)
    type(runtime_state), intent(in)                  :: runtime
    type(runtime_cache_bundle), intent(inout)        :: runtime_cache
    type(runtime_optimization_store), intent(inout)  :: optimization_store
    type(model_manifest), intent(in)                 :: manifest
    integer(i64), intent(in)                         :: allowed_backend_mask
    integer(i64), intent(in)                         :: elapsed_us
    integer(i64), intent(out)                        :: plan_id
    integer(i32), intent(out)                        :: selection_mode
    integer(i32), intent(out)                        :: backend_family
    integer(i32), intent(out)                        :: execution_route
    type(plan_request)                               :: stage_request
    type(weight_cache_key)                           :: optimization_key
    type(weight_cache_key)                           :: candidate_key
    character(len=MAX_CACHE_KEY_LEN)                 :: optimization_key_text
    character(len=MAX_CACHE_KEY_LEN)                 :: candidate_key_text
    character(len=MAX_CACHE_KEY_LEN)                 :: candidate_key_texts(3)
    integer(i64)                                     :: candidate_plan_ids(3)
    integer(i32)                                     :: candidate_backend_families(3)
    integer(i32)                                     :: candidate_execution_routes(3)
    integer(i32)                                     :: optimization_backend_family
    integer(i32)                                     :: candidate_count
    integer(i32)                                     :: candidate_index
    logical                                          :: was_hit
    logical                                          :: reused_winner

    call enumerate_candidate_routes(allowed_backend_mask, candidate_backend_families, &
      candidate_execution_routes, candidate_count)
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)

    call build_weight_cache_key(manifest, "unbound", "logical", optimization_backend_family, &
      MIZU_EXEC_ROUTE_NONE, optimization_key)
    optimization_key_text = append_allowed_mask_identity(trim(optimization_key%key_text), allowed_backend_mask)
    call initialize_plan_request(stage_request, MIZU_STAGE_MODEL_LOAD, OP_FAMILY_NONE, &
      manifest%model_family, allowed_backend_mask)
    stage_request%shape_signature = 0_i64
    stage_request%shape_signature(1) = manifest%logical_model_hash
    stage_request%shape_signature(2) = manifest%projector%revision_identity
    stage_request%planner_version_hint = int(manifest%runtime_version%planner_version, kind=i64)

    do candidate_index = 1_i32, candidate_count
      call build_weight_cache_key(manifest, "unbound", "logical", candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), candidate_key)
      candidate_key_texts(candidate_index) = trim(candidate_key%key_text)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_texts(candidate_index)))
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    call touch_weight_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    call record_weight_artifact_metadata(runtime_cache, trim(candidate_key_text), &
      build_stage_artifact_metadata(MIZU_STAGE_MODEL_LOAD, backend_family, execution_route, &
        trim(candidate_key_text), stage_request, runtime%config%cache_root))
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_WEIGHT_HIT, was_hit, reused_winner)
  end function resolve_weight_cache_flags

  subroutine resolve_session_stage_cache(runtime, runtime_cache, optimization_store, model, session, &
                                         elapsed_us, plan_id, selection_mode, backend_family, &
                                         execution_route, cache_flags)
    type(runtime_state), intent(in)                 :: runtime
    type(runtime_cache_bundle), intent(inout)       :: runtime_cache
    type(runtime_optimization_store), intent(inout) :: optimization_store
    type(model_state), intent(in)                   :: model
    type(session_state), intent(in)                 :: session
    integer(i64), intent(in)                        :: elapsed_us
    integer(i64), intent(out)                       :: plan_id
    integer(i32), intent(out)                       :: selection_mode
    integer(i32), intent(out)                       :: backend_family
    integer(i32), intent(out)                       :: execution_route
    integer(i64), intent(out)                       :: cache_flags
    type(model_manifest)                            :: manifest
    type(session_cache_key)                         :: optimization_key
    type(session_cache_key)                         :: candidate_key
    character(len=MAX_CACHE_KEY_LEN)                :: optimization_key_text
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_text
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_texts(3)
    integer(i64)                                    :: candidate_plan_ids(3)
    integer(i32)                                    :: candidate_backend_families(3)
    integer(i32)                                    :: candidate_execution_routes(3)
    integer(i32)                                    :: optimization_backend_family
    integer(i32)                                    :: candidate_count
    integer(i32)                                    :: candidate_index
    logical                                         :: was_hit
    logical                                         :: reused_winner

    call populate_manifest_identity(model, manifest)
    call enumerate_candidate_routes(model%info%allowed_backend_mask, candidate_backend_families, &
      candidate_execution_routes, candidate_count)
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)

    call build_session_cache_key(manifest, "unbound", optimization_backend_family, MIZU_EXEC_ROUTE_NONE, &
      session%config%max_context_tokens, session%config%max_decode_tokens, optimization_key)
    optimization_key_text = append_allowed_mask_identity(trim(optimization_key%key_text), &
      model%info%allowed_backend_mask)

    do candidate_index = 1_i32, candidate_count
      call build_session_cache_key(manifest, "unbound", candidate_backend_families(candidate_index), &
        candidate_execution_routes(candidate_index), session%config%max_context_tokens, &
        session%config%max_decode_tokens, candidate_key)
      candidate_key_texts(candidate_index) = trim(candidate_key%key_text)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_texts(candidate_index)))
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    call touch_session_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_SESSION_HIT, was_hit, reused_winner)
  end subroutine resolve_session_stage_cache

  subroutine persist_session_checkpoint(runtime, runtime_cache, model, session, checkpoint_ready)
    type(runtime_state), intent(in)           :: runtime
    type(runtime_cache_bundle), intent(inout) :: runtime_cache
    type(model_state), intent(in)             :: model
    type(session_state), intent(in)           :: session
    logical, intent(out)                      :: checkpoint_ready
    type(artifact_metadata_record)            :: metadata
    character(len=MAX_CACHE_KEY_LEN)          :: checkpoint_key_text
    character(len=4 * MAX_LIVE_CONTEXT_BYTES + 256) :: payload_text
    integer(i64)                             :: payload_bytes

    checkpoint_ready = .false.
    if (len_trim(runtime%config%cache_root) == 0) return
    if (session%live_context_byte_count <= 0_i32) return
    if (.not. backend_context_bytes_are_valid(session%live_context_backend_family, &
        session%live_context_execution_route, session%live_context_bytes, session%live_context_byte_count)) return

    call build_session_checkpoint_key(model, session, checkpoint_key_text)
    if (len_trim(checkpoint_key_text) == 0) return

    metadata = build_stage_artifact_metadata(MIZU_STAGE_PARK, session%live_context_backend_family, &
      session%live_context_execution_route, trim(checkpoint_key_text))
    call build_session_checkpoint_payload_text(session, payload_text, payload_bytes)
    call materialize_artifact_payload(runtime%config%cache_root, metadata, trim(payload_text), payload_bytes)
    call record_session_artifact_metadata(runtime_cache, trim(checkpoint_key_text), metadata)
    checkpoint_ready = metadata%is_materialized
  end subroutine persist_session_checkpoint

  subroutine restore_session_checkpoint(cache_root, runtime_cache, model, session, restored_ok)
    character(len=*), intent(in)              :: cache_root
    type(runtime_cache_bundle), intent(in)    :: runtime_cache
    type(model_state), intent(in)             :: model
    type(session_state), intent(inout)        :: session
    logical, intent(out)                      :: restored_ok
    type(artifact_metadata_record)            :: metadata
    character(len=MAX_CACHE_KEY_LEN)          :: checkpoint_key_text
    character(len=MAX_PATH_LEN)               :: full_path
    integer(i64)                              :: kv_token_count
    integer(i64)                              :: live_context_hash
    integer(i8)                               :: context_bytes(MAX_LIVE_CONTEXT_BYTES)
    integer(i64)                              :: context_artifact_hash
    integer(i32)                              :: backend_family
    integer(i32)                              :: execution_route
    integer(i32)                              :: context_byte_count
    integer(i32)                              :: context_producer_stage
    logical                                   :: found
    logical                                   :: lineage_known
    logical                                   :: loaded_ok

    restored_ok = .false.
    if (len_trim(cache_root) == 0) return
    if (session%live_context_backend_family == MIZU_BACKEND_FAMILY_NONE) return
    if (session%live_context_execution_route == MIZU_EXEC_ROUTE_NONE) return

    call build_session_checkpoint_key(model, session, checkpoint_key_text)
    if (len_trim(checkpoint_key_text) == 0) return

    call lookup_session_artifact_metadata(runtime_cache, trim(checkpoint_key_text), metadata, found)
    if (.not. found) return
    if (.not. metadata%is_materialized) return
    if (len_trim(metadata%payload_path) == 0) return

    full_path = join_cache_root_with_payload_path(cache_root, metadata%payload_path)
    if (len_trim(full_path) == 0) return

    call load_session_checkpoint_payload(trim(full_path), kv_token_count, live_context_hash, backend_family, &
      execution_route, context_bytes, context_byte_count, loaded_ok)
    if (.not. loaded_ok) return
    if (.not. backend_context_bytes_are_valid(backend_family, execution_route, context_bytes, context_byte_count)) return
    if (backend_family /= session%live_context_backend_family) return
    if (execution_route /= session%live_context_execution_route) return
    if (kv_token_count /= session%kv_token_count) return
    if (live_context_hash /= session%live_context_hash) return
    if (context_byte_count /= session%live_context_byte_count) return
    call extract_backend_context_lineage(backend_family, execution_route, context_bytes, context_byte_count, &
      context_producer_stage, context_artifact_hash, lineage_known)
    if (session%live_context_producer_stage /= MIZU_STAGE_NONE) then
      if (.not. lineage_known) return
      if (context_producer_stage /= session%live_context_producer_stage) return
    end if
    if (session%live_context_artifact_hash /= 0_i64) then
      if (.not. lineage_known) return
      if (context_artifact_hash /= session%live_context_artifact_hash) return
    end if

    session%kv_token_count = kv_token_count
    session%live_context_hash = live_context_hash
    session%has_live_context = .true.
    call store_live_context_record(session, backend_family, execution_route, context_bytes, context_byte_count, &
      producer_stage=context_producer_stage, artifact_hash=context_artifact_hash)
    restored_ok = .true.
  end subroutine restore_session_checkpoint

  subroutine build_session_checkpoint_key(model, session, checkpoint_key_text)
    type(model_state), intent(in)         :: model
    type(session_state), intent(in)       :: session
    character(len=*), intent(out)         :: checkpoint_key_text
    type(model_manifest)                  :: manifest
    type(session_cache_key)               :: checkpoint_key

    checkpoint_key_text = ""
    if (session%live_context_backend_family == MIZU_BACKEND_FAMILY_NONE) return
    if (session%live_context_execution_route == MIZU_EXEC_ROUTE_NONE) return

    call populate_manifest_identity(model, manifest)
    call build_session_cache_key(manifest, "checkpoint", session%live_context_backend_family, &
      session%live_context_execution_route, session%config%max_context_tokens, &
      session%config%max_decode_tokens, checkpoint_key)
    write(checkpoint_key_text, '(A,":ctx_hash=",I0,":kv=",I0,":ctx_bytes=",I0)') trim(checkpoint_key%key_text), &
      session%live_context_hash, session%kv_token_count, session%live_context_byte_count
  end subroutine build_session_checkpoint_key

  subroutine build_session_checkpoint_payload_text(session, payload_text, payload_bytes)
    type(session_state), intent(in)       :: session
    character(len=*), intent(out)         :: payload_text
    integer(i64), intent(out)             :: payload_bytes
    character(len=2 * MAX_LIVE_CONTEXT_BYTES) :: hex_text

    call encode_bytes_as_hex(session%live_context_bytes, session%live_context_byte_count, hex_text)
    payload_text = ""
    write(payload_text, '(I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,A)') session%live_context_backend_family, &
      session%live_context_execution_route, session%kv_token_count, session%live_context_hash, &
      session%live_context_byte_count, trim(hex_text)
    payload_bytes = int(len_trim(payload_text), kind=i64)
  end subroutine build_session_checkpoint_payload_text

  subroutine load_session_checkpoint_payload(file_path, kv_token_count, live_context_hash, backend_family, &
                                             execution_route, context_bytes, context_byte_count, loaded_ok)
    character(len=*), intent(in)          :: file_path
    integer(i64), intent(out)             :: kv_token_count
    integer(i64), intent(out)             :: live_context_hash
    integer(i32), intent(out)             :: backend_family
    integer(i32), intent(out)             :: execution_route
    integer(i8), intent(out)              :: context_bytes(:)
    integer(i32), intent(out)             :: context_byte_count
    logical, intent(out)                  :: loaded_ok
    character(len=4 * MAX_LIVE_CONTEXT_BYTES + 256) :: line
    character(len=2 * MAX_LIVE_CONTEXT_BYTES) :: hex_text
    integer(i32)                          :: unit_id
    integer(i32)                          :: ios
    logical                               :: exists

    kv_token_count = 0_i64
    live_context_hash = 0_i64
    backend_family = MIZU_BACKEND_FAMILY_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE
    context_bytes = 0_i8
    context_byte_count = 0_i32
    loaded_ok = .false.

    inquire(file=trim(file_path), exist=exists)
    if (.not. exists) return

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) return
    read(unit_id, "(A)", iostat=ios) line
    close(unit_id)
    if (ios /= 0_i32) return

    hex_text = ""
    read(line, *, iostat=ios) backend_family, execution_route, kv_token_count, live_context_hash, &
      context_byte_count, hex_text
    if (ios /= 0_i32) return

    call decode_hex_to_bytes(trim(hex_text), context_byte_count, context_bytes, loaded_ok)
  end subroutine load_session_checkpoint_payload

  pure logical function backend_context_bytes_are_valid(backend_family, execution_route, context_bytes, &
                                                        context_byte_count) result(is_valid)
    integer(i32), intent(in) :: backend_family
    integer(i32), intent(in) :: execution_route
    integer(i8), intent(in)  :: context_bytes(:)
    integer(i32), intent(in) :: context_byte_count

    is_valid = .false.
    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      if (execution_route /= MIZU_EXEC_ROUTE_ANE .and. execution_route /= MIZU_EXEC_ROUTE_METAL) return
      is_valid = apple_context_bytes_are_valid(context_bytes, context_byte_count)
    case (MIZU_BACKEND_FAMILY_CUDA)
      if (execution_route /= MIZU_EXEC_ROUTE_CUDA) return
      is_valid = cuda_context_bytes_are_valid(context_bytes, context_byte_count)
    end select
  end function backend_context_bytes_are_valid

  pure subroutine extract_backend_context_lineage(backend_family, execution_route, context_bytes, &
                                                  context_byte_count, producer_stage, artifact_hash, &
                                                  lineage_known)
    integer(i32), intent(in)  :: backend_family
    integer(i32), intent(in)  :: execution_route
    integer(i8), intent(in)   :: context_bytes(:)
    integer(i32), intent(in)  :: context_byte_count
    integer(i32), intent(out) :: producer_stage
    integer(i64), intent(out) :: artifact_hash
    logical, intent(out)      :: lineage_known
    integer(i32)              :: context_route

    producer_stage = MIZU_STAGE_NONE
    artifact_hash = 0_i64
    lineage_known = .false.
    context_route = MIZU_EXEC_ROUTE_NONE

    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      call extract_apple_context_lineage(context_bytes, context_byte_count, producer_stage, context_route, &
        artifact_hash, lineage_known)
      if (lineage_known) lineage_known = (context_route == execution_route)
    case (MIZU_BACKEND_FAMILY_CUDA)
      call extract_cuda_context_lineage(context_bytes, context_byte_count, producer_stage, artifact_hash, &
        lineage_known)
      if (lineage_known) lineage_known = (execution_route == MIZU_EXEC_ROUTE_CUDA)
    end select
  end subroutine extract_backend_context_lineage

  subroutine encode_bytes_as_hex(byte_values, byte_count, hex_text)
    integer(i8), intent(in)               :: byte_values(:)
    integer(i32), intent(in)              :: byte_count
    character(len=*), intent(out)         :: hex_text
    character(len=16), parameter          :: HEX_DIGITS = "0123456789ABCDEF"
    integer(i32)                          :: byte_index
    integer(i32)                          :: encoded_count
    integer(i32)                          :: byte_value

    hex_text = ""
    encoded_count = max(0_i32, min(byte_count, min(int(size(byte_values), kind=i32), len(hex_text) / 2)))
    do byte_index = 1_i32, encoded_count
      byte_value = int(byte_values(byte_index), kind=i32)
      if (byte_value < 0_i32) byte_value = byte_value + 256_i32
      hex_text((2 * byte_index) - 1:(2 * byte_index) - 1) = HEX_DIGITS((byte_value / 16_i32) + 1:(byte_value / 16_i32) + 1)
      hex_text(2 * byte_index:2 * byte_index) = HEX_DIGITS(mod(byte_value, 16_i32) + 1:mod(byte_value, 16_i32) + 1)
    end do
  end subroutine encode_bytes_as_hex

  subroutine decode_hex_to_bytes(hex_text, byte_count, byte_values, decoded_ok)
    character(len=*), intent(in)          :: hex_text
    integer(i32), intent(in)              :: byte_count
    integer(i8), intent(out)              :: byte_values(:)
    logical, intent(out)                  :: decoded_ok
    integer(i32)                          :: byte_index
    integer(i32)                          :: decoded_count
    integer(i32)                          :: upper_nibble
    integer(i32)                          :: lower_nibble
    integer(i32)                          :: byte_value

    byte_values = 0_i8
    decoded_ok = .false.
    decoded_count = max(0_i32, min(byte_count, min(int(size(byte_values), kind=i32), len_trim(hex_text) / 2)))
    if (decoded_count <= 0_i32) then
      decoded_ok = (byte_count <= 0_i32)
      return
    end if
    if ((2 * decoded_count) > len_trim(hex_text)) return

    do byte_index = 1_i32, decoded_count
      upper_nibble = hex_digit_value(hex_text((2 * byte_index) - 1:(2 * byte_index) - 1))
      lower_nibble = hex_digit_value(hex_text(2 * byte_index:2 * byte_index))
      if (upper_nibble < 0_i32 .or. lower_nibble < 0_i32) return

      byte_value = (16_i32 * upper_nibble) + lower_nibble
      if (byte_value > 127_i32) then
        byte_values(byte_index) = int(byte_value - 256_i32, kind=i8)
      else
        byte_values(byte_index) = int(byte_value, kind=i8)
      end if
    end do

    decoded_ok = .true.
  end subroutine decode_hex_to_bytes

  pure integer(i32) function hex_digit_value(hex_char) result(digit_value)
    character(len=*), intent(in) :: hex_char
    integer(i32)                 :: ascii_code

    digit_value = -1_i32
    if (len_trim(hex_char) <= 0) return

    ascii_code = iachar(hex_char(1:1))
    select case (ascii_code)
    case (iachar("0"):iachar("9"))
      digit_value = ascii_code - iachar("0")
    case (iachar("A"):iachar("F"))
      digit_value = 10_i32 + ascii_code - iachar("A")
    case (iachar("a"):iachar("f"))
      digit_value = 10_i32 + ascii_code - iachar("a")
    case default
      digit_value = -1_i32
    end select
  end function hex_digit_value

  subroutine prepare_plan_stage_candidate(runtime, optimization_store, model, stage_kind, op_family, &
                                          shape, token_count, optimization_key_text, &
                                          candidate_key_text, plan_id, selection_mode, &
                                          backend_family, execution_route, artifact_metadata)
    type(runtime_state), intent(in)                  :: runtime
    type(runtime_optimization_store), intent(in)     :: optimization_store
    type(model_state), intent(in)                    :: model
    integer(i32), intent(in)                         :: stage_kind
    integer(i32), intent(in)                         :: op_family
    integer(i64), intent(in)                         :: shape(3)
    integer(i64), intent(in)                         :: token_count
    character(len=*), intent(out)                    :: optimization_key_text
    character(len=*), intent(out)                    :: candidate_key_text
    integer(i64), intent(out)                        :: plan_id
    integer(i32), intent(out)                        :: selection_mode
    integer(i32), intent(out)                        :: backend_family
    integer(i32), intent(out)                        :: execution_route
    type(artifact_metadata_record), intent(out)      :: artifact_metadata
    type(model_manifest)                             :: manifest
    type(plan_request)                               :: stage_request
    type(plan_cache_key)                             :: optimization_key
    type(plan_cache_key)                             :: candidate_key
    character(len=MAX_CACHE_KEY_LEN)                 :: candidate_key_texts(3)
    integer(i64)                                     :: candidate_plan_ids(3)
    integer(i32)                                     :: candidate_backend_families(3)
    integer(i32)                                     :: candidate_execution_routes(3)
    integer(i32)                                     :: optimization_backend_family
    integer(i32)                                     :: candidate_count
    integer(i32)                                     :: candidate_index

    call populate_manifest_identity(model, manifest)
    call enumerate_candidate_routes(model%info%allowed_backend_mask, candidate_backend_families, &
      candidate_execution_routes, candidate_count)
    optimization_key_text = ""
    candidate_key_text = ""
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    optimization_backend_family = derive_optimization_backend_family(candidate_backend_families, candidate_count)

    call initialize_plan_request(stage_request, stage_kind, op_family, model%info%model_family, &
      model%info%allowed_backend_mask)
    stage_request%shape_signature = 0_i64
    stage_request%shape_signature(1:3) = shape
    stage_request%token_count = max(0_i64, token_count)
    stage_request%planner_version_hint = 1_i64

    call build_plan_cache_key(manifest, "unbound", "logical", stage_kind, optimization_backend_family, &
      MIZU_EXEC_ROUTE_NONE, MIZU_DTYPE_BF16, 3_i32, shape, optimization_key)
    optimization_key_text = append_allowed_mask_identity(trim(optimization_key%key_text), &
      model%info%allowed_backend_mask)

    do candidate_index = 1_i32, candidate_count
      call build_plan_cache_key(manifest, "unbound", "logical", stage_kind, &
        candidate_backend_families(candidate_index), candidate_execution_routes(candidate_index), &
        MIZU_DTYPE_BF16, 3_i32, shape, candidate_key)
      candidate_key_text = trim(candidate_key%key_text)
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_text))
      candidate_key_texts(candidate_index) = trim(candidate_key_text)
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    artifact_metadata = build_stage_artifact_metadata(stage_kind, backend_family, execution_route, &
      trim(candidate_key_text), stage_request, runtime%config%cache_root)
  end subroutine prepare_plan_stage_candidate

  subroutine finalize_plan_stage_cache(runtime_cache, optimization_store, optimization_key_text, &
                                       candidate_key_text, plan_id, selection_mode, elapsed_us, &
                                       artifact_metadata, cache_flags)
    type(runtime_cache_bundle), intent(inout)       :: runtime_cache
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=*), intent(in)                    :: optimization_key_text
    character(len=*), intent(in)                    :: candidate_key_text
    integer(i64), intent(in)                        :: plan_id
    integer(i32), intent(in)                        :: selection_mode
    integer(i64), intent(in)                        :: elapsed_us
    type(artifact_metadata_record), intent(in)      :: artifact_metadata
    integer(i64), intent(out)                       :: cache_flags
    logical                                         :: was_hit
    logical                                         :: reused_winner

    call touch_plan_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    call record_plan_artifact_metadata(runtime_cache, trim(candidate_key_text), artifact_metadata)
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_PLAN_HIT, was_hit, reused_winner)
  end subroutine finalize_plan_stage_cache

  subroutine reserve_stage_workspace(runtime, artifact_metadata, was_reserved, status_code)
    type(runtime_state), intent(inout)             :: runtime
    type(artifact_metadata_record), intent(in)     :: artifact_metadata
    logical, intent(out)                           :: was_reserved
    integer(i32), intent(out)                      :: status_code

    was_reserved = .false.
    status_code = MIZU_STATUS_OK
    if (artifact_metadata%workspace_bytes <= 0_i64) return

    call reserve_workspace_bytes(runtime%workspace, artifact_metadata%workspace_bytes, status_code)
    if (status_code /= MIZU_STATUS_OK) then
      call set_runtime_error(runtime, status_code, "workspace reservation failed")
      return
    end if

    was_reserved = .true.
  end subroutine reserve_stage_workspace

  subroutine release_stage_workspace(runtime, was_reserved)
    type(runtime_state), intent(inout) :: runtime
    logical, intent(in)                :: was_reserved

    if (.not. was_reserved) return
    call release_workspace_bytes(runtime%workspace)
  end subroutine release_stage_workspace

  subroutine prepare_projector_stage_candidate(runtime, optimization_store, model, staged_modal_byte_count, &
                                               staged_modal_kind, staged_modal_dtype, staged_modal_slot_name, &
                                               optimization_key_text, candidate_key_text, plan_id, &
                                               selection_mode, backend_family, execution_route, &
                                               artifact_metadata, placeholder_count)
    type(runtime_state), intent(in)                 :: runtime
    type(runtime_optimization_store), intent(in)    :: optimization_store
    type(model_state), intent(in)                   :: model
    integer(i64), intent(in)                        :: staged_modal_byte_count
    integer(i32), intent(in)                        :: staged_modal_kind
    integer(i32), intent(in)                        :: staged_modal_dtype
    character(len=*), intent(in)                    :: staged_modal_slot_name
    character(len=*), intent(out)                   :: optimization_key_text
    character(len=*), intent(out)                   :: candidate_key_text
    integer(i64), intent(out)                       :: plan_id
    integer(i32), intent(out)                       :: selection_mode
    integer(i32), intent(out)                       :: backend_family
    integer(i32), intent(out)                       :: execution_route
    type(artifact_metadata_record), intent(out)     :: artifact_metadata
    integer(i32), intent(out)                       :: placeholder_count
    type(plan_request)                              :: stage_request
    type(model_manifest)                            :: manifest
    type(multimodal_cache_key)                      :: key
    integer(i32)                                    :: modality_kind
    integer(i32)                                    :: modality_dtype
    character(len=MAX_PATH_LEN)                     :: slot_name
    character(len=MAX_CACHE_KEY_LEN)                :: candidate_key_texts(3)
    integer(i64)                                    :: candidate_plan_ids(3)
    integer(i32)                                    :: candidate_backend_families(3)
    integer(i32)                                    :: candidate_execution_routes(3)
    integer(i32)                                    :: candidate_count
    integer(i32)                                    :: candidate_index

    call populate_manifest_identity(model, manifest)
    slot_name = trim(staged_modal_slot_name)
    if (len_trim(slot_name) == 0) slot_name = "image"
    modality_kind = staged_modal_kind
    if (modality_kind <= 0_i32) modality_kind = MIZU_MODALITY_KIND_IMAGE
    modality_dtype = staged_modal_dtype
    if (modality_dtype <= 0_i32) modality_dtype = MIZU_DTYPE_U8
    call initialize_plan_request(stage_request, MIZU_STAGE_PROJECTOR, OP_FAMILY_PROJECTOR, &
      model%info%model_family, model%info%allowed_backend_mask)
    stage_request%shape_signature = 0_i64
    stage_request%shape_signature(1) = max(0_i64, staged_modal_byte_count)
    stage_request%shape_signature(2) = int(modality_kind, kind=i64)
    stage_request%shape_signature(3) = int(modality_dtype, kind=i64)
    stage_request%planner_version_hint = 1_i64
    stage_request%projector%is_present = manifest%projector%is_present
    stage_request%projector%placeholder_count = manifest%projector%placeholder_count
    stage_request%projector%input_dtype = manifest%projector%input_dtype
    stage_request%projector%embedding_dtype = manifest%projector%embedding_dtype
    stage_request%projector%slot_name = manifest%projector%slot_name

    call enumerate_candidate_routes(model%info%allowed_backend_mask, candidate_backend_families, &
      candidate_execution_routes, candidate_count)
    optimization_key_text = ""
    candidate_key_text = ""
    candidate_key_texts = ""
    candidate_plan_ids = 0_i64
    call build_multimodal_cache_key(manifest, "unbound", trim(slot_name), modality_kind, &
      modality_dtype, max(0_i64, staged_modal_byte_count), key)
    optimization_key_text = append_allowed_mask_identity(trim(key%key_text), model%info%allowed_backend_mask)

    do candidate_index = 1_i32, candidate_count
      candidate_key_texts(candidate_index) = append_route_identity(trim(key%key_text), &
        candidate_backend_families(candidate_index), candidate_execution_routes(candidate_index))
      candidate_plan_ids(candidate_index) = hash_text64(trim(candidate_key_texts(candidate_index)))
    end do

    call resolve_stage_candidate(runtime, optimization_store, trim(optimization_key_text), candidate_count, &
      candidate_backend_families, candidate_execution_routes, candidate_plan_ids, candidate_key_texts, &
      candidate_key_text, plan_id, selection_mode, backend_family, execution_route)
    artifact_metadata = build_stage_artifact_metadata(MIZU_STAGE_PROJECTOR, backend_family, execution_route, &
      trim(candidate_key_text), stage_request, runtime%config%cache_root)
    placeholder_count = max(1_i32, stage_request%projector%placeholder_count)
  end subroutine prepare_projector_stage_candidate

  subroutine finalize_projector_stage_cache(runtime_cache, optimization_store, optimization_key_text, &
                                            candidate_key_text, plan_id, selection_mode, elapsed_us, &
                                            artifact_metadata, cache_flags)
    type(runtime_cache_bundle), intent(inout)       :: runtime_cache
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=*), intent(in)                    :: optimization_key_text
    character(len=*), intent(in)                    :: candidate_key_text
    integer(i64), intent(in)                        :: plan_id
    integer(i32), intent(in)                        :: selection_mode
    integer(i64), intent(in)                        :: elapsed_us
    type(artifact_metadata_record), intent(in)      :: artifact_metadata
    integer(i64), intent(out)                       :: cache_flags
    logical                                         :: was_hit
    logical                                         :: reused_winner

    call touch_multimodal_cache_key(runtime_cache, trim(candidate_key_text), was_hit)
    call record_multimodal_artifact_metadata(runtime_cache, trim(candidate_key_text), artifact_metadata)
    reused_winner = (selection_mode == MIZU_SELECTION_MODE_REUSE)
    call record_execution_sample(optimization_store, trim(optimization_key_text), plan_id, elapsed_us, &
      trim(candidate_key_text))
    cache_flags = compose_cache_flags(MIZU_CACHE_FLAG_MM_HIT, was_hit, reused_winner)
  end subroutine finalize_projector_stage_cache

  subroutine resolve_stage_candidate(runtime, optimization_store, optimization_key_text, &
                                     candidate_count, candidate_backend_families, &
                                     candidate_execution_routes, candidate_plan_ids, &
                                     candidate_key_texts, candidate_key_text, plan_id, &
                                     selection_mode, backend_family, execution_route)
    type(runtime_state), intent(in)                :: runtime
    type(runtime_optimization_store), intent(in)   :: optimization_store
    character(len=*), intent(in)                   :: optimization_key_text
    integer(i32), intent(in)                       :: candidate_count
    integer(i32), intent(in)                       :: candidate_backend_families(:)
    integer(i32), intent(in)                       :: candidate_execution_routes(:)
    integer(i64), intent(in)                       :: candidate_plan_ids(:)
    character(len=*), intent(in)                   :: candidate_key_texts(:)
    character(len=*), intent(out)                  :: candidate_key_text
    integer(i64), intent(out)                      :: plan_id
    integer(i32), intent(out)                      :: selection_mode
    integer(i32), intent(out)                      :: backend_family
    integer(i32), intent(out)                      :: execution_route
    character(len=MAX_CACHE_KEY_LEN)               :: winner_candidate_key_text
    integer(i64)                                   :: total_samples
    integer(i64)                                   :: winner_plan_id
    integer(i32)                                   :: candidate_budget
    integer(i32)                                   :: candidate_index
    integer(i32)                                   :: observed_candidate_count
    integer(i32)                                   :: winner_index
    logical                                        :: has_winner

    candidate_key_text = ""
    plan_id = 0_i64
    selection_mode = MIZU_SELECTION_MODE_NONE
    backend_family = MIZU_BACKEND_FAMILY_NONE
    execution_route = MIZU_EXEC_ROUTE_NONE

    if (candidate_count <= 0_i32) return
    call assign_stage_candidate(1_i32, candidate_backend_families, candidate_execution_routes, &
      candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
    selection_mode = MIZU_SELECTION_MODE_DIRECT
    if (len_trim(optimization_key_text) == 0) return

    if (runtime%config%optimization_mode == MIZU_OPTIMIZATION_MODE_DISABLED) return

    candidate_budget = min(max(1_i32, runtime%config%exploration_budget), candidate_count)
    call lookup_optimization_entry_stats(optimization_store, trim(optimization_key_text), &
      total_samples, observed_candidate_count)
    call lookup_winner_candidate(optimization_store, trim(optimization_key_text), winner_plan_id, &
      winner_candidate_key_text, has_winner)
    winner_index = find_candidate_index(candidate_key_texts, candidate_plan_ids, candidate_count, &
      winner_candidate_key_text, winner_plan_id)

    if (candidate_budget <= 1_i32 .or. candidate_count <= 1_i32) then
      if (has_winner .and. winner_index > 0_i32) then
        call assign_stage_candidate(winner_index, candidate_backend_families, candidate_execution_routes, &
          candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
        selection_mode = MIZU_SELECTION_MODE_REUSE
      end if
      return
    end if

    if (has_winner .and. winner_index > 0_i32 .and. total_samples >= int(candidate_budget, kind=i64) .and. &
        observed_candidate_count >= candidate_budget) then
      call assign_stage_candidate(winner_index, candidate_backend_families, candidate_execution_routes, &
        candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
      selection_mode = MIZU_SELECTION_MODE_REUSE
      return
    end if

    candidate_index = 1_i32 + mod(int(total_samples, kind=i32), candidate_budget)
    call assign_stage_candidate(candidate_index, candidate_backend_families, candidate_execution_routes, &
      candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, backend_family, execution_route)
    selection_mode = MIZU_SELECTION_MODE_EXPLORATORY
  end subroutine resolve_stage_candidate

  pure integer(i64) function compose_cache_flags(hit_flag, was_hit, reused_winner) result(cache_flags)
    integer(i64), intent(in) :: hit_flag
    logical, intent(in)      :: was_hit
    logical, intent(in)      :: reused_winner

    cache_flags = MIZU_CACHE_FLAG_NONE
    if (was_hit) cache_flags = ior(cache_flags, hit_flag)
    if (reused_winner) cache_flags = ior(cache_flags, MIZU_CACHE_FLAG_WINNER_REUSED)
  end function compose_cache_flags

  pure subroutine enumerate_candidate_routes(backend_mask, backend_families, execution_routes, candidate_count)
    integer(i64), intent(in)  :: backend_mask
    integer(i32), intent(out) :: backend_families(:)
    integer(i32), intent(out) :: execution_routes(:)
    integer(i32), intent(out) :: candidate_count

    backend_families = MIZU_BACKEND_FAMILY_NONE
    execution_routes = MIZU_EXEC_ROUTE_NONE
    candidate_count = 0_i32

    if (iand(backend_mask, MIZU_BACKEND_MASK_APPLE_ANE) /= 0_i64) then
      candidate_count = candidate_count + 1_i32
      backend_families(candidate_count) = MIZU_BACKEND_FAMILY_APPLE
      execution_routes(candidate_count) = MIZU_EXEC_ROUTE_ANE
    end if
    if (iand(backend_mask, MIZU_BACKEND_MASK_APPLE_METAL) /= 0_i64) then
      candidate_count = candidate_count + 1_i32
      backend_families(candidate_count) = MIZU_BACKEND_FAMILY_APPLE
      execution_routes(candidate_count) = MIZU_EXEC_ROUTE_METAL
    end if
    if (iand(backend_mask, MIZU_BACKEND_MASK_CUDA) /= 0_i64) then
      candidate_count = candidate_count + 1_i32
      backend_families(candidate_count) = MIZU_BACKEND_FAMILY_CUDA
      execution_routes(candidate_count) = MIZU_EXEC_ROUTE_CUDA
    end if
  end subroutine enumerate_candidate_routes

  pure integer(i32) function derive_optimization_backend_family(candidate_backend_families, candidate_count) &
      result(backend_family)
    integer(i32), intent(in) :: candidate_backend_families(:)
    integer(i32), intent(in) :: candidate_count
    integer(i32)             :: candidate_index

    backend_family = MIZU_BACKEND_FAMILY_NONE
    if (candidate_count <= 0_i32) return

    backend_family = candidate_backend_families(1)
    do candidate_index = 2_i32, candidate_count
      if (candidate_backend_families(candidate_index) /= backend_family) then
        backend_family = MIZU_BACKEND_FAMILY_NONE
        return
      end if
    end do
  end function derive_optimization_backend_family

  function append_allowed_mask_identity(base_key_text, allowed_backend_mask) result(optimization_key_text)
    character(len=*), intent(in) :: base_key_text
    integer(i64), intent(in)     :: allowed_backend_mask
    character(len=MAX_CACHE_KEY_LEN) :: optimization_key_text

    optimization_key_text = ""
    if (len_trim(base_key_text) == 0) return

    write(optimization_key_text, '(A,":allowmask=",I0)') trim(base_key_text), allowed_backend_mask
  end function append_allowed_mask_identity

  function append_route_identity(base_key_text, backend_family, execution_route) result(candidate_key_text)
    character(len=*), intent(in) :: base_key_text
    integer(i32), intent(in)     :: backend_family
    integer(i32), intent(in)     :: execution_route
    character(len=MAX_CACHE_KEY_LEN) :: candidate_key_text

    candidate_key_text = ""
    if (len_trim(base_key_text) == 0) return

    write(candidate_key_text, '(A,":candidate_backend=",I0,":candidate_route=",I0)') &
      trim(base_key_text), backend_family, execution_route
  end function append_route_identity

  function build_stage_artifact_metadata(stage_kind, backend_family, execution_route, candidate_key_text, &
                                         request, cache_root) result(metadata)
    integer(i32), intent(in)      :: stage_kind
    integer(i32), intent(in)      :: backend_family
    integer(i32), intent(in)      :: execution_route
    character(len=*), intent(in)  :: candidate_key_text
    type(plan_request), intent(in), optional :: request
    character(len=*), intent(in), optional   :: cache_root
    type(artifact_metadata_record) :: metadata
    type(planner_result)           :: planning_result
    type(plan_request)             :: planning_request
    character(len=MAX_NAME_LEN)    :: fingerprint_token
    character(len=max(APPLE_ARTIFACT_PAYLOAD_LEN, CUDA_ARTIFACT_PAYLOAD_LEN)) :: payload_text
    integer(i64)                   :: payload_bytes
    integer(i32)                   :: status_code

    metadata = artifact_metadata_record()
    metadata%backend_family = backend_family
    metadata%execution_route = execution_route
    metadata%stage_kind = stage_kind
    metadata%is_materialized = .false.
    metadata%payload_bytes = 0_i64
    metadata%workspace_bytes = 0_i64
    metadata%artifact_format = build_artifact_format_label(stage_kind, backend_family, execution_route)
    fingerprint_token = build_artifact_fingerprint_token(trim(candidate_key_text))
    metadata%payload_fingerprint = trim(fingerprint_token)
    metadata%payload_path = build_artifact_payload_path(stage_kind, backend_family, execution_route, &
      trim(fingerprint_token))

    if (.not. present(request)) return
    planning_request = request

    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      planning_request%preferred_backend_mask = ior(planning_request%preferred_backend_mask, MIZU_BACKEND_MASK_APPLE_ANE)
    case (MIZU_EXEC_ROUTE_METAL)
      planning_request%preferred_backend_mask = ior(planning_request%preferred_backend_mask, MIZU_BACKEND_MASK_APPLE_METAL)
    case (MIZU_EXEC_ROUTE_CUDA)
      planning_request%preferred_backend_mask = ior(planning_request%preferred_backend_mask, MIZU_BACKEND_MASK_CUDA)
    end select

    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      if (execution_route /= MIZU_EXEC_ROUTE_ANE .and. execution_route /= MIZU_EXEC_ROUTE_METAL) return
      call plan_apple_stage(planning_request, planning_result, status_code)
      if (status_code /= MIZU_STATUS_OK) return
      if (.not. planner_result_is_success(planning_result)) return

      metadata%artifact_format = trim(planning_result%chosen_plan%pack_format)
      metadata%workspace_bytes = max(0_i64, planning_result%chosen_plan%workspace_bytes)
      call build_apple_artifact_payload_text(planning_request, planning_result%chosen_plan, trim(candidate_key_text), &
        payload_text, payload_bytes)
      if (present(cache_root)) then
        call materialize_artifact_payload(trim(cache_root), metadata, trim(payload_text), payload_bytes)
      end if
    case (MIZU_BACKEND_FAMILY_CUDA)
      if (execution_route /= MIZU_EXEC_ROUTE_CUDA) return
      call plan_cuda_stage(planning_request, planning_result, status_code)
      if (status_code /= MIZU_STATUS_OK) return
      if (.not. planner_result_is_success(planning_result)) return

      metadata%artifact_format = trim(planning_result%chosen_plan%pack_format)
      metadata%workspace_bytes = max(0_i64, planning_result%chosen_plan%workspace_bytes)
      call build_cuda_artifact_payload_text(planning_request, planning_result%chosen_plan, trim(candidate_key_text), &
        payload_text, payload_bytes)
      if (present(cache_root)) then
        call materialize_artifact_payload(trim(cache_root), metadata, trim(payload_text), payload_bytes)
      end if
    end select
  end function build_stage_artifact_metadata

  function build_artifact_format_label(stage_kind, backend_family, execution_route) result(format_label)
    integer(i32), intent(in)    :: stage_kind
    integer(i32), intent(in)    :: backend_family
    integer(i32), intent(in)    :: execution_route
    character(len=MAX_NAME_LEN) :: format_label
    character(len=MAX_NAME_LEN) :: family_token
    character(len=MAX_NAME_LEN) :: route_token
    character(len=MAX_NAME_LEN) :: stage_token

    format_label = ""
    family_token = artifact_family_token(backend_family)
    route_token = artifact_route_token(execution_route)
    stage_token = artifact_stage_token(stage_kind)
    write(format_label, '(A,"_",A,"_",A,"_v1")') trim(family_token), trim(route_token), trim(stage_token)
  end function build_artifact_format_label

  function build_artifact_payload_path(stage_kind, backend_family, execution_route, fingerprint_token) &
      result(payload_path)
    integer(i32), intent(in)    :: stage_kind
    integer(i32), intent(in)    :: backend_family
    integer(i32), intent(in)    :: execution_route
    character(len=*), intent(in) :: fingerprint_token
    character(len=MAX_PATH_LEN) :: payload_path
    character(len=MAX_NAME_LEN) :: family_token
    character(len=MAX_NAME_LEN) :: route_token

    payload_path = ""
    family_token = artifact_family_token(backend_family)
    route_token = artifact_route_token(execution_route)

    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      write(payload_path, '(A,"/",A,"/",A,"/weights/",A,".pack")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_PROJECTOR)
      write(payload_path, '(A,"/",A,"/",A,"/projector/",A,".mm")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_PREFILL)
      write(payload_path, '(A,"/",A,"/",A,"/plans/prefill/",A,".plan")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_DECODE)
      write(payload_path, '(A,"/",A,"/",A,"/plans/decode/",A,".plan")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case (MIZU_STAGE_PARK, MIZU_STAGE_RESUME)
      write(payload_path, '(A,"/",A,"/",A,"/sessions/",A,".session")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    case default
      write(payload_path, '(A,"/",A,"/",A,"/misc/",A,".artifact")') &
        "artifacts", trim(family_token), trim(route_token), trim(fingerprint_token)
    end select
  end function build_artifact_payload_path

  function build_artifact_fingerprint_token(candidate_key_text) result(fingerprint_token)
    character(len=*), intent(in) :: candidate_key_text
    character(len=MAX_NAME_LEN)  :: fingerprint_token
    integer(i64)                 :: key_hash

    fingerprint_token = ""
    if (len_trim(candidate_key_text) == 0) return

    key_hash = hash_text64(trim(candidate_key_text))
    write(fingerprint_token, '(Z16.16)') key_hash
  end function build_artifact_fingerprint_token

  subroutine materialize_artifact_payload(cache_root, metadata, payload_text, payload_bytes)
    character(len=*), intent(in)              :: cache_root
    type(artifact_metadata_record), intent(inout) :: metadata
    character(len=*), intent(in)              :: payload_text
    integer(i64), intent(in)                  :: payload_bytes
    character(len=MAX_PATH_LEN)               :: full_path
    character(len=MAX_PATH_LEN)               :: parent_dir
    integer(i64)                              :: existing_size
    integer(i32)                              :: unit_id
    integer(i32)                              :: ios
    logical                                   :: exists

    if (len_trim(cache_root) == 0) return
    if (len_trim(metadata%payload_path) == 0) return

    full_path = join_cache_root_with_payload_path(cache_root, metadata%payload_path)
    inquire(file=trim(full_path), exist=exists, size=existing_size)
    if (exists) then
      metadata%is_materialized = .true.
      metadata%payload_bytes = max(1_i64, existing_size)
      return
    end if

    parent_dir = parent_directory_path(full_path)
    if (len_trim(parent_dir) > 0) call ensure_directory_exists(parent_dir)

    open(newunit=unit_id, file=trim(full_path), status="replace", action="write", iostat=ios)
    if (ios /= 0_i32) return
    write(unit_id, "(A)", iostat=ios) trim(payload_text)
    close(unit_id)
    if (ios /= 0_i32) return

    metadata%is_materialized = .true.
    metadata%payload_bytes = max(1_i64, payload_bytes)
  end subroutine materialize_artifact_payload

  function join_cache_root_with_payload_path(cache_root, payload_path) result(full_path)
    character(len=*), intent(in) :: cache_root
    character(len=*), intent(in) :: payload_path
    character(len=MAX_PATH_LEN)  :: full_path
    integer                      :: root_len

    full_path = ""
    if (len_trim(cache_root) == 0 .or. len_trim(payload_path) == 0) return

    root_len = len_trim(cache_root)
    if (cache_root(root_len:root_len) == "/") then
      full_path = trim(cache_root) // trim(payload_path)
    else
      full_path = trim(cache_root) // "/" // trim(payload_path)
    end if
  end function join_cache_root_with_payload_path

  function parent_directory_path(file_path) result(parent_path)
    character(len=*), intent(in) :: file_path
    character(len=MAX_PATH_LEN)  :: parent_path
    integer                      :: index_char

    parent_path = ""
    do index_char = len_trim(file_path), 1, -1
      if (file_path(index_char:index_char) == "/") then
        if (index_char > 1) then
          parent_path = file_path(1:index_char-1)
        end if
        return
      end if
    end do
  end function parent_directory_path

  pure function artifact_family_token(backend_family) result(family_token)
    integer(i32), intent(in)    :: backend_family
    character(len=MAX_NAME_LEN) :: family_token

    select case (backend_family)
    case (MIZU_BACKEND_FAMILY_APPLE)
      family_token = "apple"
    case (MIZU_BACKEND_FAMILY_CUDA)
      family_token = "cuda"
    case default
      family_token = "generic"
    end select
  end function artifact_family_token

  pure function artifact_route_token(execution_route) result(route_token)
    integer(i32), intent(in)    :: execution_route
    character(len=MAX_NAME_LEN) :: route_token

    select case (execution_route)
    case (MIZU_EXEC_ROUTE_ANE)
      route_token = "ane"
    case (MIZU_EXEC_ROUTE_METAL)
      route_token = "metal"
    case (MIZU_EXEC_ROUTE_CUDA)
      route_token = "cuda"
    case default
      route_token = "generic"
    end select
  end function artifact_route_token

  pure function artifact_stage_token(stage_kind) result(stage_token)
    integer(i32), intent(in)    :: stage_kind
    character(len=MAX_NAME_LEN) :: stage_token

    select case (stage_kind)
    case (MIZU_STAGE_MODEL_LOAD)
      stage_token = "weight_pack"
    case (MIZU_STAGE_PROJECTOR)
      stage_token = "projector_cache"
    case (MIZU_STAGE_PREFILL)
      stage_token = "prefill_plan"
    case (MIZU_STAGE_DECODE)
      stage_token = "decode_plan"
    case (MIZU_STAGE_PARK, MIZU_STAGE_RESUME)
      stage_token = "session_checkpoint"
    case default
      stage_token = "artifact"
    end select
  end function artifact_stage_token

  subroutine assign_stage_candidate(candidate_index, candidate_backend_families, candidate_execution_routes, &
                                    candidate_plan_ids, candidate_key_texts, candidate_key_text, plan_id, &
                                    backend_family, execution_route)
    integer(i32), intent(in)      :: candidate_index
    integer(i32), intent(in)      :: candidate_backend_families(:)
    integer(i32), intent(in)      :: candidate_execution_routes(:)
    integer(i64), intent(in)      :: candidate_plan_ids(:)
    character(len=*), intent(in)  :: candidate_key_texts(:)
    character(len=*), intent(out) :: candidate_key_text
    integer(i64), intent(out)     :: plan_id
    integer(i32), intent(out)     :: backend_family
    integer(i32), intent(out)     :: execution_route

    candidate_key_text = trim(candidate_key_texts(candidate_index))
    plan_id = candidate_plan_ids(candidate_index)
    backend_family = candidate_backend_families(candidate_index)
    execution_route = candidate_execution_routes(candidate_index)
  end subroutine assign_stage_candidate

  pure integer(i32) function find_candidate_index(candidate_key_texts, candidate_plan_ids, candidate_count, &
                                                  winner_candidate_key_text, winner_plan_id) result(candidate_index)
    character(len=*), intent(in) :: candidate_key_texts(:)
    integer(i64), intent(in)     :: candidate_plan_ids(:)
    integer(i32), intent(in)     :: candidate_count
    character(len=*), intent(in) :: winner_candidate_key_text
    integer(i64), intent(in)     :: winner_plan_id
    integer(i32)                 :: index

    candidate_index = 0_i32
    if (len_trim(winner_candidate_key_text) > 0) then
      do index = 1_i32, candidate_count
        if (trim(candidate_key_texts(index)) == trim(winner_candidate_key_text)) then
          candidate_index = index
          return
        end if
      end do
    end if

    if (winner_plan_id /= 0_i64) then
      do index = 1_i32, candidate_count
        if (candidate_plan_ids(index) == winner_plan_id) then
          candidate_index = index
          return
        end if
      end do
    end if
  end function find_candidate_index

  integer(i64) function monotonic_timestamp_us() result(timestamp_us)
    integer(i64) :: clock_count
    integer(i64) :: clock_rate

    call system_clock(clock_count, clock_rate)
    if (clock_rate <= 0_i64) then
      timestamp_us = 0_i64
      return
    end if

    timestamp_us = (clock_count * 1000000_i64) / clock_rate
  end function monotonic_timestamp_us

  integer(i64) function elapsed_since_us(started_us) result(elapsed_us)
    integer(i64), intent(in) :: started_us
    integer(i64)             :: finished_us

    if (started_us <= 0_i64) then
      elapsed_us = 1_i64
      return
    end if

    finished_us = monotonic_timestamp_us()
    elapsed_us = max(1_i64, finished_us - started_us)
  end function elapsed_since_us

  subroutine hydrate_runtime_cache_state(runtime, runtime_cache)
    type(runtime_state), intent(in)          :: runtime
    type(runtime_cache_bundle), intent(inout) :: runtime_cache
    character(len=MAX_PATH_LEN)              :: store_path
    logical                                  :: loaded_ok

    store_path = build_runtime_artifact_cache_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call load_runtime_cache_bundle(runtime_cache, trim(store_path), loaded_ok)
  end subroutine hydrate_runtime_cache_state

  subroutine persist_runtime_cache_state(runtime, runtime_cache)
    type(runtime_state), intent(in)         :: runtime
    type(runtime_cache_bundle), intent(in)  :: runtime_cache
    character(len=MAX_PATH_LEN)             :: store_path
    logical                                 :: saved_ok

    store_path = build_runtime_artifact_cache_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call ensure_directory_exists(runtime%config%cache_root)
    call save_runtime_cache_bundle(runtime_cache, trim(store_path), saved_ok)
  end subroutine persist_runtime_cache_state

  subroutine hydrate_runtime_optimization_state(runtime, optimization_store)
    type(runtime_state), intent(in)                :: runtime
    type(runtime_optimization_store), intent(inout) :: optimization_store
    character(len=MAX_PATH_LEN)                    :: store_path
    logical                                        :: loaded_ok

    store_path = build_runtime_optimization_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call load_runtime_optimization_store(optimization_store, trim(store_path), loaded_ok)
  end subroutine hydrate_runtime_optimization_state

  subroutine persist_runtime_optimization_state(runtime, optimization_store)
    type(runtime_state), intent(in)                :: runtime
    type(runtime_optimization_store), intent(in)   :: optimization_store
    character(len=MAX_PATH_LEN)                    :: store_path
    logical                                        :: saved_ok

    store_path = build_runtime_optimization_store_path(runtime%config%cache_root)
    if (len_trim(store_path) == 0) return

    call ensure_directory_exists(runtime%config%cache_root)
    call save_runtime_optimization_store(optimization_store, trim(store_path), saved_ok)
  end subroutine persist_runtime_optimization_state

  function build_runtime_artifact_cache_store_path(cache_root) result(store_path)
    character(len=*), intent(in) :: cache_root
    character(len=MAX_PATH_LEN)  :: store_path
    integer                      :: root_len

    store_path = ""
    root_len = len_trim(cache_root)
    if (root_len == 0) return

    if (cache_root(root_len:root_len) == "/") then
      store_path = trim(cache_root) // "artifact_cache_v1.txt"
    else
      store_path = trim(cache_root) // "/artifact_cache_v1.txt"
    end if
  end function build_runtime_artifact_cache_store_path

  function build_runtime_optimization_store_path(cache_root) result(store_path)
    character(len=*), intent(in) :: cache_root
    character(len=MAX_PATH_LEN)  :: store_path
    integer                      :: root_len

    store_path = ""
    root_len = len_trim(cache_root)
    if (root_len == 0) return

    if (cache_root(root_len:root_len) == "/") then
      store_path = trim(cache_root) // "optimization_store_v1.txt"
    else
      store_path = trim(cache_root) // "/optimization_store_v1.txt"
    end if
  end function build_runtime_optimization_store_path

  subroutine ensure_directory_exists(directory_path)
    character(len=*), intent(in) :: directory_path
    character(len=(MAX_PATH_LEN * 2) + 16) :: command_text
    integer :: exit_status

    if (len_trim(directory_path) == 0) return

    command_text = "mkdir -p " // trim(shell_quote_text(trim(directory_path)))
    call execute_command_line(trim(command_text), exitstat=exit_status)
  end subroutine ensure_directory_exists

  function shell_quote_text(text) result(quoted_text)
    character(len=*), intent(in) :: text
    character(len=(MAX_PATH_LEN * 2) + 16) :: quoted_text
    integer :: index_char
    integer :: cursor

    quoted_text = ""
    if (len_trim(text) == 0) return

    cursor = 1
    quoted_text(cursor:cursor) = '"'
    cursor = cursor + 1

    do index_char = 1, len_trim(text)
      select case (text(index_char:index_char))
      case ('\')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = "\\"
          cursor = cursor + 2
        end if
      case ('"')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = '\"'
          cursor = cursor + 2
        end if
      case ('$')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = "\$"
          cursor = cursor + 2
        end if
      case ('`')
        if (cursor + 1 <= len(quoted_text)) then
          quoted_text(cursor:cursor+1) = "\`"
          cursor = cursor + 2
        end if
      case default
        if (cursor <= len(quoted_text)) then
          quoted_text(cursor:cursor) = text(index_char:index_char)
          cursor = cursor + 1
        end if
      end select
    end do

    if (cursor <= len(quoted_text)) quoted_text(cursor:cursor) = '"'
  end function shell_quote_text

  function copy_c_string_ptr(string_ptr, default_text) result(text)
    type(c_ptr), value         :: string_ptr
    character(len=*), intent(in) :: default_text
    character(len=MAX_PATH_LEN)  :: text

    text = ""
    call copy_c_string_ptr_to_fortran(string_ptr, text)
    if (len_trim(text) == 0) then
      text = trim(default_text)
    end if
  end function copy_c_string_ptr

  subroutine copy_c_string_ptr_to_fortran(string_ptr, text)
    type(c_ptr), value       :: string_ptr
    character(len=*), intent(out) :: text
    character(kind=c_char), pointer :: chars(:)
    integer(c_size_t)        :: c_length
    integer                  :: copy_len

    text = ""

    if (.not. c_associated(string_ptr)) return

    c_length = c_strlen(string_ptr)
    if (c_length == 0_c_size_t) return

    copy_len = min(int(c_length), len(text))
    call c_f_pointer(string_ptr, chars, [copy_len])
    text(1:copy_len) = transfer(chars(1:copy_len), text(1:copy_len))
  end subroutine copy_c_string_ptr_to_fortran

  subroutine copy_fortran_string_to_c(text, buffer_ptr, capacity)
    character(len=*), intent(in) :: text
    type(c_ptr), value           :: buffer_ptr
    integer(c_size_t), value     :: capacity
    character(kind=c_char), pointer :: buffer(:)
    integer                        :: copy_len, index_char

    if (.not. c_associated(buffer_ptr)) return
    if (capacity == 0_c_size_t) return

    call c_f_pointer(buffer_ptr, buffer, [int(capacity)])
    buffer = c_null_char

    copy_len = min(len_trim(text), int(capacity) - 1)
    do index_char = 1, copy_len
      buffer(index_char) = text(index_char:index_char)
    end do
    buffer(copy_len + 1) = c_null_char
  end subroutine copy_fortran_string_to_c

  subroutine write_size_t_pointer(size_ptr, value)
    type(c_ptr), value :: size_ptr
    integer(i64), intent(in) :: value
    integer(c_size_t), pointer :: out_size

    if (.not. c_associated(size_ptr)) return
    call c_f_pointer(size_ptr, out_size)
    if (associated(out_size)) then
      out_size = int(max(0_i64, value), kind=c_size_t)
    end if
  end subroutine write_size_t_pointer

  pure function make_stage_report(stage_kind, backend_family, execution_route, fallback_reason, &
                                  selection_mode, cold_state, cache_flags, plan_id, elapsed_us) &
      result(report)
    integer(i32), intent(in) :: stage_kind
    integer(i32), intent(in) :: backend_family
    integer(i32), intent(in) :: execution_route
    integer(i32), intent(in) :: fallback_reason
    integer(i32), intent(in) :: selection_mode
    integer(i32), intent(in) :: cold_state
    integer(i64), intent(in) :: cache_flags
    integer(i64), intent(in) :: plan_id
    integer(i64), intent(in) :: elapsed_us
    type(execution_report)   :: report

    report%stage_kind      = stage_kind
    report%backend_family  = backend_family
    report%execution_route = execution_route
    report%plan_id         = plan_id
    report%selection_mode  = selection_mode
    report%cold_state      = cold_state
    report%fallback_reason = fallback_reason
    report%cache_flags     = cache_flags
    report%elapsed_us      = elapsed_us
  end function make_stage_report

  subroutine copy_internal_report_to_c(report, c_report)
    type(execution_report), intent(in)     :: report
    type(c_execution_report), intent(out)  :: c_report

    c_report%stage_kind      = int(report%stage_kind, kind=c_int32_t)
    c_report%backend_family  = int(report%backend_family, kind=c_int32_t)
    c_report%execution_route = int(report%execution_route, kind=c_int32_t)
    c_report%plan_id         = int(report%plan_id, kind=c_int64_t)
    c_report%selection_mode  = int(report%selection_mode, kind=c_int32_t)
    c_report%cold_state      = int(report%cold_state, kind=c_int32_t)
    c_report%fallback_reason = int(report%fallback_reason, kind=c_int32_t)
    c_report%cache_flags     = int(report%cache_flags, kind=c_int64_t)
    c_report%elapsed_us      = int(report%elapsed_us, kind=c_int64_t)
  end subroutine copy_internal_report_to_c

  integer(i32) function prepare_report_buffer(report_buffer_ptr, required_count) result(status_code)
    type(c_ptr), value :: report_buffer_ptr
    integer(i64), intent(in) :: required_count
    type(c_report_buffer), pointer :: report_buffer

    status_code = MIZU_STATUS_OK
    if (.not. c_associated(report_buffer_ptr)) return

    call c_f_pointer(report_buffer_ptr, report_buffer)
    if (.not. associated(report_buffer)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    report_buffer%report_count = int(required_count, kind=c_size_t)
    if (report_buffer%report_capacity < int(required_count, kind=c_size_t)) then
      status_code = MIZU_STATUS_BUFFER_TOO_SMALL
    else if (required_count > 0_i64 .and. .not. c_associated(report_buffer%reports)) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
    end if
  end function prepare_report_buffer

  subroutine fill_report_buffer(report_buffer_ptr, primary_report, secondary_report)
    type(c_ptr), value           :: report_buffer_ptr
    type(execution_report), intent(in) :: primary_report
    type(execution_report), intent(in) :: secondary_report
    type(c_report_buffer), pointer     :: report_buffer
    type(c_execution_report), pointer  :: reports(:)
    integer                            :: report_count

    if (.not. c_associated(report_buffer_ptr)) return

    call c_f_pointer(report_buffer_ptr, report_buffer)
    if (.not. associated(report_buffer)) return
    if (.not. c_associated(report_buffer%reports)) return

    report_count = int(report_buffer%report_count)
    call c_f_pointer(report_buffer%reports, reports, [report_count])
    if (.not. associated(reports)) return

    if (secondary_report%stage_kind /= MIZU_STAGE_NONE .and. report_count >= 2) then
      call copy_internal_report_to_c(secondary_report, reports(1))
      call copy_internal_report_to_c(primary_report, reports(2))
    else if (report_count >= 1) then
      call copy_internal_report_to_c(primary_report, reports(1))
    end if
  end subroutine fill_report_buffer

end module mod_c_api
