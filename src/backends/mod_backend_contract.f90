module mod_backend_contract
  use mod_kinds,  only: i32, i64, MAX_NAME_LEN, MAX_PATH_LEN, MAX_TENSOR_RANK
  use mod_status, only: MIZU_STATUS_OK
  use mod_types,  only: MIZU_STAGE_NONE, MIZU_MODEL_FAMILY_UNKNOWN, &
                        MIZU_BACKEND_MASK_NONE, MIZU_BACKEND_FAMILY_NONE, &
                        MIZU_EXEC_ROUTE_NONE, MIZU_SELECTION_MODE_NONE, &
                        MIZU_COLD_STATE_UNKNOWN, MIZU_FALLBACK_REASON_NONE, &
                        MIZU_CACHE_FLAG_NONE, MIZU_STOP_REASON_NONE, &
                        backend_descriptor, tensor_descriptor, projector_descriptor, &
                        execution_report, session_handle, workspace_handle

  implicit none

  private
  public :: OP_FAMILY_NONE, OP_FAMILY_PROJECTOR, OP_FAMILY_PREFILL
  public :: OP_FAMILY_DECODE
  public :: capability_probe_request, planner_constraints
  public :: capability_probe_result, backend_workspace_contract
  public :: cache_key_bundle, plan_request, plan_candidate
  public :: planner_result, execution_request, execution_result
  public :: backend_telemetry_record, optimization_record
  public :: backend_contract
  public :: initialize_capability_probe_request, initialize_plan_request
  public :: initialize_execution_request
  public :: backend_contract_is_complete, planner_result_is_success

  integer(i32), parameter :: OP_FAMILY_NONE      = 0_i32
  integer(i32), parameter :: OP_FAMILY_PROJECTOR = 1_i32
  integer(i32), parameter :: OP_FAMILY_PREFILL   = 2_i32
  integer(i32), parameter :: OP_FAMILY_DECODE    = 3_i32

  type :: capability_probe_request
    integer(i32)                :: requested_stage_kind = MIZU_STAGE_NONE
    integer(i64)                :: allowed_backend_mask = MIZU_BACKEND_MASK_NONE
    character(len=MAX_NAME_LEN) :: requested_device     = ""
  end type capability_probe_request

  type :: planner_constraints
    integer(i64) :: supported_dtype_mask = 0_i64
    integer(i64) :: supported_op_mask    = 0_i64
    integer(i64) :: max_workspace_bytes  = 0_i64
    integer(i64) :: max_sequence_tokens  = 0_i64
    integer(i64) :: planner_version      = 0_i64
  end type planner_constraints

  type :: capability_probe_result
    type(backend_descriptor)  :: descriptor
    type(planner_constraints) :: constraints
    integer(i32)              :: status_code = MIZU_STATUS_OK
  end type capability_probe_result

  type :: backend_workspace_contract
    integer(i64) :: required_bytes      = 0_i64
    integer(i64) :: preferred_alignment = 0_i64
    logical      :: requires_zero_fill  = .false.
  end type backend_workspace_contract

  type :: cache_key_bundle
    character(len=MAX_PATH_LEN) :: plan_key            = ""
    character(len=MAX_PATH_LEN) :: weight_pack_key     = ""
    character(len=MAX_PATH_LEN) :: candidate_family_key = ""
  end type cache_key_bundle

  type :: plan_request
    integer(i32)                  :: stage_kind            = MIZU_STAGE_NONE
    integer(i32)                  :: op_family             = OP_FAMILY_NONE
    integer(i32)                  :: model_family          = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i64)                  :: allowed_backend_mask  = MIZU_BACKEND_MASK_NONE
    integer(i64)                  :: preferred_backend_mask = MIZU_BACKEND_MASK_NONE
    integer(i64)                  :: shape_signature(MAX_TENSOR_RANK) = 0_i64
    integer(i64)                  :: token_count           = 0_i64
    integer(i64)                  :: planner_version_hint  = 0_i64
    type(tensor_descriptor)       :: primary_tensor
    type(projector_descriptor)    :: projector
    type(backend_workspace_contract) :: workspace
  end type plan_request

  type :: plan_candidate
    integer(i64)               :: plan_id         = 0_i64
    integer(i32)               :: backend_family  = MIZU_BACKEND_FAMILY_NONE
    integer(i32)               :: execution_route = MIZU_EXEC_ROUTE_NONE
    integer(i32)               :: selection_mode  = MIZU_SELECTION_MODE_NONE
    integer(i64)               :: cache_flags     = MIZU_CACHE_FLAG_NONE
    integer(i64)               :: workspace_bytes = 0_i64
    character(len=MAX_NAME_LEN) :: pack_format    = ""
  end type plan_candidate

  type :: planner_result
    integer(i32)               :: status_code      = MIZU_STATUS_OK
    integer(i32)               :: fallback_reason  = MIZU_FALLBACK_REASON_NONE
    integer(i32)               :: candidate_count  = 0_i32
    logical                    :: requires_fallback = .false.
    type(plan_candidate)       :: chosen_plan
  end type planner_result

  type :: execution_request
    integer(i32)                 :: stage_kind          = MIZU_STAGE_NONE
    type(plan_candidate)         :: plan
    type(session_handle)         :: session_owner
    type(workspace_handle)       :: workspace_owner
    integer(i64)                 :: token_budget        = 0_i64
    integer(i64)                 :: input_token_count   = 0_i64
    integer(i64)                 :: projector_input_count = 0_i64
  end type execution_request

  type :: execution_result
    integer(i32)             :: status_code         = MIZU_STATUS_OK
    integer(i64)             :: emitted_token_count = 0_i64
    integer(i32)             :: stop_reason         = MIZU_STOP_REASON_NONE
    type(execution_report)   :: report
  end type execution_result

  type :: backend_telemetry_record
    type(execution_report) :: report
    integer(i32)           :: candidate_count = 0_i32
    integer(i64)           :: workspace_bytes = 0_i64
  end type backend_telemetry_record

  type :: optimization_record
    integer(i64) :: plan_id              = 0_i64
    integer(i64) :: execution_count      = 0_i64
    integer(i64) :: cumulative_elapsed_us = 0_i64
    integer(i32) :: cold_state           = MIZU_COLD_STATE_UNKNOWN
    integer(i32) :: selection_mode       = MIZU_SELECTION_MODE_NONE
  end type optimization_record

  abstract interface
    subroutine backend_probe_i(request, result, status_code)
      import capability_probe_request, capability_probe_result, i32
      type(capability_probe_request), intent(in)  :: request
      type(capability_probe_result), intent(out)  :: result
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_probe_i

    subroutine backend_plan_i(request, result, status_code)
      import plan_request, planner_result, i32
      type(plan_request), intent(in)              :: request
      type(planner_result), intent(out)           :: result
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_plan_i

    subroutine backend_execute_i(request, result, status_code)
      import execution_request, execution_result, i32
      type(execution_request), intent(in)         :: request
      type(execution_result), intent(out)         :: result
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_execute_i

    subroutine backend_cache_i(request, candidate, keys, status_code)
      import plan_request, plan_candidate, cache_key_bundle, i32
      type(plan_request), intent(in)              :: request
      type(plan_candidate), intent(in)            :: candidate
      type(cache_key_bundle), intent(out)         :: keys
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_cache_i

    subroutine backend_warm_i(keys, status_code)
      import cache_key_bundle, i32
      type(cache_key_bundle), intent(in)          :: keys
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_warm_i

    subroutine backend_evict_i(keys, status_code)
      import cache_key_bundle, i32
      type(cache_key_bundle), intent(in)          :: keys
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_evict_i

    subroutine backend_telemetry_i(record, status_code)
      import backend_telemetry_record, i32
      type(backend_telemetry_record), intent(in)  :: record
      integer(i32), intent(out)                   :: status_code
    end subroutine backend_telemetry_i
  end interface

  type :: backend_contract
    type(backend_descriptor)                                :: descriptor
    procedure(backend_probe_i),     pointer, nopass         :: probe_capabilities => null()
    procedure(backend_plan_i),      pointer, nopass         :: plan_stage         => null()
    procedure(backend_execute_i),   pointer, nopass         :: execute_stage      => null()
    procedure(backend_cache_i),     pointer, nopass         :: build_cache_keys   => null()
    procedure(backend_warm_i),      pointer, nopass         :: warm_artifacts     => null()
    procedure(backend_evict_i),     pointer, nopass         :: evict_artifacts    => null()
    procedure(backend_telemetry_i), pointer, nopass         :: record_telemetry   => null()
  end type backend_contract

contains

  subroutine initialize_capability_probe_request(request, stage_kind, allowed_backend_mask)
    type(capability_probe_request), intent(out) :: request
    integer(i32), intent(in)                    :: stage_kind
    integer(i64), intent(in)                    :: allowed_backend_mask

    request%requested_stage_kind = stage_kind
    request%allowed_backend_mask = allowed_backend_mask
    request%requested_device     = ""
  end subroutine initialize_capability_probe_request

  subroutine initialize_plan_request(request, stage_kind, op_family, model_family, &
                                     allowed_backend_mask)
    type(plan_request), intent(out) :: request
    integer(i32), intent(in)        :: stage_kind
    integer(i32), intent(in)        :: op_family
    integer(i32), intent(in)        :: model_family
    integer(i64), intent(in)        :: allowed_backend_mask

    request%stage_kind            = stage_kind
    request%op_family             = op_family
    request%model_family          = model_family
    request%allowed_backend_mask  = allowed_backend_mask
    request%preferred_backend_mask = MIZU_BACKEND_MASK_NONE
    request%shape_signature       = 0_i64
    request%token_count           = 0_i64
    request%planner_version_hint  = 0_i64
    request%primary_tensor        = tensor_descriptor()
    request%projector             = projector_descriptor()
    request%workspace             = backend_workspace_contract()
  end subroutine initialize_plan_request

  subroutine initialize_execution_request(request, stage_kind, plan)
    type(execution_request), intent(out) :: request
    integer(i32), intent(in)             :: stage_kind
    type(plan_candidate), intent(in)     :: plan

    request%stage_kind            = stage_kind
    request%plan                  = plan
    request%session_owner         = session_handle()
    request%workspace_owner       = workspace_handle()
    request%token_budget          = 0_i64
    request%input_token_count     = 0_i64
    request%projector_input_count = 0_i64
  end subroutine initialize_execution_request

  pure logical function backend_contract_is_complete(contract) result(is_complete)
    type(backend_contract), intent(in) :: contract

    is_complete = associated(contract%probe_capabilities) .and. &
      associated(contract%plan_stage) .and. &
      associated(contract%execute_stage) .and. &
      associated(contract%build_cache_keys) .and. &
      associated(contract%warm_artifacts) .and. &
      associated(contract%evict_artifacts) .and. &
      associated(contract%record_telemetry)
  end function backend_contract_is_complete

  pure logical function planner_result_is_success(result) result(is_success)
    type(planner_result), intent(in) :: result

    is_success = (result%status_code == MIZU_STATUS_OK)
  end function planner_result_is_success

end module mod_backend_contract
