module mod_runtime
  use mod_kinds,  only: i32, i64
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_BUSY, MIZU_STATUS_INVALID_STATE
  use mod_types,  only: runtime_config, runtime_state, model_open_config, model_info, &
                        model_state, execution_report, SOURCE_FORMAT_UNKNOWN, &
                        backend_descriptor
  use mod_workspace, only: initialize_workspace, reset_workspace

  implicit none

  private
  public :: initialize_runtime_state, reset_runtime_state
  public :: initialize_model_state, reset_model_state
  public :: register_model, unregister_model
  public :: register_session, unregister_session
  public :: runtime_can_destroy, validate_runtime_destroy
  public :: model_can_close, validate_model_close
  public :: set_runtime_error

contains

  subroutine initialize_runtime_state(runtime, config)
    type(runtime_state), intent(out) :: runtime
    type(runtime_config), intent(in) :: config

    runtime%config            = config
    call initialize_workspace(runtime%workspace, 0_i64)
    runtime%last_status_code  = MIZU_STATUS_OK
    runtime%live_model_count  = 0_i32
    runtime%detected_backend_mask = 0_i64
    runtime%detected_backend_count = 0_i32
    runtime%detected_backends = backend_descriptor()
    runtime%is_initialized    = .true.
    runtime%last_error_message = ""
  end subroutine initialize_runtime_state

  subroutine reset_runtime_state(runtime)
    type(runtime_state), intent(inout) :: runtime

    runtime%config             = runtime_config()
    call reset_workspace(runtime%workspace)
    runtime%last_status_code   = MIZU_STATUS_OK
    runtime%live_model_count   = 0_i32
    runtime%detected_backend_mask = 0_i64
    runtime%detected_backend_count = 0_i32
    runtime%detected_backends = backend_descriptor()
    runtime%is_initialized     = .false.
    runtime%last_error_message = ""
  end subroutine reset_runtime_state

  subroutine initialize_model_state(model, open_config, info)
    type(model_state), intent(out)       :: model
    type(model_open_config), intent(in)  :: open_config
    type(model_info), intent(in)         :: info

    model%open_config        = open_config
    model%info               = info
    model%last_report        = execution_report()
    model%source_format      = SOURCE_FORMAT_UNKNOWN
    model%logical_model_hash = 0_i64
    model%projector_revision = 0_i64
    model%tensor_count       = 0_i32
    model%modality_count     = 0_i32
    model%source_model_id    = ""
    model%has_import_bundle  = .false.
    model%import_inventory_hash = 0_i64
    model%import_tensor_bytes = 0_i64
    model%import_weight_pack_bytes = 0_i64
    model%import_projector_bytes = 0_i64
    model%import_preview_count = 0_i32
    model%import_projector_artifact_path = ""
    model%import_tensor_names = ""
    model%import_tensor_roles = ""
    model%import_tensor_paths = ""
    if (allocated(model%import_tensors)) deallocate(model%import_tensors)
    model%live_session_count = 0_i32
    model%is_open            = .true.
  end subroutine initialize_model_state

  subroutine reset_model_state(model)
    type(model_state), intent(inout) :: model

    model%open_config        = model_open_config()
    model%info               = model_info()
    model%last_report        = execution_report()
    model%source_format      = SOURCE_FORMAT_UNKNOWN
    model%logical_model_hash = 0_i64
    model%projector_revision = 0_i64
    model%tensor_count       = 0_i32
    model%modality_count     = 0_i32
    model%source_model_id    = ""
    model%has_import_bundle  = .false.
    model%import_inventory_hash = 0_i64
    model%import_tensor_bytes = 0_i64
    model%import_weight_pack_bytes = 0_i64
    model%import_projector_bytes = 0_i64
    model%import_preview_count = 0_i32
    model%import_projector_artifact_path = ""
    model%import_tensor_names = ""
    model%import_tensor_roles = ""
    model%import_tensor_paths = ""
    if (allocated(model%import_tensors)) deallocate(model%import_tensors)
    model%live_session_count = 0_i32
    model%is_open            = .false.
  end subroutine reset_model_state

  subroutine register_model(runtime)
    type(runtime_state), intent(inout) :: runtime

    runtime%live_model_count = runtime%live_model_count + 1_i32
  end subroutine register_model

  subroutine unregister_model(runtime)
    type(runtime_state), intent(inout) :: runtime

    if (runtime%live_model_count > 0_i32) then
      runtime%live_model_count = runtime%live_model_count - 1_i32
    end if
  end subroutine unregister_model

  subroutine register_session(model)
    type(model_state), intent(inout) :: model

    model%live_session_count = model%live_session_count + 1_i32
  end subroutine register_session

  subroutine unregister_session(model)
    type(model_state), intent(inout) :: model

    if (model%live_session_count > 0_i32) then
      model%live_session_count = model%live_session_count - 1_i32
    end if
  end subroutine unregister_session

  pure logical function runtime_can_destroy(runtime) result(can_destroy)
    type(runtime_state), intent(in) :: runtime

    can_destroy = runtime%is_initialized .and. (runtime%live_model_count == 0_i32)
  end function runtime_can_destroy

  pure integer(i32) function validate_runtime_destroy(runtime) result(status_code)
    type(runtime_state), intent(in) :: runtime

    if (.not. runtime%is_initialized) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (runtime%live_model_count > 0_i32) then
      status_code = MIZU_STATUS_BUSY
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_runtime_destroy

  pure logical function model_can_close(model) result(can_close)
    type(model_state), intent(in) :: model

    can_close = model%is_open .and. (model%live_session_count == 0_i32)
  end function model_can_close

  pure integer(i32) function validate_model_close(model) result(status_code)
    type(model_state), intent(in) :: model

    if (.not. model%is_open) then
      status_code = MIZU_STATUS_INVALID_STATE
    else if (model%live_session_count > 0_i32) then
      status_code = MIZU_STATUS_BUSY
    else
      status_code = MIZU_STATUS_OK
    end if
  end function validate_model_close

  subroutine set_runtime_error(runtime, status_code, message)
    type(runtime_state), intent(inout)      :: runtime
    integer(i32), intent(in)                :: status_code
    character(len=*), intent(in), optional  :: message
    integer                                 :: copy_len

    runtime%last_status_code  = status_code
    runtime%last_error_message = ""

    if (present(message)) then
      copy_len = min(len_trim(message), len(runtime%last_error_message))
      if (copy_len > 0) then
        runtime%last_error_message(1:copy_len) = message(1:copy_len)
      end if
    end if
  end subroutine set_runtime_error

end module mod_runtime
