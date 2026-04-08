module mod_model_loader
  use mod_kinds,  only: i32, i64, MAX_TENSOR_RANK, MAX_NAME_LEN, MAX_SLOT_NAME_LEN, &
                        MAX_PATH_LEN
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                        MIZU_STATUS_UNSUPPORTED_MODEL, MIZU_STATUS_IO_ERROR
  use mod_types,  only: MIZU_MODEL_FAMILY_UNKNOWN, MIZU_MODEL_FAMILY_QWEN3_5, &
                        MIZU_MODEL_FAMILY_GEMMA4, MIZU_MODEL_FEATURE_NONE, &
                        MIZU_MODEL_FEATURE_MULTIMODAL, MIZU_MODEL_FEATURE_PROJECTOR, &
                        MIZU_MODALITY_KIND_UNKNOWN, MIZU_MODALITY_KIND_IMAGE, &
                        MIZU_MODALITY_KIND_TENSOR, MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS, &
                        MIZU_STORAGE_KIND_UNKNOWN, MIZU_STORAGE_KIND_ENCODED_BYTES, &
                        MIZU_STORAGE_KIND_HOST_TENSOR, &
                        MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS, MIZU_DTYPE_UNKNOWN, &
                        MIZU_DTYPE_U8, MIZU_DTYPE_I32, MIZU_DTYPE_F16, MIZU_DTYPE_BF16, &
                        MIZU_DTYPE_F32, SOURCE_FORMAT_MIZU_MANIFEST, &
                        SOURCE_FORMAT_BUILTIN_TARGET
  use mod_model_manifest, only: model_manifest, tensor_manifest, modality_contract, &
                                initialize_model_manifest, validate_model_manifest, &
                                manifest_tensor_count, manifest_modality_count, &
                                hash_text64
  use mod_model_import_layout, only: load_import_layout_into_manifest

  implicit none

  private
  public :: load_model_manifest_from_root

  integer(i32), parameter :: DEFAULT_TENSOR_COUNT = 5_i32
  integer(i32), parameter :: DEFAULT_MODALITY_COUNT = 1_i32
  integer(i32), parameter :: MAX_MANIFEST_LINE_LEN = 1024_i32

contains

  integer(i32) function load_model_manifest_from_root(model_root, manifest) result(status_code)
    character(len=*), intent(in) :: model_root
    type(model_manifest), intent(out) :: manifest
    character(len=MAX_PATH_LEN) :: manifest_path
    character(len=MAX_PATH_LEN) :: identity_text
    logical :: has_manifest
    logical :: used_import_layout

    call initialize_model_manifest(manifest)

    if (len_trim(model_root) == 0) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    call resolve_manifest_path(model_root, manifest_path)
    inquire(file=trim(manifest_path), exist=has_manifest)

    if (has_manifest) then
      status_code = load_model_manifest_file(trim(manifest_path), manifest)
    else
      status_code = build_builtin_target_manifest(model_root, manifest)
    end if
    if (status_code /= MIZU_STATUS_OK) return

    status_code = load_import_layout_into_manifest(model_root, manifest, used_import_layout)
    if (status_code /= MIZU_STATUS_OK) return

    if (manifest%provenance%source_family == MIZU_MODEL_FAMILY_UNKNOWN) then
      manifest%provenance%source_family = manifest%model_family
    end if

    if (len_trim(manifest%provenance%source_root) == 0) then
      manifest%provenance%source_root = trim(model_root)
    end if

    if (len_trim(manifest%provenance%source_model_id) == 0) then
      manifest%provenance%source_model_id = model_family_name(manifest%model_family)
    end if

    if (manifest%projector%is_present) then
      manifest%model_features = ior(manifest%model_features, MIZU_MODEL_FEATURE_PROJECTOR)
    end if

    if (manifest_modality_count(manifest) > 0_i32) then
      manifest%model_features = ior(manifest%model_features, MIZU_MODEL_FEATURE_MULTIMODAL)
    end if

    if (manifest%provenance%source_format == SOURCE_FORMAT_BUILTIN_TARGET .and. &
        manifest%model_features == MIZU_MODEL_FEATURE_NONE) then
      manifest%model_features = ior(MIZU_MODEL_FEATURE_MULTIMODAL, MIZU_MODEL_FEATURE_PROJECTOR)
    end if

    if (manifest_tensor_count(manifest) <= 0_i32) then
      call seed_tensor_inventory(manifest, DEFAULT_TENSOR_COUNT)
    end if

    if (iand(manifest%model_features, MIZU_MODEL_FEATURE_MULTIMODAL) /= 0_i64 .and. &
        manifest_modality_count(manifest) <= 0_i32) then
      call ensure_modalities(manifest, DEFAULT_MODALITY_COUNT)
    end if

    if (iand(manifest%model_features, MIZU_MODEL_FEATURE_PROJECTOR) /= 0_i64) then
      call ensure_projector_defaults(manifest)
    end if

    if (manifest%logical_model_hash == 0_i64) then
      identity_text = trim(manifest%provenance%source_hash_text)
      if (len_trim(identity_text) == 0) then
        identity_text = trim(manifest%provenance%source_model_id) // ":" // &
                        trim(manifest%provenance%source_root)
      end if
      manifest%logical_model_hash = hash_text64(trim(identity_text))
    end if

    if (manifest%projector%is_present .and. manifest%projector%revision_identity == 0_i64) then
      manifest%projector%revision_identity = hash_text64(trim(manifest%provenance%source_model_id) // &
        ":projector:" // trim(manifest%projector%slot_name) // ":" // &
        trim(manifest%provenance%source_revision))
    end if

    status_code = validate_model_manifest(manifest)
  end function load_model_manifest_from_root

  subroutine resolve_manifest_path(model_root, manifest_path)
    character(len=*), intent(in)  :: model_root
    character(len=*), intent(out) :: manifest_path
    integer(i32)                  :: root_len

    manifest_path = ""
    root_len = int(len_trim(model_root), kind=i32)
    if (root_len >= 5_i32) then
      if (model_root(root_len - 4_i32:root_len) == ".mizu") then
        manifest_path = trim(model_root)
        return
      end if
    end if

    manifest_path = trim(model_root) // "/manifest.mizu"
  end subroutine resolve_manifest_path

  integer(i32) function load_model_manifest_file(manifest_path, manifest) result(status_code)
    character(len=*), intent(in) :: manifest_path
    type(model_manifest), intent(out) :: manifest
    integer :: io_status
    integer :: unit_number
    character(len=MAX_MANIFEST_LINE_LEN) :: line
    character(len=MAX_NAME_LEN) :: key
    character(len=MAX_PATH_LEN) :: value
    integer(i32) :: separator_index

    call initialize_model_manifest(manifest)
    manifest%provenance%source_format = SOURCE_FORMAT_MIZU_MANIFEST
    manifest%provenance%source_root   = trim(manifest_path)

    open(newunit=unit_number, file=trim(manifest_path), status="old", action="read", &
         iostat=io_status)
    if (io_status /= 0) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    do
      read(unit_number, "(A)", iostat=io_status) line
      if (io_status /= 0) exit

      line = adjustl(line)
      if (len_trim(line) == 0) cycle
      if (line(1:1) == "#") cycle

      separator_index = index(line, "=")
      if (separator_index <= 1_i32) cycle

      key = to_lower_ascii(trim(adjustl(line(1:separator_index - 1_i32))))
      value = trim(adjustl(line(separator_index + 1_i32:)))

      status_code = apply_manifest_pair(manifest, trim(key), trim(value))
      if (status_code /= MIZU_STATUS_OK) then
        close(unit_number)
        return
      end if
    end do

    close(unit_number)

    if (io_status > 0) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    if (manifest%model_family == MIZU_MODEL_FAMILY_UNKNOWN) then
      manifest%model_family = infer_model_family(manifest%provenance%source_model_id)
    end if
    if (manifest%model_family == MIZU_MODEL_FAMILY_UNKNOWN) then
      manifest%model_family = infer_model_family(manifest%provenance%source_root)
    end if

    status_code = MIZU_STATUS_OK
  end function load_model_manifest_file

  integer(i32) function apply_manifest_pair(manifest, key, value) result(status_code)
    type(model_manifest), intent(inout) :: manifest
    character(len=*), intent(in)        :: key
    character(len=*), intent(in)        :: value
    integer(i32)                        :: count_value
    logical                             :: logical_value
    logical                             :: is_valid

    status_code = MIZU_STATUS_OK

    select case (trim(key))
    case ("family")
      manifest%model_family = parse_model_family(value)
    case ("source_model_id")
      manifest%provenance%source_model_id = trim(value)
    case ("source_revision")
      manifest%provenance%source_revision = trim(value)
    case ("source_family")
      manifest%provenance%source_family = parse_model_family(value)
    case ("source_hash_text")
      manifest%provenance%source_hash_text = trim(value)
    case ("model_root")
      manifest%provenance%source_root = trim(value)
    case ("tokenizer")
      manifest%tokenizer_name = trim(value)
    case ("logical_model_hash")
      call parse_i64(value, manifest%logical_model_hash, is_valid)
      if (.not. is_valid) status_code = MIZU_STATUS_INVALID_ARGUMENT
    case ("projector_present")
      call parse_logical_value(value, logical_value, is_valid)
      if (.not. is_valid) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
      else
        manifest%projector%is_present = logical_value
      end if
    case ("projector_revision")
      call parse_i64(value, manifest%projector%revision_identity, is_valid)
      if (.not. is_valid) status_code = MIZU_STATUS_INVALID_ARGUMENT
    case ("projector_slot")
      manifest%projector%slot_name = trim(value)
    case ("projector_placeholder_count")
      call parse_i32(value, count_value, is_valid)
      if (.not. is_valid .or. count_value <= 0_i32) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
      else
        manifest%projector%placeholder_count = count_value
      end if
    case ("projector_input_dtype")
      manifest%projector%input_dtype = parse_dtype(value)
    case ("projector_embedding_dtype")
      manifest%projector%embedding_dtype = parse_dtype(value)
    case ("tensor_count")
      call parse_i32(value, count_value, is_valid)
      if (.not. is_valid .or. count_value <= 0_i32) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
      else
        call seed_tensor_inventory(manifest, count_value)
      end if
    case ("modality_count")
      call parse_i32(value, count_value, is_valid)
      if (.not. is_valid .or. count_value <= 0_i32) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
      else
        call ensure_modalities(manifest, count_value)
      end if
    case ("modality_slot", "modal_slot")
      call ensure_modalities(manifest, DEFAULT_MODALITY_COUNT)
      manifest%modalities(1)%slot_name = trim(value)
    case ("modality_kind")
      call ensure_modalities(manifest, DEFAULT_MODALITY_COUNT)
      manifest%modalities(1)%modality_kind = parse_modality_kind(value)
    case ("modality_storage")
      call ensure_modalities(manifest, DEFAULT_MODALITY_COUNT)
      manifest%modalities(1)%storage_kind = parse_storage_kind(value)
    case ("modality_dtype")
      call ensure_modalities(manifest, DEFAULT_MODALITY_COUNT)
      manifest%modalities(1)%dtype = parse_dtype(value)
    case ("model_features")
      manifest%model_features = parse_feature_mask(value)
    case default
      continue
    end select
  end function apply_manifest_pair

  integer(i32) function build_builtin_target_manifest(model_root, manifest) result(status_code)
    character(len=*), intent(in) :: model_root
    type(model_manifest), intent(out) :: manifest

    call initialize_model_manifest(manifest)
    manifest%provenance%source_format = SOURCE_FORMAT_BUILTIN_TARGET
    manifest%provenance%source_root   = trim(model_root)
    manifest%model_family             = infer_model_family(model_root)
    manifest%provenance%source_family = manifest%model_family
    manifest%model_features           = ior(MIZU_MODEL_FEATURE_MULTIMODAL, MIZU_MODEL_FEATURE_PROJECTOR)

    select case (manifest%model_family)
    case (MIZU_MODEL_FAMILY_QWEN3_5)
      manifest%provenance%source_model_id = "qwen-3.5-9b"
      manifest%provenance%source_revision = "builtin-target"
      manifest%provenance%source_hash_text = "qwen-3.5-9b:builtin"
      manifest%tokenizer_name = "qwen3_5"
    case (MIZU_MODEL_FAMILY_GEMMA4)
      manifest%provenance%source_model_id = "gemma4-21b"
      manifest%provenance%source_revision = "builtin-target"
      manifest%provenance%source_hash_text = "gemma4-21b:builtin"
      manifest%tokenizer_name = "gemma4"
    case default
      status_code = MIZU_STATUS_UNSUPPORTED_MODEL
      return
    end select

    call ensure_modalities(manifest, DEFAULT_MODALITY_COUNT)
    call ensure_projector_defaults(manifest)
    call seed_tensor_inventory(manifest, DEFAULT_TENSOR_COUNT)

    manifest%logical_model_hash = hash_text64(trim(manifest%provenance%source_model_id) // ":" // &
      trim(manifest%provenance%source_root))
    manifest%projector%revision_identity = hash_text64(trim(manifest%provenance%source_model_id) // &
      ":projector:" // trim(manifest%provenance%source_revision))

    status_code = MIZU_STATUS_OK
  end function build_builtin_target_manifest

  subroutine seed_tensor_inventory(manifest, tensor_count)
    type(model_manifest), intent(inout) :: manifest
    integer(i32), intent(in)            :: tensor_count
    integer(i32)                        :: actual_count
    integer(i64)                        :: hidden_size
    integer(i64)                        :: vocab_size
    integer(i64)                        :: projector_width
    integer(i32)                        :: index
    character(len=32)                   :: index_text

    actual_count = max(tensor_count, DEFAULT_TENSOR_COUNT)

    select case (manifest%model_family)
    case (MIZU_MODEL_FAMILY_QWEN3_5)
      hidden_size = 3584_i64
      vocab_size = 152064_i64
      projector_width = 1280_i64
    case (MIZU_MODEL_FAMILY_GEMMA4)
      hidden_size = 4608_i64
      vocab_size = 256000_i64
      projector_width = 1152_i64
    case default
      hidden_size = 4096_i64
      vocab_size = 131072_i64
      projector_width = 1024_i64
    end select

    if (allocated(manifest%tensors)) deallocate(manifest%tensors)
    allocate(manifest%tensors(actual_count))

    do index = 1_i32, actual_count
      write(index_text, "(I0)") index
      manifest%tensors(index)%tensor_name = "tensor_" // trim(index_text)
      manifest%tensors(index)%tensor_role = "generic"
      manifest%tensors(index)%dtype = MIZU_DTYPE_BF16
      manifest%tensors(index)%rank = 1_i32
      manifest%tensors(index)%shape = 0_i64
      manifest%tensors(index)%shape(1) = hidden_size
      manifest%tensors(index)%layout_name = "packed"
    end do

    call set_tensor_entry(manifest%tensors(1), "token_embeddings", "embedding_table", &
      MIZU_DTYPE_BF16, [vocab_size, hidden_size], "row_major")
    call set_tensor_entry(manifest%tensors(2), "decoder_blocks", "decoder_stack", &
      MIZU_DTYPE_BF16, [hidden_size, hidden_size], "packed")
    call set_tensor_entry(manifest%tensors(3), "final_norm", "normalization", &
      MIZU_DTYPE_F32, [hidden_size], "vector")
    call set_tensor_entry(manifest%tensors(4), "lm_head", "token_projection", &
      MIZU_DTYPE_BF16, [hidden_size, vocab_size], "row_major")
    call set_tensor_entry(manifest%tensors(5), "vision_projector", "multimodal_projector", &
      MIZU_DTYPE_F16, [projector_width, hidden_size], "packed")
  end subroutine seed_tensor_inventory

  subroutine set_tensor_entry(tensor, tensor_name, tensor_role, dtype, shape_values, layout_name)
    type(tensor_manifest), intent(inout) :: tensor
    character(len=*), intent(in)         :: tensor_name
    character(len=*), intent(in)         :: tensor_role
    integer(i32), intent(in)             :: dtype
    integer(i64), intent(in)             :: shape_values(:)
    character(len=*), intent(in)         :: layout_name
    integer(i32)                         :: shape_rank

    tensor = tensor_manifest()
    tensor%tensor_name = trim(tensor_name)
    tensor%tensor_role = trim(tensor_role)
    tensor%dtype = dtype
    tensor%layout_name = trim(layout_name)
    shape_rank = int(min(size(shape_values), MAX_TENSOR_RANK), kind=i32)
    tensor%rank = shape_rank
    tensor%shape(1:shape_rank) = shape_values(1:shape_rank)
  end subroutine set_tensor_entry

  subroutine ensure_modalities(manifest, count)
    type(model_manifest), intent(inout) :: manifest
    integer(i32), intent(in)            :: count
    type(modality_contract), allocatable :: resized(:)
    integer(i32)                         :: copy_count
    integer(i32)                         :: index

    if (count <= 0_i32) return

    if (.not. allocated(manifest%modalities)) then
      allocate(manifest%modalities(count))
    else if (size(manifest%modalities) < count) then
      allocate(resized(count))
      resized = modality_contract()
      copy_count = int(size(manifest%modalities), kind=i32)
      resized(1:copy_count) = manifest%modalities(1:copy_count)
      call move_alloc(resized, manifest%modalities)
    end if

    do index = 1_i32, int(size(manifest%modalities), kind=i32)
      if (manifest%modalities(index)%placeholder_ordinal <= 0_i32) then
        manifest%modalities(index)%placeholder_ordinal = index
      end if
      if (manifest%modalities(index)%modality_kind == MIZU_MODALITY_KIND_UNKNOWN) then
        manifest%modalities(index)%modality_kind = MIZU_MODALITY_KIND_IMAGE
      end if
      if (manifest%modalities(index)%storage_kind == MIZU_STORAGE_KIND_UNKNOWN) then
        manifest%modalities(index)%storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES
      end if
      if (manifest%modalities(index)%dtype == MIZU_DTYPE_UNKNOWN) then
        manifest%modalities(index)%dtype = MIZU_DTYPE_U8
      end if
      if (len_trim(manifest%modalities(index)%slot_name) == 0) then
        manifest%modalities(index)%slot_name = default_slot_name(index)
      end if
    end do
  end subroutine ensure_modalities

  subroutine ensure_projector_defaults(manifest)
    type(model_manifest), intent(inout) :: manifest

    manifest%projector%is_present = .true.
    if (manifest%projector%placeholder_count <= 0_i32) then
      manifest%projector%placeholder_count = 1_i32
    end if
    manifest%projector%input_modality_kind = MIZU_MODALITY_KIND_IMAGE
    if (manifest%projector%input_dtype == MIZU_DTYPE_UNKNOWN) then
      manifest%projector%input_dtype = MIZU_DTYPE_F16
    end if
    if (manifest%projector%embedding_dtype == MIZU_DTYPE_UNKNOWN) then
      manifest%projector%embedding_dtype = MIZU_DTYPE_F16
    end if
    if (len_trim(manifest%projector%slot_name) == 0) then
      manifest%projector%slot_name = "image"
    end if
  end subroutine ensure_projector_defaults

  pure integer(i32) function infer_model_family(text) result(model_family)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text

    lowered_text = to_lower_ascii(text)
    if (index(lowered_text, "qwen") > 0) then
      model_family = MIZU_MODEL_FAMILY_QWEN3_5
    else if (index(lowered_text, "gemma") > 0) then
      model_family = MIZU_MODEL_FAMILY_GEMMA4
    else
      model_family = MIZU_MODEL_FAMILY_UNKNOWN
    end if
  end function infer_model_family

  pure integer(i32) function parse_model_family(text) result(model_family)
    character(len=*), intent(in) :: text

    model_family = infer_model_family(text)
  end function parse_model_family

  pure integer(i32) function parse_dtype(text) result(dtype)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text

    lowered_text = to_lower_ascii(text)
    select case (trim(lowered_text))
    case ("u8", "uint8")
      dtype = MIZU_DTYPE_U8
    case ("i32", "int32")
      dtype = MIZU_DTYPE_I32
    case ("f16", "float16")
      dtype = MIZU_DTYPE_F16
    case ("bf16", "bfloat16")
      dtype = MIZU_DTYPE_BF16
    case ("f32", "float32")
      dtype = MIZU_DTYPE_F32
    case default
      dtype = MIZU_DTYPE_UNKNOWN
    end select
  end function parse_dtype

  pure integer(i32) function parse_modality_kind(text) result(modality_kind)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text

    lowered_text = to_lower_ascii(text)
    select case (trim(lowered_text))
    case ("image")
      modality_kind = MIZU_MODALITY_KIND_IMAGE
    case ("tensor")
      modality_kind = MIZU_MODALITY_KIND_TENSOR
    case ("projector_embeddings")
      modality_kind = MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS
    case default
      modality_kind = MIZU_MODALITY_KIND_UNKNOWN
    end select
  end function parse_modality_kind

  pure integer(i32) function parse_storage_kind(text) result(storage_kind)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text

    lowered_text = to_lower_ascii(text)
    select case (trim(lowered_text))
    case ("encoded_bytes")
      storage_kind = MIZU_STORAGE_KIND_ENCODED_BYTES
    case ("host_tensor")
      storage_kind = MIZU_STORAGE_KIND_HOST_TENSOR
    case ("projector_embeddings")
      storage_kind = MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS
    case default
      storage_kind = MIZU_STORAGE_KIND_UNKNOWN
    end select
  end function parse_storage_kind

  pure integer(i64) function parse_feature_mask(text) result(feature_mask)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text

    lowered_text = to_lower_ascii(text)
    feature_mask = MIZU_MODEL_FEATURE_NONE

    if (index(lowered_text, "multimodal") > 0) then
      feature_mask = ior(feature_mask, MIZU_MODEL_FEATURE_MULTIMODAL)
    end if
    if (index(lowered_text, "projector") > 0) then
      feature_mask = ior(feature_mask, MIZU_MODEL_FEATURE_PROJECTOR)
    end if
  end function parse_feature_mask

  subroutine parse_i32(text, value, is_valid)
    character(len=*), intent(in) :: text
    integer(i32), intent(out)    :: value
    logical, intent(out)         :: is_valid
    integer                      :: io_status

    read(text, *, iostat=io_status) value
    is_valid = (io_status == 0)
    if (.not. is_valid) value = 0_i32
  end subroutine parse_i32

  subroutine parse_i64(text, value, is_valid)
    character(len=*), intent(in) :: text
    integer(i64), intent(out)    :: value
    logical, intent(out)         :: is_valid
    integer                      :: io_status

    read(text, *, iostat=io_status) value
    is_valid = (io_status == 0)
    if (.not. is_valid) value = 0_i64
  end subroutine parse_i64

  subroutine parse_logical_value(text, value, is_valid)
    character(len=*), intent(in) :: text
    logical, intent(out)         :: value
    logical, intent(out)         :: is_valid
    character(len=len(text))     :: lowered_text

    lowered_text = trim(to_lower_ascii(text))
    select case (trim(lowered_text))
    case ("1", "true", "yes", "on")
      value = .true.
      is_valid = .true.
    case ("0", "false", "no", "off")
      value = .false.
      is_valid = .true.
    case default
      value = .false.
      is_valid = .false.
    end select
  end subroutine parse_logical_value

  pure function to_lower_ascii(text) result(lowered)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered
    integer(i32)                 :: index
    integer(i32)                 :: code_point

    lowered = text
    do index = 1_i32, len(text)
      code_point = iachar(lowered(index:index), kind=i32)
      if (code_point >= iachar("A") .and. code_point <= iachar("Z")) then
        lowered(index:index) = achar(code_point + 32_i32)
      end if
    end do
  end function to_lower_ascii

  pure function default_slot_name(index) result(slot_name)
    integer(i32), intent(in) :: index
    character(len=MAX_SLOT_NAME_LEN) :: slot_name
    character(len=32) :: index_text

    if (index == 1_i32) then
      slot_name = "image"
    else
      write(index_text, "(I0)") index
      slot_name = "modal_" // trim(index_text)
    end if
  end function default_slot_name

  pure function model_family_name(model_family) result(name)
    integer(i32), intent(in) :: model_family
    character(len=MAX_NAME_LEN) :: name

    select case (model_family)
    case (MIZU_MODEL_FAMILY_QWEN3_5)
      name = "qwen-3.5-9b"
    case (MIZU_MODEL_FAMILY_GEMMA4)
      name = "gemma4-21b"
    case default
      name = "unknown"
    end select
  end function model_family_name

end module mod_model_loader
