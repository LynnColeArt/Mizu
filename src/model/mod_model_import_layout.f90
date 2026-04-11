module mod_model_import_layout
  use mod_kinds,  only: i8, i32, i64, MAX_TENSOR_RANK, MAX_NAME_LEN, MAX_SLOT_NAME_LEN, &
                        MAX_PATH_LEN
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, MIZU_STATUS_IO_ERROR
  use mod_types,  only: MIZU_MODEL_FAMILY_UNKNOWN, MIZU_MODEL_FAMILY_QWEN3_5, &
                        MIZU_MODEL_FAMILY_GEMMA4, MIZU_MODEL_FEATURE_MULTIMODAL, &
                        MIZU_MODEL_FEATURE_PROJECTOR, MIZU_MODALITY_KIND_UNKNOWN, &
                        MIZU_MODALITY_KIND_IMAGE, MIZU_MODALITY_KIND_TENSOR, &
                        MIZU_MODALITY_KIND_PROJECTOR_EMBEDDINGS, MIZU_STORAGE_KIND_UNKNOWN, &
                        MIZU_STORAGE_KIND_ENCODED_BYTES, MIZU_STORAGE_KIND_HOST_TENSOR, &
                        MIZU_STORAGE_KIND_PROJECTOR_EMBEDDINGS, MIZU_DTYPE_UNKNOWN, &
                        MIZU_DTYPE_U8, MIZU_DTYPE_I32, MIZU_DTYPE_F16, MIZU_DTYPE_BF16, &
                        MIZU_DTYPE_F32, SOURCE_FORMAT_MIZU_IMPORT_BUNDLE
  use mod_model_manifest, only: model_manifest, tensor_manifest, modality_contract

  implicit none

  private
  public :: IMPORT_LAYOUT_DIR_NAME, IMPORT_LAYOUT_VERSION
  public :: load_import_layout_into_manifest

  integer(i32), parameter :: IMPORT_LAYOUT_VERSION = 1_i32
  character(len=*), parameter :: IMPORT_LAYOUT_DIR_NAME = "mizu_import"
  integer(i32), parameter :: MAX_IMPORT_LINE_LEN = 2048_i32

contains

  integer(i32) function load_import_layout_into_manifest(model_root, manifest, used_layout) result(status_code)
    character(len=*), intent(in)   :: model_root
    type(model_manifest), intent(inout) :: manifest
    logical, intent(out)           :: used_layout
    character(len=MAX_PATH_LEN)    :: import_root
    character(len=MAX_PATH_LEN)    :: layout_path
    character(len=MAX_PATH_LEN)    :: tensor_inventory_rel
    character(len=MAX_PATH_LEN)    :: modality_inventory_rel
    character(len=MAX_PATH_LEN)    :: projector_inventory_rel
    character(len=MAX_PATH_LEN)    :: tensor_inventory_path
    character(len=MAX_PATH_LEN)    :: modality_inventory_path
    character(len=MAX_PATH_LEN)    :: projector_inventory_path
    integer(i32)                   :: layout_version
    logical                        :: exists

    used_layout = .false.
    status_code = MIZU_STATUS_OK

    call resolve_import_paths(model_root, import_root, layout_path)
    inquire(file=trim(layout_path), exist=exists)
    if (.not. exists) return

    tensor_inventory_rel = "tensors.tsv"
    modality_inventory_rel = "modalities.tsv"
    projector_inventory_rel = "projector.mizu"
    layout_version = 0_i32

    status_code = parse_layout_file(layout_path, manifest, layout_version, tensor_inventory_rel, &
      modality_inventory_rel, projector_inventory_rel)
    if (status_code /= MIZU_STATUS_OK) return

    if (layout_version /= IMPORT_LAYOUT_VERSION) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    tensor_inventory_path = join_import_path(import_root, tensor_inventory_rel)
    if (len_trim(tensor_inventory_path) == 0) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if
    inquire(file=trim(tensor_inventory_path), exist=exists)
    if (.not. exists) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    status_code = load_tensor_inventory(tensor_inventory_path, import_root, manifest)
    if (status_code /= MIZU_STATUS_OK) return

    modality_inventory_path = join_import_path(import_root, modality_inventory_rel)
    if (len_trim(modality_inventory_path) > 0) then
      inquire(file=trim(modality_inventory_path), exist=exists)
      if (.not. exists) then
        status_code = MIZU_STATUS_IO_ERROR
        return
      end if
      status_code = load_modality_inventory(modality_inventory_path, manifest)
      if (status_code /= MIZU_STATUS_OK) return
    end if

    projector_inventory_path = join_import_path(import_root, projector_inventory_rel)
    if (len_trim(projector_inventory_path) > 0) then
      inquire(file=trim(projector_inventory_path), exist=exists)
      if (.not. exists) then
        status_code = MIZU_STATUS_IO_ERROR
        return
      end if
      status_code = load_projector_inventory(projector_inventory_path, import_root, manifest)
      if (status_code /= MIZU_STATUS_OK) return
    end if

    manifest%provenance%source_format = SOURCE_FORMAT_MIZU_IMPORT_BUNDLE
    manifest%provenance%source_root = trim(model_root)
    used_layout = .true.
  end function load_import_layout_into_manifest

  subroutine resolve_import_paths(model_root, import_root, layout_path)
    character(len=*), intent(in)  :: model_root
    character(len=*), intent(out) :: import_root
    character(len=*), intent(out) :: layout_path

    import_root = trim(model_root) // "/" // IMPORT_LAYOUT_DIR_NAME
    layout_path = trim(import_root) // "/layout.mizu"
  end subroutine resolve_import_paths

  integer(i32) function parse_layout_file(layout_path, manifest, layout_version, tensor_inventory_rel, &
                                          modality_inventory_rel, projector_inventory_rel) result(status_code)
    character(len=*), intent(in)   :: layout_path
    type(model_manifest), intent(inout) :: manifest
    integer(i32), intent(out)      :: layout_version
    character(len=*), intent(inout) :: tensor_inventory_rel
    character(len=*), intent(inout) :: modality_inventory_rel
    character(len=*), intent(inout) :: projector_inventory_rel
    integer(i32)                   :: unit_id
    integer(i32)                   :: ios
    integer(i32)                   :: separator_index
    character(len=MAX_IMPORT_LINE_LEN) :: line
    character(len=MAX_NAME_LEN)    :: key
    character(len=MAX_PATH_LEN)    :: value
    logical                        :: logical_value
    logical                        :: is_valid
    integer(i32)                   :: count_value
    integer(i64)                   :: i64_value

    layout_version = 0_i32
    status_code = MIZU_STATUS_OK

    open(newunit=unit_id, file=trim(layout_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit

      line = adjustl(line)
      if (len_trim(line) == 0) cycle
      if (line(1:1) == "#") cycle

      separator_index = index(line, "=")
      if (separator_index <= 1_i32) cycle

      key = to_lower_ascii(trim(adjustl(line(1:separator_index - 1_i32))))
      value = trim(adjustl(line(separator_index + 1_i32:)))

      select case (trim(key))
      case ("layout_version")
        call parse_i32(value, layout_version, is_valid)
        if (.not. is_valid) status_code = MIZU_STATUS_INVALID_ARGUMENT
      case ("tensor_inventory")
        call normalize_inventory_path(value, tensor_inventory_rel)
      case ("modality_inventory")
        call normalize_inventory_path(value, modality_inventory_rel)
      case ("projector_inventory")
        call normalize_inventory_path(value, projector_inventory_rel)
      case ("family")
        manifest%model_family = parse_model_family(value)
      case ("source_model_id")
        manifest%provenance%source_model_id = trim(value)
      case ("source_revision")
        manifest%provenance%source_revision = trim(value)
      case ("source_hash_text")
        manifest%provenance%source_hash_text = trim(value)
      case ("tokenizer")
        manifest%tokenizer_name = trim(value)
      case ("logical_model_hash")
        call parse_i64(value, i64_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%logical_model_hash = i64_value
        end if
      case ("model_features")
        manifest%model_features = parse_feature_mask(value)
      case ("projector_present")
        call parse_logical_value(value, logical_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%projector%is_present = logical_value
        end if
      case ("projector_slot")
        manifest%projector%slot_name = trim(value)
      case ("projector_placeholder_count")
        call parse_i32(value, count_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%projector%placeholder_count = count_value
        end if
      case ("projector_input_dtype")
        manifest%projector%input_dtype = parse_dtype(value)
      case ("projector_embedding_dtype")
        manifest%projector%embedding_dtype = parse_dtype(value)
      case ("projector_revision")
        call parse_i64(value, i64_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%projector%revision_identity = i64_value
        end if
      case default
        continue
      end select

      if (status_code /= MIZU_STATUS_OK) exit
    end do

    close(unit_id)
    if (status_code /= MIZU_STATUS_OK) return
    if (ios > 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if
  end function parse_layout_file

  integer(i32) function load_tensor_inventory(file_path, import_root, manifest) result(status_code)
    character(len=*), intent(in)   :: file_path
    character(len=*), intent(in)   :: import_root
    type(model_manifest), intent(inout) :: manifest
    integer(i32)                   :: line_count
    integer(i32)                   :: unit_id
    integer(i32)                   :: ios
    integer(i32)                   :: tensor_index
    integer(i32)                   :: field_count
    integer(i32)                   :: rank_value
    character(len=MAX_IMPORT_LINE_LEN) :: line
    character(len=MAX_PATH_LEN)    :: fields(7)
    character(len=MAX_PATH_LEN)    :: source_path
    character(len=MAX_NAME_LEN)    :: storage_type
    logical                        :: is_valid

    line_count = count_data_lines(file_path)
    if (line_count <= 0_i32) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (allocated(manifest%tensors)) deallocate(manifest%tensors)
    allocate(manifest%tensors(line_count))
    manifest%tensors = tensor_manifest()

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    tensor_index = 0_i32
    status_code = MIZU_STATUS_OK
    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit
      if (is_ignored_inventory_line(line)) cycle

      call split_pipe_fields(line, fields, field_count)
      if (field_count < 6_i32 .or. field_count > 7_i32) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        exit
      end if

      source_path = trim(fields(5))
      storage_type = trim(fields(3))
      if (field_count >= 7_i32) storage_type = trim(fields(7))
      if (.not. is_safe_relative_path(source_path)) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        exit
      end if
      if (.not. import_artifact_exists(import_root, source_path)) then
        status_code = MIZU_STATUS_IO_ERROR
        exit
      end if

      tensor_index = tensor_index + 1_i32
      manifest%tensors(tensor_index)%tensor_name = trim(fields(1))
      manifest%tensors(tensor_index)%tensor_role = trim(fields(2))
      manifest%tensors(tensor_index)%dtype = parse_dtype(fields(3))
      manifest%tensors(tensor_index)%layout_name = trim(fields(4))
      manifest%tensors(tensor_index)%source_path = trim(source_path)
      manifest%tensors(tensor_index)%storage_type = trim(storage_type)
      call parse_shape_vector(fields(6), manifest%tensors(tensor_index)%shape, rank_value, is_valid)
      manifest%tensors(tensor_index)%rank = rank_value
      if (.not. is_valid .or. len_trim(manifest%tensors(tensor_index)%tensor_name) == 0 .or. &
          len_trim(manifest%tensors(tensor_index)%tensor_role) == 0 .or. &
          manifest%tensors(tensor_index)%dtype == MIZU_DTYPE_UNKNOWN .or. &
          len_trim(manifest%tensors(tensor_index)%layout_name) == 0 .or. &
          len_trim(manifest%tensors(tensor_index)%storage_type) == 0) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        exit
      end if
    end do

    close(unit_id)
    if (status_code /= MIZU_STATUS_OK) return
    if (ios > 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if
  end function load_tensor_inventory

  integer(i32) function load_modality_inventory(file_path, manifest) result(status_code)
    character(len=*), intent(in)   :: file_path
    type(model_manifest), intent(inout) :: manifest
    integer(i32)                   :: line_count
    integer(i32)                   :: unit_id
    integer(i32)                   :: ios
    integer(i32)                   :: modality_index
    integer(i32)                   :: field_count
    character(len=MAX_IMPORT_LINE_LEN) :: line
    character(len=MAX_PATH_LEN)    :: fields(5)
    logical                        :: is_valid

    line_count = count_data_lines(file_path)
    if (line_count <= 0_i32) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (allocated(manifest%modalities)) deallocate(manifest%modalities)
    allocate(manifest%modalities(line_count))
    manifest%modalities = modality_contract()

    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    modality_index = 0_i32
    status_code = MIZU_STATUS_OK
    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit
      if (is_ignored_inventory_line(line)) cycle

      call split_pipe_fields(line, fields, field_count)
      if (field_count /= 5_i32) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        exit
      end if

      modality_index = modality_index + 1_i32
      call parse_i32(fields(1), manifest%modalities(modality_index)%placeholder_ordinal, is_valid)
      if (.not. is_valid) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        exit
      end if
      manifest%modalities(modality_index)%slot_name = trim(fields(2))
      manifest%modalities(modality_index)%modality_kind = parse_modality_kind(fields(3))
      manifest%modalities(modality_index)%storage_kind = parse_storage_kind(fields(4))
      manifest%modalities(modality_index)%dtype = parse_dtype(fields(5))
      if (manifest%modalities(modality_index)%placeholder_ordinal <= 0_i32 .or. &
          len_trim(manifest%modalities(modality_index)%slot_name) == 0 .or. &
          manifest%modalities(modality_index)%modality_kind == MIZU_MODALITY_KIND_UNKNOWN .or. &
          manifest%modalities(modality_index)%storage_kind == MIZU_STORAGE_KIND_UNKNOWN .or. &
          manifest%modalities(modality_index)%dtype == MIZU_DTYPE_UNKNOWN) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        exit
      end if
    end do

    close(unit_id)
    if (status_code /= MIZU_STATUS_OK) return
    if (ios > 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if
  end function load_modality_inventory

  integer(i32) function load_projector_inventory(file_path, import_root, manifest) result(status_code)
    character(len=*), intent(in)   :: file_path
    character(len=*), intent(in)   :: import_root
    type(model_manifest), intent(inout) :: manifest
    integer(i32)                   :: unit_id
    integer(i32)                   :: ios
    integer(i32)                   :: separator_index
    integer(i32)                   :: count_value
    integer(i64)                   :: i64_value
    character(len=MAX_IMPORT_LINE_LEN) :: line
    character(len=MAX_NAME_LEN)    :: key
    character(len=MAX_PATH_LEN)    :: value
    logical                        :: logical_value
    logical                        :: is_valid

    status_code = MIZU_STATUS_OK
    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit

      line = adjustl(line)
      if (len_trim(line) == 0) cycle
      if (line(1:1) == "#") cycle

      separator_index = index(line, "=")
      if (separator_index <= 1_i32) cycle

      key = to_lower_ascii(trim(adjustl(line(1:separator_index - 1_i32))))
      value = trim(adjustl(line(separator_index + 1_i32:)))

      select case (trim(key))
      case ("present")
        call parse_logical_value(value, logical_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%projector%is_present = logical_value
        end if
      case ("slot", "slot_name")
        manifest%projector%slot_name = trim(value)
      case ("placeholder_count")
        call parse_i32(value, count_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%projector%placeholder_count = count_value
        end if
      case ("input_modality_kind")
        manifest%projector%input_modality_kind = parse_modality_kind(value)
      case ("input_dtype")
        manifest%projector%input_dtype = parse_dtype(value)
      case ("embedding_dtype")
        manifest%projector%embedding_dtype = parse_dtype(value)
      case ("revision_identity")
        call parse_i64(value, i64_value, is_valid)
        if (.not. is_valid) then
          status_code = MIZU_STATUS_INVALID_ARGUMENT
        else
          manifest%projector%revision_identity = i64_value
        end if
      case ("artifact_path")
        manifest%projector%artifact_path = trim(value)
      case default
        continue
      end select

      if (status_code /= MIZU_STATUS_OK) exit
    end do

    close(unit_id)
    if (status_code /= MIZU_STATUS_OK) return
    if (ios > 0_i32) then
      status_code = MIZU_STATUS_IO_ERROR
      return
    end if

    if (manifest%projector%is_present) then
      if (.not. is_safe_relative_path(manifest%projector%artifact_path)) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
      if (.not. import_artifact_exists(import_root, manifest%projector%artifact_path)) then
        status_code = MIZU_STATUS_IO_ERROR
        return
      end if
    end if
  end function load_projector_inventory

  integer(i32) function count_data_lines(file_path) result(line_count)
    character(len=*), intent(in) :: file_path
    integer(i32)                 :: unit_id
    integer(i32)                 :: ios
    character(len=MAX_IMPORT_LINE_LEN) :: line

    line_count = 0_i32
    open(newunit=unit_id, file=trim(file_path), status="old", action="read", iostat=ios)
    if (ios /= 0_i32) return

    do
      read(unit_id, "(A)", iostat=ios) line
      if (ios /= 0_i32) exit
      if (is_ignored_inventory_line(line)) cycle
      line_count = line_count + 1_i32
    end do

    close(unit_id)
    if (ios > 0_i32) line_count = 0_i32
  end function count_data_lines

  logical function is_ignored_inventory_line(line) result(is_ignored)
    character(len=*), intent(in) :: line
    character(len=len(line))      :: trimmed_line

    trimmed_line = adjustl(line)
    is_ignored = (len_trim(trimmed_line) == 0)
    if (.not. is_ignored) is_ignored = (trimmed_line(1:1) == "#")
  end function is_ignored_inventory_line

  subroutine split_pipe_fields(line, fields, field_count)
    character(len=*), intent(in)    :: line
    character(len=*), intent(out)   :: fields(:)
    integer(i32), intent(out)       :: field_count
    integer(i32)                    :: start_index
    integer(i32)                    :: separator_index
    integer(i32)                    :: field_index
    character(len=len(line))        :: working_line

    fields = ""
    field_count = 0_i32
    working_line = trim(line)
    start_index = 1_i32

    do field_index = 1_i32, int(size(fields), kind=i32)
      if (start_index > len_trim(working_line)) exit
      separator_index = index(working_line(start_index:), "|")
      field_count = field_index
      if (separator_index == 0_i32) then
        fields(field_index) = trim(adjustl(working_line(start_index:)))
        exit
      end if
      fields(field_index) = trim(adjustl(working_line(start_index:start_index + separator_index - 2_i32)))
      if (field_index == int(size(fields), kind=i32)) then
        field_count = field_count + 1_i32
        exit
      end if
      start_index = start_index + separator_index
    end do
  end subroutine split_pipe_fields

  subroutine parse_shape_vector(text, shape_values, rank_value, is_valid)
    character(len=*), intent(in) :: text
    integer(i64), intent(out)    :: shape_values(MAX_TENSOR_RANK)
    integer(i32), intent(out)    :: rank_value
    logical, intent(out)         :: is_valid
    character(len=len(text))     :: normalized_text
    character(len=MAX_NAME_LEN)  :: field_text
    integer(i32)                 :: separator_index
    integer(i32)                 :: start_index
    integer(i32)                 :: dim_index
    integer(i64)                 :: dim_value
    logical                      :: value_ok

    shape_values = 0_i64
    rank_value = 0_i32
    is_valid = .false.
    normalized_text = trim(text)
    if (len_trim(normalized_text) == 0) return

    do dim_index = 1, len_trim(normalized_text)
      if (normalized_text(dim_index:dim_index) == ",") normalized_text(dim_index:dim_index) = "x"
    end do

    start_index = 1_i32
    dim_index = 0_i32
    do
      if (start_index > len_trim(normalized_text)) exit
      separator_index = index(normalized_text(start_index:), "x")
      dim_index = dim_index + 1_i32
      if (dim_index > MAX_TENSOR_RANK) return
      if (separator_index == 0_i32) then
        field_text = trim(adjustl(normalized_text(start_index:)))
      else
        field_text = trim(adjustl(normalized_text(start_index:start_index + separator_index - 2_i32)))
      end if
      call parse_i64(field_text, dim_value, value_ok)
      if (.not. value_ok .or. dim_value <= 0_i64) return
      shape_values(dim_index) = dim_value
      if (separator_index == 0_i32) exit
      start_index = start_index + separator_index
    end do

    rank_value = dim_index
    is_valid = (rank_value > 0_i32)
  end subroutine parse_shape_vector

  logical function import_artifact_exists(import_root, relative_path) result(exists_ok)
    character(len=*), intent(in) :: import_root
    character(len=*), intent(in) :: relative_path
    character(len=MAX_PATH_LEN)   :: full_path

    full_path = join_import_path(import_root, relative_path)
    if (len_trim(full_path) == 0) then
      exists_ok = .false.
    else
      inquire(file=trim(full_path), exist=exists_ok)
    end if
  end function import_artifact_exists

  character(len=MAX_PATH_LEN) function join_import_path(import_root, relative_path) result(full_path)
    character(len=*), intent(in) :: import_root
    character(len=*), intent(in) :: relative_path

    full_path = ""
    if (len_trim(relative_path) == 0) return
    if (.not. is_safe_relative_path(relative_path)) return
    full_path = trim(import_root) // "/" // trim(relative_path)
  end function join_import_path

  logical function is_safe_relative_path(path_text) result(is_safe)
    character(len=*), intent(in) :: path_text

    is_safe = (len_trim(path_text) > 0)
    if (.not. is_safe) return
    if (path_text(1:1) == "/") then
      is_safe = .false.
      return
    end if
    if (index(path_text, "..") > 0) then
      is_safe = .false.
      return
    end if
  end function is_safe_relative_path

  subroutine normalize_inventory_path(raw_value, normalized_value)
    character(len=*), intent(in)  :: raw_value
    character(len=*), intent(out) :: normalized_value

    normalized_value = ""
    if (trim(raw_value) == "-") return
    normalized_value = trim(raw_value)
  end subroutine normalize_inventory_path

  pure integer(i32) function parse_model_family(text) result(model_family)
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
    feature_mask = 0_i64
    if (index(lowered_text, "multimodal") > 0) then
      feature_mask = ior(feature_mask, MIZU_MODEL_FEATURE_MULTIMODAL)
    end if
    if (index(lowered_text, "projector") > 0) then
      feature_mask = ior(feature_mask, MIZU_MODEL_FEATURE_PROJECTOR)
    end if
  end function parse_feature_mask

  pure subroutine parse_i32(text, value, is_valid)
    character(len=*), intent(in) :: text
    integer(i32), intent(out)    :: value
    logical, intent(out)         :: is_valid
    integer(i32)                 :: ios

    read(text, *, iostat=ios) value
    is_valid = (ios == 0_i32)
    if (.not. is_valid) value = 0_i32
  end subroutine parse_i32

  pure subroutine parse_i64(text, value, is_valid)
    character(len=*), intent(in) :: text
    integer(i64), intent(out)    :: value
    logical, intent(out)         :: is_valid
    integer(i32)                 :: ios

    read(text, *, iostat=ios) value
    is_valid = (ios == 0_i32)
    if (.not. is_valid) value = 0_i64
  end subroutine parse_i64

  pure subroutine parse_logical_value(text, value, is_valid)
    character(len=*), intent(in) :: text
    logical, intent(out)         :: value
    logical, intent(out)         :: is_valid
    character(len=len(text))     :: lowered_text

    lowered_text = trim(to_lower_ascii(text))
    select case (trim(lowered_text))
    case ("true", "1", "yes")
      value = .true.
      is_valid = .true.
    case ("false", "0", "no")
      value = .false.
      is_valid = .true.
    case default
      value = .false.
      is_valid = .false.
    end select
  end subroutine parse_logical_value

  pure function to_lower_ascii(text) result(lowered_text)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text
    integer(i32)                 :: index_value
    integer(i32)                 :: code_point

    lowered_text = text
    do index_value = 1_i32, int(len(text), kind=i32)
      code_point = iachar(lowered_text(index_value:index_value), kind=i32)
      if (code_point >= iachar("A", kind=i32) .and. code_point <= iachar("Z", kind=i32)) then
        lowered_text(index_value:index_value) = achar(code_point + 32_i32)
      end if
    end do
  end function to_lower_ascii

end module mod_model_import_layout
