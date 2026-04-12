module mod_model_manifest
  use mod_kinds,  only: i32, i64, MAX_TENSOR_RANK, MAX_NAME_LEN, MAX_SLOT_NAME_LEN, &
                        MAX_PATH_LEN
  use mod_status, only: MIZU_STATUS_OK, MIZU_STATUS_INVALID_ARGUMENT, &
                        MIZU_STATUS_UNSUPPORTED_MODEL
  use mod_types,  only: MIZU_ABI_VERSION, MIZU_MODEL_FAMILY_UNKNOWN, &
                        MIZU_MODEL_FEATURE_MULTIMODAL, MIZU_MODEL_FEATURE_PROJECTOR, &
                        MIZU_MODALITY_KIND_UNKNOWN, MIZU_MODALITY_KIND_IMAGE, &
                        MIZU_STORAGE_KIND_UNKNOWN, MIZU_STORAGE_KIND_ENCODED_BYTES, &
                        MIZU_DTYPE_UNKNOWN, SOURCE_FORMAT_UNKNOWN, model_info

  implicit none

  private
  public :: MANIFEST_VERSION_MAJOR, MANIFEST_VERSION_MINOR
  public :: runtime_version_fields, source_provenance, modality_contract
  public :: tensor_manifest, projector_manifest, model_manifest
  public :: initialize_model_manifest, reset_model_manifest
  public :: validate_model_manifest, populate_model_info_from_manifest
  public :: manifest_tensor_count, manifest_modality_count
  public :: hash_text64

  integer(i32), parameter :: MANIFEST_VERSION_MAJOR = 0_i32
  integer(i32), parameter :: MANIFEST_VERSION_MINOR = 1_i32

  type :: runtime_version_fields
    integer(i32) :: abi_version     = MIZU_ABI_VERSION
    integer(i32) :: manifest_major  = MANIFEST_VERSION_MAJOR
    integer(i32) :: manifest_minor  = MANIFEST_VERSION_MINOR
    integer(i32) :: planner_version = 0_i32
    integer(i32) :: pack_version    = 0_i32
  end type runtime_version_fields

  type :: source_provenance
    integer(i32)                :: source_format   = SOURCE_FORMAT_UNKNOWN
    integer(i32)                :: source_family   = MIZU_MODEL_FAMILY_UNKNOWN
    character(len=MAX_NAME_LEN) :: source_model_id = ""
    character(len=MAX_NAME_LEN) :: source_revision = ""
    character(len=MAX_NAME_LEN) :: source_hash_text = ""
    character(len=MAX_PATH_LEN) :: source_root     = ""
  end type source_provenance

  type :: modality_contract
    integer(i32)                     :: placeholder_ordinal = 0_i32
    integer(i32)                     :: modality_kind       = MIZU_MODALITY_KIND_UNKNOWN
    integer(i32)                     :: storage_kind        = MIZU_STORAGE_KIND_UNKNOWN
    integer(i32)                     :: dtype               = MIZU_DTYPE_UNKNOWN
    character(len=MAX_SLOT_NAME_LEN) :: slot_name           = ""
  end type modality_contract

  type :: tensor_manifest
    integer(i32)                :: dtype       = MIZU_DTYPE_UNKNOWN
    integer(i32)                :: rank        = 0_i32
    integer(i64)                :: shape(MAX_TENSOR_RANK) = 0_i64
    integer(i64)                :: source_offset = -1_i64
    character(len=MAX_NAME_LEN) :: tensor_name = ""
    character(len=MAX_NAME_LEN) :: tensor_role = ""
    character(len=MAX_NAME_LEN) :: layout_name = ""
    character(len=MAX_NAME_LEN) :: storage_type = ""
    character(len=MAX_PATH_LEN) :: source_path = ""
  end type tensor_manifest

  type :: projector_manifest
    logical                          :: is_present         = .false.
    integer(i32)                     :: placeholder_count  = 0_i32
    integer(i32)                     :: input_modality_kind = MIZU_MODALITY_KIND_IMAGE
    integer(i32)                     :: input_dtype        = MIZU_DTYPE_UNKNOWN
    integer(i32)                     :: embedding_dtype    = MIZU_DTYPE_UNKNOWN
    integer(i64)                     :: revision_identity  = 0_i64
    character(len=MAX_SLOT_NAME_LEN) :: slot_name          = ""
    character(len=MAX_PATH_LEN)      :: artifact_path      = ""
  end type projector_manifest

  type :: model_manifest
    type(runtime_version_fields)     :: runtime_version
    type(source_provenance)          :: provenance
    integer(i32)                     :: model_family       = MIZU_MODEL_FAMILY_UNKNOWN
    integer(i64)                     :: model_features     = 0_i64
    integer(i64)                     :: logical_model_hash = 0_i64
    character(len=MAX_NAME_LEN)      :: tokenizer_name     = ""
    type(projector_manifest)         :: projector
    type(tensor_manifest), allocatable :: tensors(:)
    type(modality_contract), allocatable :: modalities(:)
  end type model_manifest

contains

  subroutine initialize_model_manifest(manifest)
    type(model_manifest), intent(out) :: manifest

    manifest = model_manifest()
    manifest%runtime_version%abi_version    = MIZU_ABI_VERSION
    manifest%runtime_version%manifest_major = MANIFEST_VERSION_MAJOR
    manifest%runtime_version%manifest_minor = MANIFEST_VERSION_MINOR
  end subroutine initialize_model_manifest

  subroutine reset_model_manifest(manifest)
    type(model_manifest), intent(inout) :: manifest

    manifest = model_manifest()
    manifest%runtime_version%abi_version    = MIZU_ABI_VERSION
    manifest%runtime_version%manifest_major = MANIFEST_VERSION_MAJOR
    manifest%runtime_version%manifest_minor = MANIFEST_VERSION_MINOR
  end subroutine reset_model_manifest

  integer(i32) function validate_model_manifest(manifest) result(status_code)
    type(model_manifest), intent(in) :: manifest
    logical :: requires_modalities
    logical :: requires_projector

    if (manifest%model_family == MIZU_MODEL_FAMILY_UNKNOWN) then
      status_code = MIZU_STATUS_UNSUPPORTED_MODEL
      return
    end if

    if (manifest%logical_model_hash == 0_i64) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    if (manifest_tensor_count(manifest) <= 0_i32) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    requires_modalities = (iand(manifest%model_features, MIZU_MODEL_FEATURE_MULTIMODAL) /= 0_i64)
    if (requires_modalities .and. manifest_modality_count(manifest) <= 0_i32) then
      status_code = MIZU_STATUS_INVALID_ARGUMENT
      return
    end if

    requires_projector = (iand(manifest%model_features, MIZU_MODEL_FEATURE_PROJECTOR) /= 0_i64)
    if (requires_projector) then
      if (.not. manifest%projector%is_present) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
      if (manifest%projector%placeholder_count <= 0_i32) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
      if (len_trim(manifest%projector%slot_name) == 0) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
      if (manifest%projector%revision_identity == 0_i64) then
        status_code = MIZU_STATUS_INVALID_ARGUMENT
        return
      end if
    end if

    status_code = MIZU_STATUS_OK
  end function validate_model_manifest

  subroutine populate_model_info_from_manifest(manifest, info)
    type(model_manifest), intent(in) :: manifest
    type(model_info), intent(out)    :: info

    info = model_info()
    info%model_family    = manifest%model_family
    info%model_features  = manifest%model_features
    info%projector_slot_count = merge(1_i32, 0_i32, manifest%projector%is_present)
  end subroutine populate_model_info_from_manifest

  pure integer(i32) function manifest_tensor_count(manifest) result(count)
    type(model_manifest), intent(in) :: manifest

    if (allocated(manifest%tensors)) then
      count = int(size(manifest%tensors), kind=i32)
    else
      count = 0_i32
    end if
  end function manifest_tensor_count

  pure integer(i32) function manifest_modality_count(manifest) result(count)
    type(model_manifest), intent(in) :: manifest

    if (allocated(manifest%modalities)) then
      count = int(size(manifest%modalities), kind=i32)
    else
      count = 0_i32
    end if
  end function manifest_modality_count

  pure integer(i64) function hash_text64(text) result(hash_value)
    character(len=*), intent(in) :: text
    integer(i32)                 :: index
    integer(i32)                 :: code_point

    hash_value = int(z'6A09E667F3BCC909', kind=i64)
    do index = 1_i32, len_trim(text)
      code_point = iachar(text(index:index), kind=i32)
      hash_value = ieor(hash_value, ishftc(hash_value, 7))
      hash_value = ieor(hash_value, ishft(hash_value, -11))
      hash_value = ieor(hash_value, int(code_point, kind=i64))
      hash_value = ieor(hash_value, int(index, kind=i64))
    end do

    if (hash_value == 0_i64) then
      hash_value = 1_i64
    end if
  end function hash_text64

end module mod_model_manifest
