module mod_kinds
  use iso_fortran_env, only: int8, int16, int32, int64, real32, real64
  use iso_c_binding,   only: c_bool, c_char, c_int8_t, c_int32_t, c_int64_t, c_size_t

  implicit none

  private
  public :: i8, i16, i32, i64, r32, r64
  public :: c_bool_kind, c_char_kind, c_i8, c_i32, c_i64, c_size_kind
  public :: KILOBYTE, MEGABYTE, GIGABYTE
  public :: MAX_TENSOR_RANK, MAX_NAME_LEN, MAX_SLOT_NAME_LEN
  public :: MAX_PATH_LEN, MAX_ERROR_MESSAGE_LEN

  integer, parameter :: i8  = int8
  integer, parameter :: i16 = int16
  integer, parameter :: i32 = int32
  integer, parameter :: i64 = int64
  integer, parameter :: r32 = real32
  integer, parameter :: r64 = real64

  integer, parameter :: c_bool_kind = c_bool
  integer, parameter :: c_char_kind = c_char
  integer, parameter :: c_i8        = c_int8_t
  integer, parameter :: c_i32       = c_int32_t
  integer, parameter :: c_i64       = c_int64_t
  integer, parameter :: c_size_kind = c_size_t

  integer(i64), parameter :: KILOBYTE = 1024_i64
  integer(i64), parameter :: MEGABYTE = 1024_i64 * KILOBYTE
  integer(i64), parameter :: GIGABYTE = 1024_i64 * MEGABYTE

  integer(i32), parameter :: MAX_TENSOR_RANK      = 8_i32
  integer(i32), parameter :: MAX_NAME_LEN         = 128_i32
  integer(i32), parameter :: MAX_SLOT_NAME_LEN    = 64_i32
  integer(i32), parameter :: MAX_PATH_LEN         = 512_i32
  integer(i32), parameter :: MAX_ERROR_MESSAGE_LEN = 256_i32

end module mod_kinds
