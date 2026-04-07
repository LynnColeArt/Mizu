module mod_backend_probe_support
  use mod_kinds, only: i32

  implicit none

  private
  public :: read_boolean_env_override, command_succeeds

contains

  subroutine read_boolean_env_override(name, has_override, value)
    character(len=*), intent(in) :: name
    logical, intent(out)         :: has_override
    logical, intent(out)         :: value
    character(len=32)            :: raw_value
    character(len=32)            :: normalized_value
    integer                      :: status_code
    integer                      :: value_length

    has_override = .false.
    value = .false.
    raw_value = ""
    normalized_value = ""

    call get_environment_variable(trim(name), raw_value, length=value_length, status=status_code)
    if (status_code /= 0) return
    if (value_length <= 0) return

    has_override = .true.
    normalized_value = lowercase_ascii(trim(raw_value))

    select case (trim(normalized_value))
    case ("1", "true", "yes", "on")
      value = .true.
    case default
      value = .false.
    end select
  end subroutine read_boolean_env_override

  logical function command_succeeds(command_text) result(is_success)
    character(len=*), intent(in) :: command_text
    integer                      :: exit_status
    integer                      :: command_status

    is_success = .false.
    if (len_trim(command_text) == 0) return

    exit_status = 1
    call execute_command_line(trim(command_text), exitstat=exit_status, cmdstat=command_status)
    if (command_status /= 0) return
    is_success = (exit_status == 0)
  end function command_succeeds

  pure function lowercase_ascii(text) result(lowered_text)
    character(len=*), intent(in) :: text
    character(len=len(text))     :: lowered_text
    integer(i32)                 :: index_char
    integer(i32)                 :: code_point

    lowered_text = text
    do index_char = 1_i32, len(text)
      code_point = iachar(text(index_char:index_char), kind=i32)
      if (code_point >= iachar("A", kind=i32) .and. code_point <= iachar("Z", kind=i32)) then
        lowered_text(index_char:index_char) = achar(code_point + 32_i32)
      end if
    end do
  end function lowercase_ascii

end module mod_backend_probe_support
