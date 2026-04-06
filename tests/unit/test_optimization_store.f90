program test_optimization_store
  use mod_kinds,              only: i64
  use mod_optimization_store, only: runtime_optimization_store, &
                                    initialize_runtime_optimization_store, &
                                    reset_runtime_optimization_store, &
                                    record_execution_sample, lookup_winner_plan_id, &
                                    load_runtime_optimization_store, &
                                    save_runtime_optimization_store

  implicit none

  type(runtime_optimization_store) :: store
  type(runtime_optimization_store) :: reloaded_store
  integer(i64)                     :: winner_plan_id
  logical                          :: has_winner
  logical                          :: saved_ok
  logical                          :: loaded_ok
  character(len=*), parameter      :: store_path = "/tmp/mizu_test_optimization_store.txt"

  call initialize_runtime_optimization_store(store)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_false("empty store should not have winner", has_winner)

  call record_execution_sample(store, "prefill:key:a", 101_i64, 10_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_true("winner should exist after first sample", has_winner)
  call expect_equal_i64("first winner", winner_plan_id, 101_i64)

  call record_execution_sample(store, "prefill:key:a", 202_i64, 5_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_equal_i64("lower average candidate should take winner", winner_plan_id, 202_i64)

  call record_execution_sample(store, "prefill:key:a", 101_i64, 1_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_equal_i64("winner should remain faster incumbent", winner_plan_id, 202_i64)

  call record_execution_sample(store, "prefill:key:a", 101_i64, 1_i64)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_equal_i64("winner should promote after better measured average", winner_plan_id, 101_i64)

  call execute_command_line("rm -f " // store_path)
  call save_runtime_optimization_store(store, store_path, saved_ok)
  call expect_true("optimization store save should succeed", saved_ok)

  call initialize_runtime_optimization_store(reloaded_store)
  call load_runtime_optimization_store(reloaded_store, store_path, loaded_ok)
  call expect_true("optimization store load should succeed", loaded_ok)
  call lookup_winner_plan_id(reloaded_store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_true("reloaded store should preserve winner", has_winner)
  call expect_equal_i64("reloaded winner should match", winner_plan_id, 101_i64)
  call execute_command_line("rm -f " // store_path)

  call reset_runtime_optimization_store(store)
  call lookup_winner_plan_id(store, "prefill:key:a", winner_plan_id, has_winner)
  call expect_false("reset store should clear winner", has_winner)

  write(*, "(A)") "test_optimization_store: PASS"

contains

  subroutine expect_equal_i64(label, actual, expected)
    character(len=*), intent(in) :: label
    integer(i64), intent(in)     :: actual
    integer(i64), intent(in)     :: expected

    if (actual /= expected) then
      write(*, "(A,1X,I0,1X,A,1X,I0)") trim(label), actual, "/=", expected
      error stop 1
    end if
  end subroutine expect_equal_i64

  subroutine expect_true(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (.not. condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_true

  subroutine expect_false(label, condition)
    character(len=*), intent(in) :: label
    logical, intent(in)          :: condition

    if (condition) then
      write(*, "(A)") trim(label)
      error stop 1
    end if
  end subroutine expect_false

end program test_optimization_store
