# Mizu API Test Matrix

## Purpose

This document decomposes the API spec into concrete tests.

Most of the list is unit or component coverage.
Some API promises also require C ABI contract tests, hardware integration
tests, and performance checks.

The rule is simple:

- if we promise it in the API, we should know which test proves it

## Test Levels

- `U`
  - unit or component test with fake clocks, fake caches, and fake backends
- `C`
  - public C ABI contract test
- `I`
  - real backend integration test on target hardware
- `P`
  - performance or optimizer convergence test

## Required Test Harness Pieces

The unit test suite should not depend on ANE, Metal, or CUDA hardware.
It should use deterministic fixtures:

- fake monotonic clock
- temp cache root
- fake model manifest fixtures
- fake projector fixtures
- fake backend capability probes
- fake planners that can produce multiple candidate plans
- fake executors with injectable latency and outcome data
- fake eviction and cache-corruption injectors

The most important fake backends are:

- `fake_apple_ane_only`
- `fake_apple_metal_only`
- `fake_apple_split`
  - projector can run on ANE, decoder can only run on Metal
- `fake_cuda`
- `fake_dual_plan_backend`
  - returns two valid plans with controllable timing so optimizer behavior is
    testable

## Fixture Models

At minimum we need:

- `fixture_decoder_tiny`
  - token-only path for runtime and session tests
- `fixture_mm_tiny`
  - projector plus decoder path with placeholder metadata
- `fixture_mm_preembedded`
  - projector-bypass path using precomputed embeddings

## Coverage Matrix

### 1. ABI and Struct Contract

- `API-U001`
  Matching `abi_version` is accepted and mismatched `abi_version` is rejected
  with `MIZU_STATUS_ABI_MISMATCH`.
  Covers: `mizu_get_abi_version`, `mizu_runtime_create`

- `API-C001`
  Public structs accept zero-initialized exact-size inputs and larger
  forward-compatible inputs with ignored trailing fields.
  Covers: every public config and output struct

- `API-U002`
  Required output pointers are rejected when `NULL`.
  Covers: runtime create, model open, session open, info getters

- `API-U003`
  `destroy(NULL)` and `close(NULL)` are safe no-ops.
  Covers: runtime destroy, model close, session close

- `API-U004`
  Destroying a runtime with live models returns `MIZU_STATUS_BUSY`.
  Covers: runtime lifetime ownership

- `API-U005`
  Closing a model with live sessions returns `MIZU_STATUS_BUSY`.
  Covers: model lifetime ownership

- `API-U006`
  Error text copying reports required buffer size and truncates safely.
  Covers: `mizu_runtime_copy_last_error`

### 2. Model Open and Identity

- `API-U010`
  Opening a valid supported model fixture succeeds and resolves a model handle.
  Covers: `mizu_model_open`

- `API-U011`
  Opening an invalid manifest fails with a stable non-success status and
  leaves no partially opened model handle.
  Covers: manifest validation

- `API-U012`
  Opening a model with projector metadata missing required assets fails.
  Covers: multimodal model validation

- `API-U013`
  Model info reports whether projector support is present and which backend
  routes are allowed for the instance.
  Covers: `mizu_model_get_info`

- `API-U014`
  After `mizu_model_open`, the most recent model report is a `MODEL_LOAD`
  report with route, timing, and cache metadata.
  Covers: `mizu_model_get_last_report`

### 3. Session Lifecycle and State Rules

- `API-U020`
  Opening and closing a session on a valid model succeeds.
  Covers: `mizu_session_open`, `mizu_session_close`

- `API-U021`
  `prefill` with no staged inputs returns `MIZU_STATUS_INVALID_STATE`.
  Covers: session state validation

- `API-U022`
  `decode_step` before any successful prefill returns
  `MIZU_STATUS_INVALID_STATE`.
  Covers: session state validation

- `API-U023`
  `attach_tokens` and `attach_modal_input` are rejected while a session is
  parked.
  Covers: parked-state rules

- `API-U024`
  `resume` on an active non-parked session returns
  `MIZU_STATUS_INVALID_STATE`.
  Covers: parked-state rules

- `API-U025`
  `park` followed by `resume` restores a live session without rebuilding model
  state.
  Covers: session parking contract

- `API-U026`
  `resume` after forced parked-state eviction returns
  `MIZU_STATUS_SESSION_EVICTED`.
  Covers: session cache and eviction contract

- `API-U027`
  Session info reflects pending-input presence, live-context presence, parked
  status, and KV token count.
  Covers: `mizu_session_get_info`

- `API-U028`
  `park` emits a `PARK` report when the caller provides a report buffer.
  Covers: `mizu_session_park`

- `API-U029`
  `resume` emits a `RESUME` report when the caller provides a report buffer.
  Covers: `mizu_session_resume`

### 4. Token and Modal Input Staging

- `API-U030`
  Attached tokens are copied into session staging and remain correct if the
  caller mutates the original token buffer after the call returns.
  Covers: `mizu_session_attach_tokens`

- `API-U031`
  Multiple token attachment calls preserve append order until prefill.
  Covers: staged token ordering

- `API-U032`
  Unknown `slot_name_z` or invalid `placeholder_ordinal` is rejected.
  Covers: `mizu_session_attach_modal_input`

- `API-U033`
  Unsupported modality or storage kind is rejected with a stable unsupported
  status.
  Covers: modality validation

- `API-U034`
  Borrowed modal buffers may be released immediately after `prefill` returns
  without corrupting live session state.
  Covers: borrowed-input lifetime

- `API-U035`
  Precomputed projector embeddings bypass projector execution on the next
  prefill.
  Covers: pre-embedded multimodal path

- `API-U036`
  `clear_pending_inputs` removes staged inputs without touching existing live
  context.
  Covers: `mizu_session_clear_pending_inputs`

### 5. Prefill, Decode, and Output

- `API-U040`
  Token-only prefill consumes all staged inputs, clears pending state, and
  emits exactly one `PREFILL` report.
  Covers: `mizu_session_prefill`

- `API-U041`
  Multimodal prefill emits `PROJECTOR` followed by `PREFILL` in report order.
  Covers: report buffer sequencing

- `API-U042`
  `decode_step` respects `token_budget` and never emits more tokens than the
  caller allowed.
  Covers: `mizu_session_decode_step`

- `API-U043`
  `decode_step` with pending staged inputs is rejected until another prefill
  runs.
  Covers: prefill-before-decode rule

- `API-U044`
  `decode_step` reports `MIZU_STATUS_END_OF_SEQUENCE` and a stable stop reason
  when EOS or another terminal condition is reached.
  Covers: stop-state contract

- `API-U045`
  `read_output` returns the most recent emitted token IDs without advancing
  decode state.
  Covers: `mizu_session_read_output`

- `API-U046`
  `read_output` returns `MIZU_STATUS_BUFFER_TOO_SMALL` and the required byte
  count when the caller buffer is too small.
  Covers: output buffer contract

- `API-U047`
  A second decode step extends live context rather than discarding prior KV
  state.
  Covers: decode continuation contract

- `API-U048`
  A report buffer that is too small returns `MIZU_STATUS_BUFFER_TOO_SMALL` and
  reports the required report count.
  Covers: `mizu_report_buffer_t`

- `API-U049`
  A decode result token buffer that is too small returns
  `MIZU_STATUS_BUFFER_TOO_SMALL` and reports the required token count.
  Covers: `mizu_decode_result_t`

### 6. Backend Routing and No-Silent-Fallback

- `API-U050`
  An ANE-only allowed backend mask fails with `MIZU_STATUS_NO_VALID_PLAN` when
  the fake planner can only produce a Metal route.
  Covers: strict backend routing

- `API-U051`
  When both ANE and Metal are allowed, a Metal fallback is reflected honestly
  in the execution report with a fallback reason.
  Covers: Apple route reporting

- `API-U052`
  Split Apple execution reports stage-specific routes correctly.
  Example: projector on ANE, prefill on Metal.
  Covers: multi-stage route honesty

- `API-U053`
  CUDA execution uses the same report fields and state transitions as Apple.
  Covers: cross-backend contract parity

- `API-U054`
  `MIZU_STATUS_NO_VALID_PLAN` leaves session state unchanged.
  Covers: planner failure safety

- `API-U055`
  `mizu_session_get_last_report` mirrors the most recent stage report emitted
  by prefill, decode, park, or resume.
  Covers: report convenience view

### 7. Optimizer and Plan Reuse

- `API-U060`
  With optimization disabled, execution still reports route and timing but
  never marks selection as exploratory.
  Covers: optimization-mode behavior

- `API-U061`
  With learning enabled, bounded exploration occurs only while evidence is
  insufficient.
  Covers: optimizer exploration policy

- `API-U062`
  Repeated identical work with a dual-plan fake backend converges on the
  faster plan and then reuses it.
  Covers: winner selection and reuse

- `API-U063`
  `selection_mode` distinguishes direct execution, exploratory execution, and
  winner reuse.
  Covers: report semantics

- `API-U064`
  `cold_state` distinguishes first-time cold execution from warm plan reuse.
  Covers: report semantics

- `API-U065`
  Changing runtime version, planner version, or device identity invalidates a
  previously cached winner.
  Covers: optimization invalidation rules

- `API-U066`
  A corrupted persisted optimization record is ignored safely and replaced when
  new valid evidence exists.
  Covers: optimizer persistence safety

### 8. Cache Keys and Persistence

- `API-U070`
  Plan cache keys are deterministic for identical model, device, stage, shape,
  pack, and planner identity.
  Covers: plan cache key generation

- `API-U071`
  Weight cache keys are deterministic for identical logical model, backend,
  pack format, and runtime version.
  Covers: weight cache key generation

- `API-U072`
  Distinct relevant identities produce distinct cache keys.
  Covers: key uniqueness

- `API-U073`
  Disk-backed cache records round-trip cleanly through write and read.
  Covers: persistence format

- `API-U074`
  Cache corruption or partial writes do not crash the runtime and do not return
  false warm hits.
  Covers: cache hardening

### 9. Public C ABI and Go Binding

- `API-C010`
  The first `include/mizu.h` compiles cleanly as C and as C++.
  Covers: header hygiene

- `API-C011`
  Opaque handles remain opaque from the public header and cannot be stack-sized
  by consumers.
  Covers: handle policy

- `API-C012`
  String and buffer ownership conventions are representable cleanly in Go
  without custom allocators.
  Covers: binding friendliness

- `API-C013`
  A thin Go binding can execute runtime create, model open, session open,
  token attach, prefill, decode, and close against fake backends.
  Covers: end-to-end embedding contract

### 10. Real Backend Integration

- `API-I001`
  On Apple hardware, reported ANE versus Metal routes match the actual planner
  decision path for the executed stage.
  Covers: hardware truthfulness

- `API-I002`
  On Apple hardware, a disallowed Metal fallback really fails instead of
  silently executing.
  Covers: no-silent-fallback on target

- `API-I003`
  On CUDA hardware, the public lifecycle matches the Apple lifecycle and emits
  structurally equivalent reports.
  Covers: backend parity

- `API-I004`
  A target multimodal fixture executes through projector plus decoder on each
  supported backend family.
  Covers: multimodal end-to-end integration

### 11. Performance and Convergence

- `API-P001`
  Warm model load is measurably faster than cold model load for the same model
  and device.
  Covers: load-stage caching

- `API-P002`
  Repeated identical multimodal prefill becomes faster or steadier after winner
  selection stabilizes.
  Covers: optimizer payoff

- `API-P003`
  Repeated decode on the same shape band reuses a stable fast path after the
  exploration budget is exhausted.
  Covers: decode-stage convergence

## Suggested Implementation Order

The first implementation wave should aim at these tests in order:

1. `API-U001` through `API-U006`
2. `API-U010` through `API-U029`
3. `API-U030` through `API-U036`
4. `API-U040` through `API-U049`
5. `API-U050` through `API-U055`
6. `API-U060` through `API-U074`
7. `API-C010` through `API-C013`

That ordering keeps the work aligned with the likely source tree:

- status and common types first
- runtime and session state next
- staging and execution after that
- optimizer and cache contracts once the lifecycle is real

## Immediate Translation Tasks

This matrix should drive the next practical work:

1. create a `test/` directory skeleton with suites grouped by the sections
   above
2. build fake backend fixtures before real backend work
3. map each implemented public function to at least one `U` or `C` test before
   wiring real hardware
