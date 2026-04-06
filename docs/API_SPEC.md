# Mizu API Spec

## Purpose

This document defines the public runtime contract for Mizu v0.

It sits between the architecture doc and the eventual source contracts.
The goal is to describe the smallest useful API surface for a lightweight
multimodal inference runtime with:

- explicit lifecycle
- explicit multimodal staging
- explicit backend reporting
- explicit self-optimization visibility

The public surface is intentionally small.
The internal runtime may be more sophisticated than the public API suggests.

## Scope for v0

Mizu v0 exposes:

- a stable embeddable C ABI
- thin Go bindings over the C ABI
- synchronous blocking calls for the happy path
- token-oriented decoder inference
- multimodal attachment for projector-backed models
- session parking and resume
- stage-aware execution reports

Mizu v0 does not expose:

- tokenizer or detokenizer services
- a general tensor execution API
- training or fine-tuning
- distributed execution
- a large async job system

## API Design Rules

1. Keep the public object graph tiny.
2. Keep ownership explicit.
3. Keep backend routing inspectable.
4. Keep input staging explicit.
5. Keep self-optimization measurable rather than magical.
6. Keep the C boundary boring enough for thin Go bindings.

## Public Object Graph

Mizu exposes three long-lived public handles:

- `mizu_runtime_t`
  - owns process-scoped caches, backend registry, diagnostics, and
    optimization policy
- `mizu_model_t`
  - owns one validated model manifest plus backend pack and plan namespaces
- `mizu_session_t`
  - owns one active or parked conversation state, pending staged inputs, KV
    state, and recent execution reports

Ownership is hierarchical:

- runtime owns models
- model owns sessions

The API should not silently cascade destruction.
Closing a parent with live children returns `MIZU_STATUS_BUSY`.

## ABI and Compatibility Rules

The first public header should follow these rules:

- every public struct begins with `struct_size`
- every config struct includes `abi_version`
- the header exposes `mizu_get_abi_version()`
- callers must zero-initialize all public structs before use
- unknown trailing fields in a larger struct must be ignored safely
- output structs must tolerate the caller passing a smaller supported prefix

This keeps the C ABI forward-compatible without making it complicated.

## Status Model

Every public function returns `mizu_status_code_t`.

The status space should be grouped into stable bands:

- success:
  - `MIZU_STATUS_OK`
  - `MIZU_STATUS_END_OF_SEQUENCE`
- caller contract failures:
  - `MIZU_STATUS_INVALID_ARGUMENT`
  - `MIZU_STATUS_INVALID_STATE`
  - `MIZU_STATUS_BUFFER_TOO_SMALL`
  - `MIZU_STATUS_ABI_MISMATCH`
  - `MIZU_STATUS_BUSY`
- support and planning failures:
  - `MIZU_STATUS_UNSUPPORTED_MODEL`
  - `MIZU_STATUS_UNSUPPORTED_MODALITY`
  - `MIZU_STATUS_BACKEND_UNAVAILABLE`
  - `MIZU_STATUS_NO_VALID_PLAN`
  - `MIZU_STATUS_SESSION_EVICTED`
- runtime failures:
  - `MIZU_STATUS_IO_ERROR`
  - `MIZU_STATUS_CACHE_ERROR`
  - `MIZU_STATUS_EXECUTION_ERROR`
  - `MIZU_STATUS_INTERNAL_ERROR`

Human-readable diagnostics should be copied through a caller-owned buffer:

```c
uint32_t mizu_get_abi_version(void);

mizu_status_code_t mizu_runtime_copy_last_error(
    const mizu_runtime_t *runtime,
    char *buffer,
    size_t capacity,
    size_t *out_required);
```

## Ownership and Memory Rules

The public API should use a small set of predictable memory rules:

- runtime, model, and session handles are runtime-owned objects and must be
  closed explicitly
- input token arrays are copied into the session staging area on successful
  `mizu_session_attach_tokens`
- multimodal inputs may be copied or borrowed depending on the requested
  lifetime policy
- borrowed multimodal buffers need only remain valid until `prefill` returns
- output buffers are always caller-owned
- the runtime must never return heap-allocated strings that the caller must
  free

This gives us low ceremony without hidden ownership traps.

## Session State Rules

The session contract is simpler if expressed as capabilities instead of a
large enum.

A session may have these conditions:

- open
- has pending staged inputs
- has live context
- parked

From that, the valid call rules are:

- `attach_tokens` and `attach_modal_input` require an open, non-parked session
- `clear_pending_inputs` requires an open session
- `prefill` requires pending inputs and a non-parked session
- `decode_step` requires live context, no pending inputs, and a non-parked
  session
- `park` requires live context and a non-parked session
- `resume` requires a parked session
- `read_output` requires at least one completed decode step

Pending inputs are explicit.
If the caller appends more tokens or modality inputs after a prefill, the next
decode must not proceed until another prefill consumes those staged inputs.

## Public Data Types

The exact field order will be settled in `include/mizu.h`, but the first API
needs these conceptual shapes.

### Handle Types

```c
typedef struct mizu_runtime mizu_runtime_t;
typedef struct mizu_model mizu_model_t;
typedef struct mizu_session mizu_session_t;
```

### Backend and Route Identifiers

Mizu must distinguish backend family from actual execution route.

- backend family examples:
  - `MIZU_BACKEND_FAMILY_APPLE`
  - `MIZU_BACKEND_FAMILY_CUDA`
- execution route examples:
  - `MIZU_EXEC_ROUTE_ANE`
  - `MIZU_EXEC_ROUTE_METAL`
  - `MIZU_EXEC_ROUTE_CUDA`

This matters because Apple planning may choose ANE for one stage and Metal for
another.

### Runtime Config

`mizu_runtime_config_t` should include:

- `struct_size`
- `abi_version`
- `cache_root_z`
- `optimization_mode`
- `exploration_budget`
- `runtime_flags`

`optimization_mode` should allow at least:

- disabled
- measure-only
- learn-and-reuse

### Model Open Config

`mizu_model_open_config_t` should include:

- `struct_size`
- `model_root_z`
- `allowed_backend_mask`
- `model_flags`

`allowed_backend_mask` must let callers express:

- Apple ANE only
- Apple Metal only
- Apple ANE preferred with Metal allowed
- CUDA only
- multiple allowed routes with planner choice

The API must support strict routing.
If the caller disallows Metal and ANE cannot execute the requested work, the
call must fail instead of silently falling back.

### Session Config

`mizu_session_config_t` should include:

- `struct_size`
- `max_context_tokens`
- `max_decode_tokens`
- `sampler_kind`
- `seed`
- `temperature`
- `top_k`
- `top_p`
- `session_flags`

The API is token-oriented.
Text tokenization stays outside Mizu for v0.

### Modal Input Descriptor

`mizu_modal_input_desc_t` should include:

- `struct_size`
- `slot_name_z`
- `placeholder_ordinal`
- `modality_kind`
- `storage_kind`
- `dtype`
- `rank`
- `shape`
- `data`
- `byte_count`
- `lifetime_policy`
- `input_flags`

The descriptor must support at least:

- encoded image bytes
- raw tensor views
- precomputed projector embeddings

`placeholder_ordinal` is important for multimodal prompt splice.
The caller must be able to attach an input to a specific placeholder position
instead of relying on silent order-only matching.

If the caller supplies precomputed projector embeddings, the runtime should
skip projector execution for that attachment.

### Decode Options

`mizu_decode_options_t` should include:

- `struct_size`
- `token_budget`
- `stop_flags`
- `decode_flags`

The first API only needs blocking decode with a caller-provided token budget.
Advanced async streaming can come later.

### Decode Result

`mizu_decode_result_t` should include:

- `struct_size`
- `token_buffer`
- `token_capacity`
- `token_count`
- `stop_reason`
- `result_flags`

The caller owns the token buffer.
If the buffer is too small, the call should return
`MIZU_STATUS_BUFFER_TOO_SMALL` and report the required count through
`token_count`.

### Output Buffer

`mizu_output_buffer_t` should include:

- `struct_size`
- `output_kind`
- `data`
- `byte_capacity`
- `bytes_written`
- `output_flags`

This keeps `read_output` generic without forcing the public API to expose a
different function for every future output class.

For v0, the only required `output_kind` is generated token IDs.
On `MIZU_STATUS_BUFFER_TOO_SMALL`, `bytes_written` should report the required
byte count.
Other output kinds are deferred for v0 and may return a stable unsupported
status once enabled.

### Execution Report

`mizu_execution_report_t` should include:

- `struct_size`
- `stage_kind`
- `backend_family`
- `execution_route`
- `plan_id`
- `selection_mode`
- `cold_state`
- `fallback_reason`
- `cache_flags`
- `elapsed_us`

The report is the public proof that Mizu is being honest about:

- what stage just ran
- where it ran
- whether it explored or reused a plan
- whether the path was cold or warm

### Report Buffer

One public call may execute more than one stage.
For example, multimodal prefill may first execute projector work and then
execute prefill.

To keep those stages visible, stage-producing calls should accept an optional
report buffer:

```c
typedef struct {
    size_t struct_size;
    mizu_execution_report_t *reports;
    size_t report_capacity;
    size_t report_count;
} mizu_report_buffer_t;
```

If the caller provides a report buffer, Mizu fills it in execution order.
If the buffer is too small, the call returns `MIZU_STATUS_BUFFER_TOO_SMALL`
and reports the required count through `report_count`.

## Required Function Set

The first public header should define a function set in this shape.

### Runtime

```c
mizu_status_code_t mizu_runtime_create(
    const mizu_runtime_config_t *config,
    mizu_runtime_t **out_runtime);

mizu_status_code_t mizu_runtime_destroy(
    mizu_runtime_t *runtime);
```

Runtime creation must:

- validate ABI version
- initialize diagnostics
- initialize backend registry
- initialize process-scoped caches
- initialize optimization policy

Runtime destruction must:

- return `MIZU_STATUS_BUSY` if live models still exist
- tolerate `NULL` as a no-op

### Model

```c
mizu_status_code_t mizu_model_open(
    mizu_runtime_t *runtime,
    const mizu_model_open_config_t *config,
    mizu_model_t **out_model);

mizu_status_code_t mizu_model_close(
    mizu_model_t *model);

mizu_status_code_t mizu_model_get_info(
    const mizu_model_t *model,
    mizu_model_info_t *out_info);

mizu_status_code_t mizu_model_get_last_report(
    const mizu_model_t *model,
    mizu_execution_report_t *out_report);
```

Model open must:

- validate the manifest
- validate that the model family is supported
- validate projector metadata when present
- resolve weight and plan cache namespaces
- build or load packed weights and hot plans as needed

Model close must return `MIZU_STATUS_BUSY` if sessions still exist.

### Session Lifecycle

```c
mizu_status_code_t mizu_session_open(
    mizu_model_t *model,
    const mizu_session_config_t *config,
    mizu_session_t **out_session);

mizu_status_code_t mizu_session_close(
    mizu_session_t *session);

mizu_status_code_t mizu_session_park(
    mizu_session_t *session,
    mizu_report_buffer_t *out_reports);

mizu_status_code_t mizu_session_resume(
    mizu_session_t *session,
    mizu_report_buffer_t *out_reports);

mizu_status_code_t mizu_session_get_info(
    const mizu_session_t *session,
    mizu_session_info_t *out_info);
```

Parking is in-process for v0.
It preserves reusable live state but does not promise cross-process recovery.

If parked state has been evicted under cache pressure, `resume` must return
`MIZU_STATUS_SESSION_EVICTED`.

### Input Staging

```c
mizu_status_code_t mizu_session_attach_tokens(
    mizu_session_t *session,
    const int32_t *tokens,
    size_t token_count,
    uint32_t attach_flags);

mizu_status_code_t mizu_session_attach_modal_input(
    mizu_session_t *session,
    const mizu_modal_input_desc_t *input);

mizu_status_code_t mizu_session_clear_pending_inputs(
    mizu_session_t *session);
```

Input staging must be additive until consumed by `prefill` or cleared by
`clear_pending_inputs`.

### Execution

```c
mizu_status_code_t mizu_session_prefill(
    mizu_session_t *session,
    mizu_report_buffer_t *out_reports);

mizu_status_code_t mizu_session_decode_step(
    mizu_session_t *session,
    const mizu_decode_options_t *options,
    mizu_decode_result_t *out_result,
    mizu_report_buffer_t *out_reports);

mizu_status_code_t mizu_session_read_output(
    mizu_session_t *session,
    mizu_output_buffer_t *out_output);
```

Execution rules:

- `prefill` consumes all currently staged tokens and modality attachments
- `prefill` may emit more than one stage report
- `decode_step` may emit zero or more token IDs up to the requested budget
- `read_output` reads the most recent staged output without advancing decode

### Reporting

```c
mizu_status_code_t mizu_session_get_last_report(
    const mizu_session_t *session,
    mizu_execution_report_t *out_report);
```

The most recent report is a convenience view.
The per-call report buffer is the authoritative source for multi-stage work.

## Stage Semantics

These stage names are part of the public contract:

- `MODEL_LOAD`
- `PROJECTOR`
- `PREFILL`
- `DECODE`
- `PARK`
- `RESUME`

Important rules:

- `mizu_model_get_last_report` returns the most recent `MODEL_LOAD` report
  after `mizu_model_open`
- token-only prefill emits `PREFILL`
- multimodal prefill emits `PROJECTOR` followed by `PREFILL`
- decode emits `DECODE`
- if Apple planning routes projector to ANE and prefill to Metal, the report
  sequence must say exactly that

## Self-Optimization Contract

Self-optimization is part of the public behavior, even if the internal
implementation evolves.

The public contract is:

- Mizu may explore multiple valid plans when policy allows
- Mizu must mark whether a stage was exploratory or reused
- Mizu must mark whether a stage was cold or warm
- Mizu must persist winning plans only behind strict identity keys
- Mizu must never hide route changes behind a generic "accelerated" label

The caller needs to be able to tell the difference between:

- direct execution with no exploration
- bounded exploration
- winner reuse

That distinction belongs in `selection_mode`.

## No Silent Fallback Rule

Mizu must not silently hide planner decisions.

If the caller allows ANE and Metal, and ANE cannot run the requested work:

- the stage may execute on Metal
- the report must say `execution_route = METAL`
- the report must include a fallback reason

If the caller disallows Metal for that operation:

- the call must fail with `MIZU_STATUS_NO_VALID_PLAN`

## Example Blocking Flow

The intended user flow is:

```c
mizu_runtime_create(...);
mizu_model_open(...);
mizu_session_open(...);

mizu_session_attach_modal_input(...);
mizu_session_attach_tokens(...);
mizu_session_prefill(...);

while (...) {
    mizu_session_decode_step(...);
    mizu_session_read_output(...);
}

mizu_session_park(...);
mizu_session_resume(...);

mizu_session_close(...);
mizu_model_close(...);
mizu_runtime_destroy(...);
```

The API is intentionally boring.
The performance work belongs behind it.

## Deferred from v0

These are intentionally out of scope for the first API shape:

- public async request queues
- tokenizer and detokenizer APIs
- generic raw tensor graph execution
- distributed inference
- training and gradient surfaces
- a large "kitchen sink" output API

## Immediate Translation Tasks

This spec should drive the next concrete implementation work:

1. translate the status model into `src/common/mod_status.f90`
2. translate the public structs into `include/mizu.h`
3. translate the session rules into runtime state checks
4. translate the report contract into backend planner and executor interfaces
