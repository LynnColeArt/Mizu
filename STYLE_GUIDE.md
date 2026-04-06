# Mizu Coding Standards and Fortran Dialect

This document defines the working dialect for Mizu.

It borrows the strongest habits from Sporkle and Glamin, then narrows them to
the actual job in front of us:

- low-ceremony local inference
- Apple ANE as a priority target
- CUDA as a first-class target
- one optional third target when it earns its keep
- multimodal decoder models with projector paths
- a Fortran-first control plane with C ABI edges and Go bindings

This is not a generic HPC style guide. It is a runtime guide.

## Core Principles

1. Strong typing always.
2. Explicit is better than implicit.
3. No hidden work on the hot path.
4. Lifetimes are part of the interface.
5. Backend-neutral API, backend-specific implementation.
6. Cache expensive things once and name them honestly.
7. Fail loudly on unsupported runtime paths. Fallback is a decision, not an accident.
8. Projectors and multimodal state are first-class, not bolt-ons.
9. Performance claims are only real when measured on the target machine.
10. If the code hides where time or memory went, rewrite it.

## Project Posture

Mizu is not trying to be a general local AI kitchen sink.

Mizu is an opinionated inference runtime for a small set of deployment targets
and a narrow family of workloads:

- decoder inference
- long-lived agent sessions
- prefix reuse and KV reuse
- multimodal inputs with projector stages
- fast startup and low orchestration overhead

Initial model targets:

- Qwen-3.5-9B multimodal with projector path
- Gemma4-21B multimodal with projector path

Initial deployment targets:

- Apple Silicon with ANE priority
- NVIDIA CUDA
- one optional third target only when it is operationally justified

## Language Split

Use the right language for the right layer.

- Fortran owns runtime orchestration, memory planning, cache policy, session
  lifecycle, model metadata, request scheduling, and testable numerical glue.
- C owns the stable ABI boundary.
- Objective-C or C bridges are acceptable behind the Apple backend boundary.
- CUDA code is acceptable behind the CUDA backend boundary.
- Go stays thin and boring. It binds to the C ABI and should not invent runtime
  semantics of its own.

If a piece of code is hot, backend-specific, and painful in Fortran, move it
behind a narrow interop surface instead of forcing elegance theater.

## Source Tree Conventions

Use a structured tree. Flat source dumps do not scale.

Recommended layout:

```text
src/
  common/
    mod_kinds.f90
    mod_types.f90
    mod_status.f90
    mod_errors.f90
    mod_memory.f90
    mod_tensor.f90
  runtime/
    mod_runtime.f90
    mod_request.f90
    mod_session.f90
    mod_scheduler.f90
    mod_workspace.f90
  cache/
    mod_cache_keys.f90
    mod_plan_cache.f90
    mod_weight_cache.f90
    mod_session_cache.f90
    mod_mm_cache.f90
  model/
    mod_model_manifest.f90
    mod_model_loader.f90
    mod_weight_pack.f90
    mod_projector.f90
  backends/
    apple/
      mod_apple_backend.f90
      mod_apple_planner.f90
      mod_apple_ane.f90
      mod_apple_metal.f90
      apple_bridge.m
    cuda/
      mod_cuda_backend.f90
      mod_cuda_planner.f90
      cuda_bridge.c
  c_api/
    mod_c_api.f90
include/
  mizu.h
tests/
  ...
```

Rules:

- Shared building blocks go in `common/`.
- Runtime lifecycle code goes in `runtime/`.
- Caching logic goes in `cache/`.
- Model and projector logic go in `model/`.
- Backend code never leaks into generic modules.
- Public C ABI declarations have one obvious home.

## Module Structure

Every module follows the same basic layout unless there is a strong reason not
to.

```fortran
module mod_example
  use iso_fortran_env, only: int32, int64, real32, real64
  use iso_c_binding,   only: c_ptr, c_int, c_size_t
  use mod_kinds
  use mod_types

  implicit none

  private
  public :: example_type, example_init, example_execute

  integer(int32), parameter :: DEFAULT_QUEUE_DEPTH = 8

  type :: example_type
    integer(int64) :: bytes_reserved = 0_int64
    logical        :: is_ready       = .false.
  contains
    procedure :: reset => example_reset
  end type example_type

contains

  ...

end module mod_example
```

Rules:

- `implicit none` always.
- `private` by default.
- constants before types, types before procedures.
- initialize all fields.
- module shape should be legible from the first screenful.

## Naming Conventions

Mizu names should sound like runtime code, not generic app code.

### Modules

- Use `mod_` prefix for Fortran modules: `mod_runtime`, `mod_session_cache`.
- Name by responsibility, not category. Avoid `mod_utils`.

### Types

- Use `snake_case` for derived type names in Fortran source:
  `session_state`, `backend_descriptor`, `projector_manifest`.
- Use nouns, not vague containers.
- Suffix by role when useful:
  - `_state` for mutable runtime state
  - `_descriptor` for static capability metadata
  - `_handle` for ownership-bearing references
  - `_manifest` for loaded model metadata
  - `_workspace` for explicit scratch arenas

### Procedures

- Use `snake_case`.
- Use verb-first names:
  `load_model_manifest`, `plan_prefill_step`, `evict_weight_pack`.
- Questions read like questions:
  `is_backend_ready`, `has_projector`, `should_fallback_to_metal`.

### Variables

- Use `snake_case`.
- Boolean names read like English:
  `is_hot`, `has_kv_cache`, `can_run_on_ane`.
- Include units in the name:
  `bytes_reserved`, `timeout_ms`, `tokens_per_step`.

### Constants

- Use `SCREAMING_SNAKE_CASE`.
- Include units or semantic scope:
  `MAX_BATCH_TOKENS`, `DEFAULT_PREFILL_CHUNK_TOKENS`, `GIGABYTE`.

### C ABI Symbols

- All exported C symbols use `mizu_` prefix.
- Example:
  `mizu_runtime_create`, `mizu_model_load`, `mizu_session_step`.

## Types and Numeric Discipline

- Always use explicit kinds.
- Use `int64` for byte counts, offsets, lengths, cache sizes, and token counts
  that might grow.
- Use `int32` only when the range is operationally bounded.
- Use `real32` or `real64` only for quantities that are truly scalar math in
  the runtime. Do not use floating point for sizes, counters, or identifiers.

Typed constants are mandatory:

```fortran
integer(int64), parameter :: KILOBYTE = 1024_int64
integer(int64), parameter :: MEGABYTE = 1024_int64 * KILOBYTE
integer(int64), parameter :: GIGABYTE = 1024_int64 * MEGABYTE
```

## Arrays, Tensors, and Layout

Fortran is column-major. Hardware backends are not obligated to care.

Rules:

- Never assume a backend-visible tensor layout from Fortran rank alone.
- Represent layout explicitly with shape, stride, dtype, and storage metadata.
- Internal kernels may rely on contiguous Fortran layout only when the contract
  says so.
- Packed backend weights are a separate artifact, not just a reinterpretation
  of the source tensor.
- Projector tensors, vision embeddings, text embeddings, and KV pages should be
  distinct runtime types or descriptors when they have different lifecycle
  rules.

If a layout matters, name it in code. Do not smuggle it through comments.

## Ownership and Lifetime

Ownership is part of the API, not an implementation detail.

Every heap allocation, backend buffer, `c_ptr`, file mapping, cache entry, or
session artifact must have an obvious owner.

Rules:

- Caller-owned buffers are preferred at public boundaries.
- Runtime-owned workspaces must be explicit and reusable.
- No surprise allocation in `step`, `decode`, or projector hot paths.
- Scratch buffers belong to a `workspace` or `plan`, not to random procedures.
- Every `c_ptr` must have a documented owner and destruction path.
- If ownership is shared, ref-count it or redesign it.

## Hot Path Rules

The hot path is:

- model load finalization
- projector execution
- prefill
- decode
- KV/cache movement
- backend plan lookup

Hot path rules:

- no hidden allocation
- no hidden file I/O
- no `print *`
- no shape inference that can be done once
- no backend discovery
- no string parsing
- no silent fallback

Prepare state before the hot path, then execute the hot path with the least
possible ceremony.

## Cache Policy

Caching is a first-class runtime feature.

Every expensive reusable artifact should have a stable cache key and a defined
invalidation story.

Mizu cache layers should be explicit:

- `plan_cache`: compiled execution plans, backend programs, graph lowerings
- `weight_cache`: packed or transformed weights per backend/device
- `session_cache`: KV pages, sampler state, prefix state, parked sessions
- `mm_cache`: multimodal embeddings, projector outputs, image preprocessing
  results when valid

Cache key inputs should include some combination of:

- model hash
- runtime version
- backend kind
- device identifier
- dtype and quantization
- tensor layout or pack format
- op family
- shape signature

Cache rules:

- warm caches explicitly
- double-check on cache miss under synchronization
- evict cold rebuildable artifacts first
- never evict live session state by accident
- log fallback reasons when a cache miss causes recompilation or repacking

Sporkle's layered cache approach is a valid inspiration here. The shape is
good. The keys and lifecycle rules need to be stricter for inference.

## Backend Rules

### General

- Generic modules describe capability and intent.
- Backend modules execute.
- Generic modules do not import Apple bridge code, CUDA bridge code, or vendor
  headers indirectly.

### Apple

ANE is a priority target, not a marketing checkbox.

Rules:

- Keep ANE planning separate from Metal execution.
- Never hide whether an operation ran on ANE, Metal, AMX, or CPU.
- Planner decisions should be explicit and inspectable.
- If ANE cannot support a shape, dtype, or op family, return a clear planner
  decision and route to the next best path intentionally.
- Apple bridge code stays behind a narrow C or Objective-C boundary.

### CUDA

- CUDA is a first-class target, not an afterthought.
- CUDA packing and plan compilation are backend artifacts and belong in the
  CUDA layer.
- Plugin or dynamic loading is acceptable if it simplifies deployment, but the
  runtime contract must stay stable.

## Multimodal and Projector Rules

Multimodal support is not "text inference plus a side quest."

Rules:

- Treat projector execution as its own stage with its own plans, caches, and
  workspace needs.
- Keep modality boundaries explicit:
  - raw image or media input
  - preprocessed vision tensor
  - projector output embeddings
  - text token stream
- Do not bury projector state inside generic session fields with vague names.
- Projector cache entries must include model and projector revision identity.
- If multimodal preprocessing is reused across requests, say so in the API.

## Error Handling and Status

Library code must be calm and explicit.

Rules:

- Return status codes or status-bearing types at public boundaries.
- Use `error stop` only for programmer bugs, broken invariants, or test code.
- Unsupported backend paths are not success.
- Fallback decisions should preserve the reason.
- Assertions are welcome in debug-only internal paths.

Good runtime code says:

- what failed
- where it failed
- whether retry is valid
- whether fallback happened
- which backend actually executed

## Concurrency and Async

Mizu should assume concurrent requests and long-lived sessions from the start.

Rules:

- no hidden mutable global state without synchronization
- read-mostly caches should optimize reads
- write paths should double-check under lock
- request lifecycle should be explicit
- blocking outer APIs are acceptable if inner state machines remain legible
- long-lived sessions must survive ordinary request churn

If a cache or session structure cannot explain how it behaves under concurrent
access, it is not ready.

## C Interoperability

`iso_c_binding` is a core tool, not a side feature.

Rules:

- Keep interop boundaries narrow.
- Use thin C-facing structs and handles.
- Do not expose Fortran-only layout assumptions in the C ABI.
- Strings crossing the ABI must use explicit buffer and length rules.
- Backend-specific pointers stay opaque outside their backend module.
- Ownership transfer across the ABI must be documented in the header and the
  Fortran wrapper.

## Documentation Style

Documentation should separate facts from plans.

Use these labels when helpful:

- implemented
- measured
- experimental
- planned

Do not write runtime mythology into core docs.
If a path is aspirational, say so.

## Testing Expectations

Every meaningful runtime surface needs a test story.

Minimum expectations:

- smoke tests for runtime creation and teardown
- backend capability tests
- cache key and cache invalidation tests
- session reuse tests
- projector path tests
- parity tests where feasible across CPU, Apple, and CUDA reference paths
- failure tests for unsupported backend routing

Model-facing tests should include fixtures that resemble the target families,
especially Qwen and Gemma multimodal projector flows.

## Formatting Rules

- 2 spaces per indentation level
- 100 character preferred line limit
- one blank line between procedures
- two blank lines between major conceptual sections when needed
- align related declarations when it improves readability
- no tabs

## Non-Goals

Mizu is not trying to:

- support every model family immediately
- support every quantization immediately
- mimic `llama.cpp` surface area
- hide backend differences behind false sameness
- trade runtime clarity for decorative abstraction

## The Practical Rule

When in doubt, choose the version of the code that makes these five things
obvious:

1. what owns the memory
2. what the cache key is
3. what backend will run
4. what happens on failure
5. what work still happens on the hot path

If those are clear, the code is probably heading in the right direction.
