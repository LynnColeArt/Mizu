# Mizu Architecture

## Purpose

Mizu is a lightweight local inference runtime for multimodal decoder models.

It is intentionally narrower than `llama.cpp`.
The goal is not to support every model, every quantization, or every user
surface. The goal is to move model state through a small, explicit runtime with
as little ceremony and overhead as possible.

Mizu is optimized around:

- long-lived local agent sessions
- fast startup after first load
- explicit cache reuse
- self-optimization for speed through measured plan selection
- multimodal projector paths
- Apple ANE as a priority execution target
- CUDA as a first-class execution target

## Scope

### Initial Model Targets

- Qwen-3.5-9B multimodal with projector path
- Gemma4-21B multimodal with projector path

### Initial Deployment Targets

- Apple Silicon:
  - ANE-priority planning and execution where supported
  - Metal fallback for unsupported ANE subgraphs or shapes
- NVIDIA CUDA
- Optional third target only when it earns operational priority

### Initial Runtime Goals

- minimal embeddable C API
- thin Go bindings for an agent system
- explicit session lifecycle
- prefix reuse and KV reuse
- projector execution as a first-class stage
- persistent plan and weight caches

## Non-Goals

Mizu v0 is not trying to:

- clone `llama.cpp`
- implement a giant CLI or server surface
- support a broad model zoo
- support every quantization format
- expose every backend feature through one flattened abstraction
- hide backend routing behind silent fallback

## Design Principles

1. Keep the public API small.
2. Keep ownership obvious.
3. Keep backend decisions inspectable.
4. Move expensive work out of the hot path.
5. Cache every rebuildable artifact with a strict key.
6. Treat multimodal stages as native runtime work, not extensions.
7. Separate planning from execution.
8. Measure stage latency explicitly on target hardware.
9. Prefer measured self-tuning over fixed performance folklore.

## Top-Level Structure

Mizu has six major layers:

1. Control plane
2. Model assets
3. Cache hierarchy
4. Adaptive optimization loop
5. Backend planners and executors
6. Embedding surfaces

### 1. Control Plane

The control plane is written primarily in Fortran and owns:

- runtime lifecycle
- model manifest loading
- session creation and teardown
- request scheduling
- memory planning
- cache lookup and eviction policy
- multimodal staging
- error and status reporting

The control plane does not own vendor-specific hot kernels when those are
better expressed in backend-native code.

### 2. Model Assets

Model assets are split into:

- source model manifest
- immutable logical weights
- projector manifest and projector weights
- backend-packed weights

The logical model description should be backend-neutral.
Backend-packed weights are derived artifacts and belong in caches.

### 3. Cache Hierarchy

Mizu treats caching as a first-class runtime concern.

The initial cache hierarchy is:

- `weight_cache`
  - packed or transformed weights per backend and device family
- `plan_cache`
  - compiled execution plans, backend programs, graph lowerings, and scratch
    sizing decisions, candidate variants, and winning plan records
- `session_cache`
  - KV pages, prefix state, parked session state, sampler state where relevant
- `mm_cache`
  - multimodal preprocessing outputs, projector outputs, and reusable modality
    embeddings when valid

### 4. Adaptive Optimization Loop

Mizu must self-optimize for speed.

That does not mean mysterious runtime behavior.
It means the runtime should observe, compare, remember, and reuse faster plans
for the exact deployment context it is actually running in.

The optimization loop owns:

- candidate plan generation
- stage timing capture
- cache-backed winner selection
- warm versus cold path tracking
- limited exploration for new shapes or changed conditions
- exploitation of known fast paths once enough evidence exists

This loop is responsible for turning repeated local work into faster local
work over time.

### 5. Backend Planners and Executors

Each backend is split into:

- capability probe
- planner
- executor
- backend-specific cache hooks

The planner decides whether an operation can run on a given backend and how.
The executor performs the work.

### 6. Embedding Surfaces

Mizu exposes:

- a stable C ABI
- thin Go bindings

The public API should stay small even if the internal runtime grows more
capable.

## Runtime Object Model

The runtime revolves around a few explicit long-lived objects.

### `runtime_context`

Owns:

- backend registry
- process-wide caches
- allocator and workspace pools
- logging and metrics sinks
- optimization history store
- tuning and exploration policy
- configuration and policy flags

### `model_manifest`

Describes:

- architecture family
- tensor inventory
- tokenizer and vocabulary metadata
- projector metadata
- supported runtime features
- source model hash and revision identity

### `model_instance`

Owns:

- immutable logical model references
- resolved backend pack references
- projector references
- plan cache namespace
- optimization namespace

### `session_state`

Owns:

- prompt state
- multimodal attachments or embedding references
- KV residency
- decode cursor
- reuse and parking metadata

### `request_state`

Represents one active unit of work.
Mizu can support blocking public APIs while still using explicit request state
internally.

### `workspace_arena`

Owns scratch memory for:

- projector staging
- prefill
- decode
- backend-visible temporary buffers

Workspace arenas are reusable and should eliminate incidental hot-path
allocation.

### `optimization_store`

Owns:

- measured stage timings by backend, model, and shape
- plan variant histories
- winner records
- cold-versus-warm execution evidence

The optimization store should be persistent for rebuildable artifacts and
lightweight enough to consult during normal execution.

## Execution Model

Mizu v0 should prefer a simple public execution model:

- load model
- open session
- attach tokens and/or modality inputs
- prefill
- decode step
- read outputs
- park or close session

Internally, the runtime can still use:

- bounded worker queues
- explicit planning phases
- request states
- future async execution paths

The public v0 API should stay low ceremony and mostly synchronous.

## Self-Optimization Model

Self-optimization is part of the runtime contract.

Mizu should optimize by measured evidence in four steps:

1. observe
2. compare
3. cache
4. reuse

### Observe

For each stage, Mizu should record:

- backend chosen
- plan identifier
- shape signature
- dtype and pack identity
- cold or warm execution status
- elapsed time

### Compare

When multiple valid plans exist, Mizu should compare them with bounded
exploration rather than compile-time superstition.

Candidate comparison is especially valuable for:

- ANE versus Metal routing on Apple
- alternative packed-weight layouts
- alternative prefill and decode plans
- projector execution variants
- small-shape versus large-shape decode strategies

### Cache

Once a winning plan has enough evidence, Mizu should persist the decision in
the plan cache and optimization store.

### Reuse

Subsequent requests with the same relevant identity should skip exploration and
reuse the winning plan directly unless:

- the runtime version changed
- the device changed
- the pack format changed
- the planner version changed
- the evidence is stale or insufficient

Self-optimization is not an excuse for unbounded experimentation.
Exploration should be:

- bounded
- logged
- cache-aware
- safe to disable
- biased toward warm reusable plans once confidence is high

## Runtime Stages

### Stage 1: Model Load

Model load is responsible for:

- reading the source manifest
- validating architecture support
- locating projector assets
- resolving cache keys
- loading or building backend-packed weights
- loading or building hot execution plans

The expensive part of this stage is expected to be cacheable.
It is also the first opportunity to reuse known-fast plans and weight packs.

### Stage 2: Multimodal Staging

Multimodal staging is responsible for:

- receiving caller-provided modality buffers or references
- normalizing them into projector input tensors
- resolving projector cache entries
- executing projector plans
- returning projector embeddings for prompt splice

This stage must be explicit in the runtime, because Qwen and Gemma multimodal
workloads depend on it.
It should also feed projector timing and planner evidence back into the
optimization store.

### Stage 3: Prefill

Prefill is responsible for:

- prompt token ingestion
- projector embedding splice
- KV population
- plan selection for the current shape

Prefill is one of the main places where plan caching and workspace reuse
matter.
It is also one of the main places where self-optimization should converge on a
known-fast plan per backend and shape band.

### Stage 4: Decode

Decode is responsible for:

- incremental token generation
- session-local KV reuse
- small-shape plan reuse
- sampler and stop-state progression

Decode should not allocate or rediscover backend capabilities.
Decode should reuse winning per-shape plans whenever possible.

### Stage 5: Park / Resume

Sessions should be able to park without destroying all reusable state.

Parking includes:

- KV state retention policy
- prompt or prefix identity
- sampler or decode continuation state where warranted
- cache pressure-aware eviction

This is essential for agent workloads.

## Backend Contract

Every backend should implement the same conceptual contract, even if the
implementation differs.

### Capability Probe

Returns:

- backend availability
- device identity
- supported dtypes
- supported op families
- memory topology
- planner constraints

### Planner

Resolves:

- whether a requested op can run on the backend
- which variant should run
- which candidate variants should be explored
- required workspace sizes
- required packed weight format
- whether fallback is needed

### Executor

Performs:

- projector execution
- prefill execution
- decode execution
- backend-visible memory movement

### Cache Hooks

Allows the backend to:

- produce weight pack keys
- produce plan keys
- produce candidate plan identities for tuning
- warm reusable executables
- release or evict backend artifacts cleanly

## Apple Backend

Apple is split into two major execution paths:

- ANE path
- Metal path

### Apple Planner

The Apple planner decides per stage or subgraph:

- can this run on ANE
- should this run on ANE
- if not, should this run on Metal
- what packed weights and plan artifacts are needed

The planner must be explicit.
Mizu should never pretend ANE ran when Metal actually executed.

### ANE Strategy

ANE is a priority target, but not every operation or shape will fit.
The ANE path should focus first on:

- projector-friendly graph segments
- shape-stable inference paths that match ANE-friendly execution
- cached compiled or transformed subgraphs where possible

The ANE path should be built with strict planner decisions and measurable
runtime reporting.

### Metal Strategy

Metal provides:

- fallback for unsupported ANE work
- Apple GPU execution for larger or unsupported shapes
- a second fast path on Apple systems

Metal is not a hidden fallback.
It is a declared peer inside the Apple backend.

## CUDA Backend

The CUDA backend is responsible for:

- backend-specific weight packing
- execution plan creation
- prefill and decode kernels
- projector execution when useful
- explicit workspace and stream management

CUDA should match the same planner and cache contracts as Apple, even if the
execution surface differs.

## Optional Third Target

The third target is intentionally deferred.
Candidates could include:

- CPU reference path
- Vulkan
- another accelerator-specific path

The third target should only be added after Apple and CUDA contracts are stable.

## Model Asset Strategy

Mizu should converge on one internal logical model format for runtime use.

That format should describe:

- tensors
- logical graph segments
- projector graph segments
- modality contracts
- tokenizer metadata
- source provenance

From there, each backend may derive:

- packed weights
- compiled plans
- backend-native projector artifacts

This prevents the core runtime from being forced to think in multiple external
format dialects at execution time.

## Cache Keys and Invalidation

Every cache entry must be keyed by explicit identity, not vibes.

Minimum key inputs should include some combination of:

- source model hash
- runtime version
- planner version
- backend kind
- device or architecture identifier
- dtype
- quantization or pack scheme
- shape signature
- projector revision identity

Invalidate on:

- runtime version changes
- device family changes
- pack format changes
- planner version changes
- source model changes

## Memory Model

Mizu uses explicit memory ownership and reusable workspaces.

### Immutable Model Memory

- source model tensors
- tokenizer metadata
- logical manifests

### Derived Backend Memory

- packed weights
- compiled executables
- backend-owned persistent buffers

### Session Memory

- KV pages
- prefix state
- multimodal attachment state

### Workspace Memory

- scratch tensors
- staging buffers
- projector temporary buffers
- plan-local temporaries

Workspace arenas are allocated through the common memory layer, which provides
aligned host buffers for backend bridge handoff. The runtime keeps a high-water
arena per workspace, grows it only when a stage asks for more bytes, preserves
existing scratch contents across growth, and tracks allocation count so tests
can guard reuse paths against accidental hot-path allocation.

General rules:

- caller-owned buffers at public boundaries where practical
- no hidden hot-path allocation
- no mixed ownership across ABI boundaries without an explicit contract

## Multimodal Pipeline

Multimodal support is a native part of the runtime.

Mizu should model the pipeline as:

1. caller provides modality input
2. runtime normalizes to projector input tensors
3. projector runs on a planned backend path
4. projector embeddings are cached when valid
5. projector embeddings are spliced into prompt state
6. prefill and decode continue through the text model

Projector execution is not "just preprocessing."
It has backend, workspace, and cache implications and should be represented as
such.

## Public API Shape

The public API should stay small.

Representative C ABI surface:

- `mizu_runtime_create`
- `mizu_runtime_destroy`
- `mizu_model_open`
- `mizu_model_close`
- `mizu_session_open`
- `mizu_session_close`
- `mizu_session_attach_tokens`
- `mizu_session_attach_modal_input`
- `mizu_session_prefill`
- `mizu_session_decode_step`
- `mizu_session_read_output`
- `mizu_session_park`
- `mizu_session_resume`

The Go binding should mirror this surface without inventing alternate lifecycle
rules.

## Observability

Mizu should expose stage-aware metrics and tracepoints.

Minimum observability:

- model load latency
- cache hit or miss by cache layer
- projector latency
- prefill latency
- decode latency
- backend chosen for each stage
- plan identifier chosen for each stage
- exploration versus exploitation decision
- fallback reason when fallback occurs
- KV residency and eviction counters
- warm-versus-cold execution counters

Performance claims are only real when measured on target hardware.

## Testing Strategy

Mizu needs proof-bearing tests in four broad groups:

1. lifecycle tests
2. cache tests
3. backend routing tests
4. multimodal correctness tests

Minimum v0 test expectations:

- runtime create/destroy smoke tests
- model load tests
- session park/resume tests
- cache invalidation tests
- Apple planner tests for ANE versus Metal decisions
- CUDA parity tests against a trusted reference path
- projector-path tests for target models

## Definition of Done for v0

Mizu v0 is successful when it can:

- load one target multimodal model family cleanly
- execute projector plus text inference on Apple with explicit ANE-versus-Metal
  routing
- execute the same runtime contract on CUDA
- reuse session state and caches across local agent turns
- choose and reuse faster plans over repeated local work
- report where time went and which backend actually ran

That is the bar.
Everything else is follow-on expansion.
