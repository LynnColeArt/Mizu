# Mizu Project Plan

## Mission

Build a lightweight local inference runtime for multimodal decoder models with:

- Apple ANE as a priority target
- CUDA as a first-class target
- low-ceremony embedding through a small C API
- thin Go bindings for an agent system
- explicit cache reuse across long-lived local sessions
- self-optimization for speed through measured plan reuse

## Planning Posture

This plan is intentionally narrower than a full general-purpose inference
platform.

We are optimizing for:

- real deployment targets
- a small number of model families
- runtime lifecycle quality
- measurable latency improvements from caching, session reuse, and learned plan
  selection

We are not optimizing for:

- maximum breadth
- broad compatibility theater
- replicating existing kitchen-sink runtimes

## Current Status

As of the initial repository setup:

- local Git repo initialized
- Fortran dialect and style guide written
- architecture document written
- project plan written

## Working Assumptions

1. Apple execution matters enough that ANE must be a first-class planning
   target, not an optional experiment.
2. CUDA must support the same runtime contract as Apple, even if internal
   implementation differs.
3. Multimodal projector support is required from the start.
4. Agent workloads make cache policy and session lifecycle just as important as
   raw kernel speed.
5. A small stable runtime surface beats a wide surface area for v0.
6. The runtime should learn faster plans from repeated local work instead of
   relying only on static heuristics.

## Phase 0: Repository Bootstrap

Status: complete

Deliverables:

- local repository initialized
- `docs/` directory created
- [Architecture](./ARCHITECTURE.md)
- [Style Guide](../STYLE_GUIDE.md)
- this project plan

## Phase 1: Core Contracts

Goal:
Define the runtime contract before implementation fans out.

Deliverables:

- core type map for runtime, model, session, request, workspace, and caches
- error and status taxonomy
- backend contract document or source-level interface
- initial C ABI shape in header form
- telemetry schema for stage timings and plan identities
- cache key schema for:
  - plan cache
  - weight cache
  - session cache
  - multimodal cache

Exit criteria:

- all future modules have a stable naming and ownership model
- Apple and CUDA backends can be implemented against one contract

## Phase 2: Source Model and Internal Asset Format

Goal:
Define one internal runtime-facing model asset dialect.

Deliverables:

- model manifest schema
- projector manifest schema
- source model import path for target families
- logical tensor inventory rules
- versioned internal asset identity

Notes:

- External conversion tooling can be ugly at first if the internal runtime
  contract stays clean.
- The internal format should separate logical weights from backend-packed
  artifacts.

Exit criteria:

- Qwen-3.5-9B and Gemma4-21B assets can be described in one runtime-facing
  format
- projector assets are represented explicitly

## Phase 3: Runtime Skeleton

Goal:
Build the Fortran control plane without backend-specific optimization yet.

Deliverables:

- `runtime_context`
- `model_instance`
- `session_state`
- `workspace_arena`
- `optimization_store`
- cache registries
- blocking request flow:
  - load model
  - open session
  - attach tokens
  - attach modality inputs
  - prefill
  - decode step
  - park/resume

Exit criteria:

- runtime lifecycle works end to end with stub or reference backend paths
- ownership and teardown are explicit and testable

## Phase 4: Apple Backend Bring-Up

Goal:
Stand up the Apple path with explicit ANE-first planning.

Deliverables:

- Apple capability probe
- Apple planner that decides:
  - ANE
  - Metal
  - explicit failure
- initial Apple bridge boundary
- projector execution path on Apple
- prefill and decode execution path on Apple
- backend-specific packed weight and plan caches

Important rule:

- no silent fallback from ANE to Metal
- every planner decision must be inspectable

Exit criteria:

- one target model path runs on Apple
- projector plus text path works
- runtime reports what actually executed

## Phase 5: CUDA Backend Bring-Up

Goal:
Stand up CUDA as a peer backend with the same runtime contract.

Deliverables:

- CUDA capability probe
- CUDA planner
- CUDA packed weight format
- CUDA projector execution path where relevant
- CUDA prefill and decode path
- parity hooks against Apple or reference outputs

Exit criteria:

- one target model path runs on CUDA
- runtime contract matches Apple at the API level

## Phase 6: Cache Hierarchy Implementation

Goal:
Turn cache policy and self-optimization into core performance features.

Deliverables:

- in-memory plan cache with strict keys
- in-memory weight cache with strict keys
- session cache for parked state and KV retention
- multimodal cache for projector outputs or reusable modality embeddings
- disk-backed cache for rebuildable artifacts
- cache warming hooks
- cache metrics and eviction reporting
- plan variant registry
- bounded exploration and exploitation policy
- winner persistence for repeated backend and shape combinations

Key behaviors:

- double-check on cache miss under synchronization
- no eviction of live session artifacts by accident
- explicit invalidation on runtime version or asset changes
- record which plan won and why
- reuse known-fast plans on subsequent equivalent work

Exit criteria:

- repeated local turns avoid unnecessary recompilation, repacking, or projector
  rework
- repeated local shapes and request patterns become faster without code changes

## Phase 7: Multimodal Path Hardening

Goal:
Make the projector path operationally real, not just present.

Deliverables:

- modality input contracts
- projector workspace planning
- projector cache key policy
- embedding splice policy into prompt state
- tests for Qwen and Gemma multimodal prompt paths

Exit criteria:

- multimodal inference is a native runtime path on Apple and CUDA

## Phase 8: C ABI and Go Binding

Goal:
Expose the runtime cleanly to the agent layer.

Deliverables:

- `include/mizu.h`
- stable C handles and ownership rules
- thin Go wrapper package
- example program that:
  - loads a model
  - opens a session
  - sends multimodal input
  - performs prefill and decode
  - resumes a parked session

Exit criteria:

- Go layer remains thin
- runtime semantics still live in Mizu, not in binding glue

## Phase 9: Measurement and Hardening

Goal:
Prove the runtime with real measurements on target machines.

Deliverables:

- stage-aware latency benchmarks
- cache hit and miss counters
- session reuse benchmarks
- Apple backend routing traces
- CUDA parity and throughput checks
- optimizer convergence benchmarks
- cold-to-warm execution comparisons
- evidence that repeated work selects faster stable plans
- failure and fallback tests

Success is measured at the stage level:

- model load
- projector
- prefill
- decode
- session resume

Not just tokens per second.

## Definition of Done for v0

Mizu v0 is done when it can:

- load at least one target multimodal model family
- run projector plus decoder inference through the public runtime contract
- execute on Apple with explicit ANE-versus-Metal reporting
- execute on CUDA with the same external lifecycle
- reuse caches and parked session state across agent turns
- measurably improve repeated execution through cached winning plans
- explain where time went

## Immediate Next Moves

The next highest-leverage tasks are:

1. define the core runtime types and status model in source
2. define the backend contract in source
3. sketch the C header
4. define the internal model manifest dialect
5. scaffold the source tree from the architecture document

## Risks

### Risk: ANE support is narrower than hoped

Response:

- keep ANE planning explicit
- preserve Metal as a declared peer inside the Apple backend
- optimize the planner, not the mythology

### Risk: Multimodal projector support becomes the hidden complexity center

Response:

- treat projectors as first-class runtime stages
- give them dedicated caches and workspaces

### Risk: Fortran becomes a bottleneck in backend hot paths

Response:

- keep orchestration in Fortran
- move truly backend-native hot work behind thin interop boundaries

### Risk: Scope balloons toward a general-purpose runtime

Response:

- keep model scope narrow
- keep backend count small
- keep the public API tiny
