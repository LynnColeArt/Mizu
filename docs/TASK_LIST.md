# Mizu Task List

This document deconstructs the project plan into concrete tasks.

It is meant to answer:

- what can be done now
- what depends on what
- what "done" looks like for each slice

## Current State

- [x] Initialize local Git repository
- [x] Create `docs/` directory
- [x] Write [Style Guide](../STYLE_GUIDE.md)
- [x] Write [Architecture](./ARCHITECTURE.md)
- [x] Write [API Spec](./API_SPEC.md)
- [x] Write [API Test Matrix](./API_TEST_MATRIX.md)
- [x] Write [Project Plan](./PROJECT_PLAN.md)
- [x] Turn planning docs into source tree and contracts

## Immediate Next Up

These are the highest-leverage next tasks.

- [x] Create the source directory skeleton from the architecture doc
- [x] Define core module map in `src/common` and `src/runtime`
- [x] Write the status and error taxonomy
- [x] Write the backend contract in source form
- [x] Translate `docs/API_SPEC.md` into the first C header `include/mizu.h`
- [x] Create the test directory skeleton from `docs/API_TEST_MATRIX.md`
- [x] Define the internal model manifest dialect
- [x] Define cache key types and cache key generation rules

## Phase 0: Repository Bootstrap

- [x] Initialize repo on `main`
- [x] Add `.gitignore`
- [x] Create `docs/README.md`
- [x] Create planning docs
- [x] Add top-level `README.md`
- [x] Add initial `Makefile` or build entrypoint
- [x] Add test directory skeleton

## Phase 1: Core Contracts

### 1.1 Common Types

- [x] Create `src/common/mod_kinds.f90`
- [x] Create `src/common/mod_types.f90`
- [x] Define runtime handle types
- [x] Define model handle types
- [x] Define session handle types
- [x] Define workspace handle types
- [x] Define backend descriptor types
- [x] Define tensor descriptor types
- [x] Define projector descriptor types

### 1.2 Status and Errors

- [x] Create `src/common/mod_status.f90`
- [x] Create `src/common/mod_errors.f90`
- [x] Define status enum bands
- [x] Define recoverable versus unrecoverable errors
- [x] Define backend planner failure reasons
- [x] Define cache failure reasons

### 1.3 Backend Contract

- [x] Create backend capability contract
- [x] Create backend planner contract
- [x] Create backend executor contract
- [x] Define backend-visible workspace contract
- [x] Define backend cache hook contract
- [x] Define backend telemetry contract

### 1.4 C ABI Shape

- [x] Create `include/mizu.h`
- [x] Define opaque handle policy
- [x] Define string boundary policy
- [x] Define buffer ownership policy
- [x] Define lifecycle entrypoints
- [x] Define session step entrypoints
- [x] Define multimodal attachment entrypoints

### 1.5 Telemetry and Optimization Contracts

- [x] Define stage timing record shape
- [x] Define plan identifier shape
- [x] Define optimization record shape
- [x] Define exploration versus exploitation metadata
- [x] Define cold versus warm execution markers

### 1.6 Cache Key Contracts

- [x] Define plan cache key schema
- [x] Define weight cache key schema
- [x] Define session cache key schema
- [x] Define multimodal cache key schema
- [x] Define invalidation version fields

## Phase 2: Internal Model Asset Dialect

### 2.1 Manifest Structure

- [x] Define model manifest schema
- [x] Define projector manifest schema
- [x] Define tensor inventory schema
- [x] Define modality contract schema
- [x] Define source provenance fields
- [x] Define runtime version fields

### 2.2 Target Model Mapping

- [ ] Map Qwen-3.5-9B multimodal tensors into the manifest dialect
- [ ] Map Qwen projector assets into the manifest dialect
- [ ] Map Gemma4-21B multimodal tensors into the manifest dialect
- [ ] Map Gemma projector assets into the manifest dialect

### 2.3 Asset Identity

- [x] Define logical model hash strategy
- [x] Define projector revision identity strategy
- [ ] Define backend pack identity strategy
- [ ] Define planner version identity strategy

### 2.4 Import Path

- [x] Decide where conversion/import tooling lives
- [x] Define importer output layout
- [x] Define validation rules for imported assets
- [x] Define failure reporting for bad manifests
- [x] Retain imported inventory lineage on runtime model state
- [x] Emit imported tensor and projector lineage into materialized backend artifacts
- [x] Derive imported byte-budget hints for backend artifact materialization
- [x] Materialize a deterministic CUDA import-driven weight-pack layout
- [x] Thread CUDA weight-pack dependency into projector and plan cache identity
- [x] Emit stage-specific CUDA `pack_use_*` records for prefill and decode
- [x] Emit compact CUDA `pack_dispatch*` records for prefill and decode
- [x] Collapse generated CUDA `pack_dispatch*` records to `pack=<index>` entries and recover dispatch metadata from `pack_use*` plus `.packbuffer`
- [x] Materialize CUDA `.dispatchbuffer` sidecars so warm plans can recover selected packed-entry indices without textual `pack_use*` recovery
- [x] Emit importer-rooted CUDA `pack_span*` records for prefill and decode
- [x] Materialize CUDA `.usagebuffer` sidecars so warm plans can recover `pack_use_*` summary without textual summary fields
- [x] Materialize CUDA `.spanbuffer` sidecars so warm plans can recover sampled span identity without textual `pack_span*` recovery
- [x] Support compact CUDA replay from binary sidecars alone, without per-entry `pack_dispatch*` plan text
- [x] Verify the public CUDA warm path still replays after generated decode plans drop per-entry `pack_use*`, `pack_dispatch*`, and `pack_span*` text
- [x] Persist effective CUDA `pack_use_hash` into public warm sidecars so generated decode plans can drop textual `pack_use_*` summary fields too
- [x] Verify the public CUDA warm path still replays after generated decode plans drop textual `pack_span_root` and `pack_span_cache` hints
- [x] Verify the public CUDA warm path still replays after generated decode plans drop textual `pack_ref_tile_cache` hints
- [x] Verify the public CUDA warm path still replays after generated decode plans drop textual `pack_ref_artifact` hints
- [x] Derive compact CUDA static pack dependency from the typed `.packbuffer` so public warm replay can also drop textual `pack_ref_hash`, `pack_ref_bytes`, `pack_ref_count`, `weight_pack_hash`, `weight_pack_bytes`, and `weight_pack_count` hints
- [x] Persist the resolved CUDA weight-pack `.packbuffer` path into binary `.usagebuffer` sidecars
- [x] Verify binary-first CUDA warm replay still works after direct pack-buffer hints and intermediate `.packtiles` files are removed
- [x] Compact generated CUDA prefill/decode plans down to stable stage metadata after sidecar materialization
- [x] Verify generated CUDA warm replay still works when compact plans rely on artifact-derived binary sidecars instead of textual compact markers
- [x] Materialize CUDA prefill/decode sidecars directly from imported tensor inventory instead of first expanding transient textual `pack_use*`, `pack_dispatch*`, and `pack_span*` payload records
- [x] Persist staged CUDA pack-page records beside cached tensor spans
- [x] Persist staged CUDA tensor-tile records beside cached tensor spans
- [x] Materialize dedicated CUDA `.tilecache` payloads under `cache_root`
- [x] Materialize pack-owned CUDA `.packtiles` payloads under `cache_root`
- [x] Materialize dedicated CUDA `.packpayload` siblings for pack-owned page/tile bytes
- [x] Materialize dedicated binary CUDA `.packbuffer` siblings for pack-owned page/tile bytes
- [x] Give CUDA `.packbuffer` a typed header and per-pack directory for warm replay
- [x] Restore CUDA dispatch offset/bytes/role/layout from typed pack-buffer records when `pack=` indices are present
- [x] Normalize compact CUDA warm artifact lineage across offset/byte and `pack=` dispatch forms
- [x] Normalize compact CUDA warm artifact lineage across equivalent binary sidecar transport paths
- [x] Emit direct CUDA `pack_ref_tile_buffer=` references for stage artifacts
- [x] Address CUDA warm pack tiles by explicit packed entry index
- [x] Derive CUDA `.packtiles` page/tile records from weight-pack materialization instead of sampled span previews
- [x] Prefer CUDA `.packtiles` materialized hash identity during warm execution when pack-owned cache is available
- [x] Persist CUDA pack-usage snapshots in live context state
- [x] Persist CUDA pack-dispatch snapshots in live context state

## Phase 3: Runtime Skeleton

### 3.1 Runtime Core

- [x] Create `src/runtime/mod_runtime.f90`
- [x] Create `src/runtime/mod_request.f90`
- [x] Create `src/runtime/mod_session.f90`
- [x] Create `src/runtime/mod_workspace.f90`
- [x] Create `src/runtime/mod_scheduler.f90`

### 3.2 Lifecycle

- [x] Implement runtime create and destroy
- [x] Implement model open and close
- [x] Implement session open and close
- [x] Implement session park and resume
- [ ] Implement request-local cleanup rules

### 3.3 Workspace and Memory

- [ ] Create `src/common/mod_memory.f90`
- [ ] Implement aligned host allocation
- [x] Implement reusable workspace arenas
- [x] Implement scratch reservation and release
- [ ] Add hot-path no-allocation assertions where practical

### 3.4 Blocking Request Flow

- [x] Implement attach tokens
- [x] Implement attach multimodal input
- [x] Implement prefill entrypoint
- [x] Implement decode step entrypoint
- [x] Implement output read entrypoint
- [x] Preserve live session context identity across prefill and decode
- [x] Replace the small CUDA live-context record with a backend-owned byte buffer

### 3.5 Optimization Store

- [x] Create `optimization_store` type
- [x] Implement in-memory timing history
- [x] Implement winner record updates
- [x] Implement persistent record format for rebuildable optimization artifacts

## Phase 4: Apple Backend Bring-Up

### 4.1 Apple Capability Layer

- [x] Create Apple backend directory structure
- [x] Define Apple device descriptor
- [x] Implement Apple capability probe
- [x] Detect ANE availability
- [x] Detect Metal availability
- [x] Report planner-visible Apple constraints

### 4.2 Apple Planner

- [x] Implement ANE-versus-Metal planning interface
- [ ] Define unsupported-shape reporting
- [ ] Define unsupported-op reporting
- [ ] Define fallback reason reporting
- [ ] Record planner decisions in telemetry

### 4.3 Apple Bridge Boundary

- [x] Create Apple bridge header surface
- [x] Create Objective-C bridge source
- [ ] Define bridge ownership rules
- [ ] Define bridge error translation rules

### 4.4 Apple Execution

- [x] Implement projector execution path on Apple
- [x] Implement prefill execution path on Apple
- [x] Implement decode execution path on Apple
- [x] Implement Apple workspace handoff
- [x] Implement Apple plan cache integration

### 4.5 Apple Validation

- [x] Write Apple hardware validation checklist for Sam
- [ ] Validate one target multimodal flow on Apple
- [ ] Verify planner reports ANE versus Metal honestly
- [ ] Verify no silent fallback occurs

## Phase 5: CUDA Backend Bring-Up

### 5.1 CUDA Capability Layer

- [x] Create CUDA backend directory structure
- [x] Define CUDA device descriptor
- [x] Implement CUDA capability probe
- [x] Define CUDA planner-visible constraints

### 5.2 CUDA Planner and Execution

- [x] Implement CUDA planner
- [x] Implement CUDA packed weight path
- [x] Implement CUDA projector path
- [x] Implement CUDA prefill path
- [x] Make CUDA prefill consume staged token and modal buffers
- [x] Implement CUDA decode path
- [x] Persist a CUDA live-context buffer across prefill and decode
- [x] Make CUDA decode depend on persisted live-session context identity
- [x] Implement CUDA workspace handoff
- [x] Materialize a parked-session checkpoint artifact for CUDA sessions
- [x] Offload parked CUDA context buffers after checkpoint materialization
- [x] Add versioned checksum validation for CUDA live-context payloads
- [x] Bind CUDA live-context payloads to producer route and artifact lineage
- [x] Replace opaque CUDA context hashing with explicit state-lane payloads
- [x] Give CUDA live-context payloads explicit KV/decode-step semantics
- [x] Expand CUDA live-context payloads into a compact windowed state image
- [x] Add page-backed slot payloads to CUDA live-context state images
- [x] Widen CUDA live-context state images into compact KV-style key/value lane
  planes with per-page digests
- [x] Add per-page tensor-layout records to CUDA KV images and preserve decode
  generation on the page that changes
- [x] Add explicit per-page control records to CUDA KV images for owner,
  capacity, epochs, logical ids, and page flags
- [x] Add per-page tensor descriptor records to CUDA KV images for storage
  offsets, byte spans, and row strides

### 5.3 CUDA Validation

- [x] Validate compact page-table rotation and recycled-slot marking under
  decode window overflow
- [x] Validate one target multimodal flow on CUDA
- [ ] Compare API-level parity against Apple
- [x] Add reference-output checks where feasible

## Phase 6: Cache Hierarchy and Self-Optimization

### 6.1 Plan Cache

- [ ] Create `src/cache/mod_plan_cache.f90`
- [ ] Implement strict plan cache keys
- [ ] Implement in-memory plan cache
- [ ] Implement disk-backed plan cache
- [ ] Implement plan cache warming

### 6.2 Weight Cache

- [ ] Create `src/cache/mod_weight_cache.f90`
- [ ] Implement backend-packed weight identity
- [ ] Implement in-memory packed weight registry
- [ ] Implement disk-backed packed weight reuse

### 6.3 Session Cache

- [ ] Create `src/cache/mod_session_cache.f90`
- [x] Implement parked session identity
- [ ] Implement KV retention policy
- [ ] Implement safe eviction policy for inactive sessions

### 6.4 Multimodal Cache

- [ ] Create `src/cache/mod_mm_cache.f90`
- [ ] Implement projector output cache keys
- [ ] Implement reusable preprocessing cache keys
- [ ] Define invalidation rules for modality reuse

### 6.5 Exploration and Exploitation

- [x] Define candidate plan registry
- [x] Implement bounded exploration policy
- [x] Implement winner selection policy
- [x] Implement winner persistence
- [ ] Implement stale-evidence invalidation
- [x] Implement cold-versus-warm tracking

### 6.6 Cache Safety

- [ ] Implement double-check cache miss pattern under synchronization
- [ ] Prevent eviction of live session state
- [ ] Add cache metrics and counters
- [ ] Add cache debug reporting

## Phase 7: Multimodal Path Hardening

### 7.1 Input Contracts

- [ ] Define modality input types
- [ ] Define normalized projector input tensors
- [ ] Define caller-visible multimodal API expectations

### 7.2 Projector Runtime

- [ ] Implement projector workspace planning
- [ ] Implement embedding splice policy
- [ ] Implement projector cache integration
- [ ] Implement projector telemetry

### 7.3 Model-Specific Validation

- [ ] Add Qwen multimodal prompt-path tests
- [ ] Add Gemma multimodal prompt-path tests
- [ ] Validate projector output reuse where legal

## Phase 8: C ABI and Go Binding

### 8.1 C ABI

- [ ] Finalize `include/mizu.h`
- [x] Implement C ABI bindings in source
- [ ] Add C smoke test for lifecycle calls
- [ ] Add C smoke test for session flow

### 8.2 Go Binding

- [ ] Create Go module for bindings
- [ ] Wrap runtime lifecycle
- [ ] Wrap model lifecycle
- [ ] Wrap session flow
- [ ] Wrap multimodal attachment path
- [ ] Add Go smoke example

## Phase 9: Measurement and Hardening

### 9.1 Stage Metrics

- [x] Measure model load latency
- [x] Measure projector latency
- [x] Measure prefill latency
- [x] Measure decode latency
- [x] Measure park and resume latency

### 9.2 Cache and Optimization Metrics

- [ ] Record cache hit and miss rates by layer
- [ ] Record exploration versus exploitation counts
- [ ] Record winning plan convergence
- [ ] Record cold-to-warm improvement

### 9.3 Backend Validation

- [ ] Record Apple planner routing traces
- [ ] Record Apple ANE versus Metal outcomes
- [ ] Record CUDA parity checks
- [ ] Record CUDA throughput baselines

### 9.4 Hardening

- [ ] Add unsupported-backend failure tests
- [ ] Add cache invalidation tests
- [ ] Add session reuse tests
- [ ] Add multimodal regression tests

## Cross-Cutting Tasks

These tasks cut across all phases.

- [ ] Keep docs in sync with source reality
- [ ] Mark measured versus experimental behavior explicitly
- [ ] Keep ownership rules visible in code and headers
- [ ] Keep hot-path allocations visible and controlled
- [ ] Keep planner decisions inspectable
- [ ] Keep Go bindings thin

## Critical Path

If we want the shortest path to a working runtime, the likely critical path is:

- [ ] Phase 1 core contracts
- [ ] Phase 2 internal model dialect
- [ ] Phase 3 runtime skeleton
- [ ] Phase 4 Apple backend bring-up
- [ ] Phase 6 cache hierarchy and self-optimization
- [ ] Phase 7 multimodal hardening
- [ ] Phase 8 C ABI and Go binding

CUDA can progress in parallel after the core contracts are stable, but Apple is
the real gating path right now because ANE is a requirement.

## Nice to Have Later

- [ ] Add top-level roadmap view once implementation starts
- [ ] Add issue labels or task tags if the repo workflow expands
- [ ] Add machine-specific benchmark notes per target hardware
