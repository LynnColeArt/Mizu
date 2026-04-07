# Mizu

Mizu is an experimental local inference runtime for multimodal decoder models.

The project is intentionally narrow:

- Apple ANE is a priority target
- CUDA is a first-class peer target
- the public surface stays small through a C ABI
- the control plane is written primarily in Fortran
- the runtime is designed to self-optimize for speed through measured reuse

Current implementation status:

- core Fortran runtime types and lifecycle scaffolding are in place
- the public C header exists in `include/mizu.h`
- model manifests, cache keys, and session flow are implemented as scaffolds
- route-aware optimization and persisted cache metadata are implemented
- runtime backend inventory and capability-probe scaffolding exist for Apple and
  CUDA
- CUDA planner scaffolding now emits route-specific plan and weight-pack records
  and materializes stub payload files under `cache_root`
- CUDA prefill and decode now run through backend-owned stub executors using the
  materialized payload records
- Apple and CUDA execution backends are not implemented yet

Build and test:

- `make test`

Documentation:

- [Architecture](./docs/ARCHITECTURE.md)
- [API Spec](./docs/API_SPEC.md)
- [Project Plan](./docs/PROJECT_PLAN.md)
- [Task List](./docs/TASK_LIST.md)
- [Current State](./docs/CURRENT_STATE.md)
- [Style Guide](./STYLE_GUIDE.md)
