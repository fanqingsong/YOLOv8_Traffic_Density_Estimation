# 0002. Separate packages for functional stages

- Status: Accepted
- Date: 2026-09-20
- Deciders: project maintainers
- Extends: ADR 0001

## Context

The Kaggle download, dataset preparation, and model training stages originally
lived in three procedural scripts. Environment access, file-system operations,
vendor calls, and CLI error handling were mixed together, which made the stages
difficult to test without Kaggle credentials or a YOLO runtime.

ADR 0001 allows those scripts as legacy exceptions but requires new code to move
toward modular packages and encapsulated classes.

## Decision

Each functional stage has its own package under `scripts/`:

- `scripts/kaggle_download/`
- `scripts/dataset_preparation/`
- `scripts/model_training/`

Each package owns its environment configuration through `from_env()`, exposes a
small public operation, and keeps vendor objects private. Package imports remain
lightweight; Ultralytics is imported only when its production backend is used.

The original `scripts/download_kaggle_dataset.py`,
`scripts/prepare_dataset.py`, and `scripts/train_model.py` files remain as thin
compatibility launchers. Compose service names, commands, paths, environment
variables, and one-shot `docker compose run --rm` usage remain stable.

Training defaults to `TRAIN_DEVICE=cpu`. GPU training continues to require the
`compose.gpu.yaml` override.

## Consequences

- Stage behavior can be tested with temporary directories and test doubles,
  without a network connection, model download, or GPU.
- New stage behavior belongs in the corresponding package, not in a launcher.
- Users can run either the compatibility scripts or package module entry points.
- The three legacy procedural exceptions in ADR 0001 are considered migrated;
  ADR 0001 remains otherwise unchanged.

## Compliance

- One capability and one primary class per module.
- Environment reads are confined to stage configuration classes.
- Public APIs return project values and do not expose Kaggle or Ultralytics
  objects.
- Compose stages remain independent one-shot services.
