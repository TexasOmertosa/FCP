# Reproducibility

This page lists the release inputs, evaluation settings, and verification scope.

## Immutable release inputs

| Input | Pinned source |
|---|---|
| FCP generators, solvers, and matched priors | [Checkpoint download table](ASSETS.md#fcp-and-action-priors). |
| Fixed evaluation task archive | [`stardust8979/FCP-evaluation-tasks`](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/00ccb6c343e053aac5c079bb96e9ee396e8aef9b). |
| Native Cube and LIBERO task definitions | [`fcp/closed_loop/cube_tasks.json`](fcp/closed_loop/cube_tasks.json) and [`fcp/closed_loop/libero_tasks.json`](fcp/closed_loop/libero_tasks.json); these store selected episodes, frame offsets, reset seeds, group IDs, and initial-state hashes. |
| Frozen lower models and original demonstrations | See the pinned [lower-model checkpoints](ASSETS.md#frozen-lower-models) and [demonstration archives](ASSETS.md#original-demonstrations-and-benchmark-tasks). |

## Data preparation

Use the preparation commands and training defaults in the repository. To reconstruct the training windows, encode the demonstrations with the matching frozen model and provide the original start and goal frame pairs plus any saved supervision indices.

## Evaluation seeds

| Environment | Fixed evaluation tasks and seeds |
|---|---|
| PushT full task | 440 fixed tasks, evaluation seed 42. |
| PushT segmented | H = 25, 50, 75, 100, 125, 150; seeds 22, 32, 42; 100 tasks per horizon and seed. |
| Cube | H = 25, 50, 75, 100, 125, 150; seeds 22, 32, 42; 100 tasks per horizon and seed. |
| Scene2 | 150 fixed queries, three groups of 50 with group seeds 8501, 8502, 8503; each query's physical reset seed is in `libero_tasks.json`. |
| Caddy | 150 fixed queries in `libero_tasks.json`; each query's physical reset seed is stored explicitly. |
| AntMaze D3 | Original H-JEPA D3 tasks, evaluation seeds 42, 43, 44; 50 tasks per seed. |
| FourRoom | Fixed H75/H100/H125 task sets, 50 tasks per horizon, base seed 42; additional H100/H125 task data are pinned in the evaluation archive. |

The action-prior proposal RNG is also fixed by the native runners. LIBERO uses `base + 1009 * query_index + 17 * elapsed`, with base 20260829 for Scene2 and 42 for Caddy. The selected model configuration records complete-draw counts and planner settings.

## Verification performed

The public GitHub revision `f41ee0d` was cloned into a fresh directory and exercised with its published checkpoint and task downloads on an A100. Checks completed:

- PushT True, full-task query 193: the runner loaded all assets and returned the archived task outcome (success; 110 executed actions).
- Cube True, H25, seed 22, first 10-task batch: the runner loaded the public task and weight assets and completed all ten tasks.
- Scene2 True query 1 and Caddy True query 28: both native LIBERO runners loaded the fixed query and simulator assets and completed. Scene2 succeeded in 226 actions; Caddy reached its 370-action budget without success, matching the recorded per-query outcome.
- A two-window PushT open-loop evaluation was run twice with the same seed. Reported MSEs were identical. A one-epoch FCG/FGSS-True/FGSS-Generated training smoke run was also repeated: all three resulting checkpoint SHA-256 hashes matched between runs.

The release check covers representative native evaluation tasks and repeated short training runs. AntMaze and FourRoom evaluation also requires the corresponding native navigation controllers. [Navigation evaluation setup](USAGE.md#navigation-weights-and-evaluation)

## Selected navigation solver update

AntMaze and FourRoom selected FCP results now both use matched SAGE-prior CEM300/30/30. [Verified counts, source receipts, prior hashes and budgets](results/navigation_cem/README.md). FourRoom replaces its previous GradientSolver results; AntMaze retains its matching CEM results. All other environment and baseline protocols remain unchanged.
