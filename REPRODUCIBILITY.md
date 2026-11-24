# Reproducibility

Use the following weights, datasets and settings to evaluate the released models.

## Weights and datasets

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

## Action planning

| Environment | Action selection | Execution |
|---|---|---|
| PushT | Matched prior + CEM, 300 candidates, 30 rounds, 30 elites | 10 steps; last30 uses final goal; terminal5+5 |
| Cube | Matched prior + CEM, 300 candidates, 30 rounds, 30 elites | 10 steps; last30 uses final goal; terminal10 |
| AntMaze | Matched SAGE prior + CEM, 300 candidates, 30 rounds, 30 elites | 10 steps; last30 uses final goal; terminal5+5; budget350 |
| FourRoom | Matched SAGE prior + bounded CEM, 300 candidates, 30 rounds, 30 elites | 10 steps; last30 uses final goal; terminal10; budgets210/280/350 |
| Scene2, Caddy | 64 matched-prior candidates, ranked once by the frozen world model | 10 steps; last30 uses final goal |

All selected FCP evaluations average eight complete FCG→FGSS samples. Keep the model-specific precision, attention settings and normalization in the code and configuration.

Use the commands in [Usage](USAGE.md). AntMaze and FourRoom require a native navigation backend implementing the documented simulator interface.
