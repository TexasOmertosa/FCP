# Reproducibility

This page records the fixed data partitions, random seeds, asset revisions, and the scope of the release check. The CSV files in [`data/splits/`](data/splits/) give the exact episode membership and are the source of truth for applying each split.

## Immutable release inputs

| Input | Pinned source |
|---|---|
| FCP generators, solvers, and matched priors | [Checkpoint download table](ASSETS.md#fcp-and-action-priors). |
| Fixed evaluation task archive | [`stardust8979/FCP-evaluation-tasks`](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/00ccb6c343e053aac5c079bb96e9ee396e8aef9b). |
| Native Cube and LIBERO task definitions | [`fcp/closed_loop/cube_tasks.json`](fcp/closed_loop/cube_tasks.json) and [`fcp/closed_loop/libero_tasks.json`](fcp/closed_loop/libero_tasks.json); these store selected episodes, frame offsets, reset seeds, group IDs, and initial-state hashes. |
| Training episode membership | [`data/splits/`](data/splits/). These files contain indices only, not demonstrations. |
| Frozen lower models and original demonstrations | See the pinned [lower-model checkpoints](ASSETS.md#frozen-lower-models) and [demonstration archives](ASSETS.md#original-demonstrations-and-benchmark-tasks). |

## Training partitions and seeds

The checked-in split files are the authoritative episode membership. “Split seed” identifies how a randomized partition or validation subset was selected; fixed upstream partitions have no newly sampled split seed.

| Environment | Training / validation / test partition | Split procedure and seed |
|---|---|---|
| PushT | 14,948 / 1,868 / 1,869 episodes | SAGE 80/10/10 episode split, seed 42; exact membership in `pusht_episode_split.csv`. The 440 full-task evaluation set is selected from the established validation pool and is separately pinned in the evaluation-task archive. |
| Cube | 8,000 / 1,000 / 1,000 episodes | One episode-level 80/10/10 permutation using NumPy `default_rng(42)`; exact membership in `cube_episode_split.csv`. |
| LIBERO Scene2 | 8,000 / 1,000 episodes | Reuses the SAGE training and validation membership; there is no additional random split. `scene2_train_val_episode_index.csv` records the partition, episode index, frame offset, and length. |
| LIBERO Caddy | 8,000 / 1,000 episodes | Reuses the SAGE training and validation membership; there is no additional random split. `caddy_train_val_episode_index.csv` records the partition, episode index, frame offset, and length. |
| AntMaze D3 | all 25,000 training episodes / 500 selected validation episodes | Train uses the complete `visual_antmaze_medium_explore_stitch_train.h5`. Validation candidates come from the separate `visual_antmaze_medium_stitch_val.h5`; seed 42 permutes eligible complete episodes, which are accumulated to the 201,000-transition budget, choosing the closer total at the boundary. The selected episode IDs are in `antmaze_episode_split.csv`. |
| FourRoom | 2,494 / 100 episodes | Whole episodes selected with seed 0 from `fourroom_tp35_d1.h5` and `fourroom_tp35_d1_val.h5`, subject to 1,000,000 and 40,000 transition budgets. Exact membership is in `fourroom_episode_split.csv`. The upper validation windows use 1,000 windows per offset without replacement with `SeedSequence([42, H, 20261101])`. |

The public training CLI's **FCP upper-model** default seed schedule is shown below. CLI flags `--seed`, `--init-seed`, `--noise-seed`, and `--val-seed` override these defaults and are saved in each training checkpoint. For PushT and FourRoom, the epoch's permutation seed is `seed + 1009 * epoch` (zero-based epoch); for Cube it is NumPy `default_rng(seed + epoch)`. Other environments use a CPU `torch.randperm` seeded with `seed + epoch`.

| Environment | FCG / FGSS shuffle seed | FCG / FGSS initialization seed | Training noise seed (FCG / FGSS) | Fixed validation-noise seed |
|---|---:|---:|---:|---:|
| PushT | 148 / 148 | 314159 / 314159 | 500000151 / 500000151 | 77148 |
| Cube | 148 / 148 | 314159 / 314159 | 148 / 148 | 77148 |
| Scene2 | 217 / 218 | 314159 / 314160 | 1000220 / 1000221 | 20261027 |
| Caddy | 217 / 218 | 314159 / 314160 | 1000220 / 1000221 | 20261027 |
| AntMaze D3 | 42 / 42 | 314159 / 314159 | 500000045 / 500000045 | 1209 |
| FourRoom | 44 / 44 | 314159 / 314160 | 44000132 / 44000132 | 77148 |

The matched action-prior trainer defaults to seed 149 outside LIBERO and 218 for LIBERO. Its fixed validation sampling seed is 1209. Generated-condition training uses the frozen FCG and fixed generated-condition cache; the cache base seed and draw stride are defined in `priors/cache.py` and recorded in its cache manifest.

The lower-model seeds are separate from the FCP upper-model seeds. PushT and Cube use the published LeWM checkpoints, and Scene2 and Caddy use the published SAGE checkpoints. The retrained AntMaze lower model uses seed 42; the retrained FourRoom lower model uses seed 44. Download both retrained lower checkpoints and their matching normalizers. [Frozen lower models](ASSETS.md#frozen-lower-models)

The CSVs fix which episodes belong to each partition. To rebuild the actual FCP training windows, encode those episodes with the listed frozen lower model and provide start/goal frame pairs to `fcp.prepare`; custom supervision requires the corresponding `--target-indices`. `fcp.prepare` consumes these arrays but does not itself choose or seed the pairs. Therefore, the episode split and seeds alone do not reproduce an exact training-window set unless the same pair-generation rule is followed.

These are the repository training CLI defaults. They specify how to rerun training with this public implementation; they do not imply that an arbitrary rerun will reproduce a released checkpoint bit-for-bit across different GPU architectures or kernel implementations. Each released checkpoint is instead identified by its SHA-256 and selected epoch in `fcp/selected_models.json`.

## Evaluation seeds

| Environment | Fixed evaluation tasks and seeds |
|---|---|
| PushT full task | 440 validation-derived tasks, evaluation seed 42. |
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
