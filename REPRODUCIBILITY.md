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

## Regenerate FourRoom H100 and H125 tasks

The generator runs the deterministic H-JEPA expert in a fresh simulator. It
takes a continuous window of exactly 100 or 125 actions whose endpoints are
three doors apart along the shortest path. Each released split contains 50
tasks, rendered with one active distractor. The first and last frames are the
start and goal images. Generation uses no Train or Val trajectories. H75 uses
the original H-JEPA task set.

From the FCP repository root, check out the simulator and install its
[dependencies](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5#1-installation)
in a Python 3.11 environment.

```bash
git clone https://github.com/kevinghst/H-JEPA.git ../H-JEPA
git -C ../H-JEPA checkout 97054ecaff83e13839195ec4b92d74c08a0127c5
python -m pip install -e '../H-JEPA[train,env]'
```

To regenerate the **released tasks**, download their JSON manifests and rerun
the expert with the recorded resets and crop indices. The reference HDF5
files and model weights are optional for this step.

```bash
mkdir -p evaluation_tasks
tasks_url=https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/resolve/00ccb6c343e053aac5c079bb96e9ee396e8aef9b/fourroom
for horizon in 100 125; do
  curl -fsSL "$tasks_url/H$horizon.json" -o "evaluation_tasks/H$horizon.json"
  python scripts/generate_fourroom_tasks.py --hjepa-root ../H-JEPA \
    --horizon "$horizon" --manifest "evaluation_tasks/H$horizon.json" \
    --output "regenerated/H$horizon.h5"
done
```

To **sample a new split**, omit the manifest and choose a sampling seed.
The released splits used seed 3072. H125 screens candidate lengths with the
same ego dynamics before rendering the full rollout.

```bash
python scripts/generate_fourroom_tasks.py --hjepa-root ../H-JEPA \
  --horizon 125 --seed 3073 --num-tasks 50 --output generated/H125.h5
```

Both commands write trajectory groups and a JSON task list alongside the HDF5
file. Each group contains `H + 1` observations and `H` actions. The model
gets 280 environment steps to solve H100 and 350 to solve H125. Use the
published manifests for a direct comparison with the reported results.
