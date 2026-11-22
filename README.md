# Imagine Beyond, Plan the Next

**Future-Context Planning with Latent World Models**

[![FCP weights and action priors](https://img.shields.io/badge/Checkpoints-Hugging%20Face-ffc84a?logo=huggingface)](https://huggingface.co/stardust8979/FCP-weights) [![Evaluation datasets](https://img.shields.io/badge/Data-Evaluation-278060)](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks)

[Checkpoints](#checkpoints) · [Data and splits](#data-and-splits) · [Training seeds](#training-seeds) · [Training](#training) · [Evaluation seeds](#evaluation-seeds) · [Evaluation](#evaluation)

## Overview

FCP predicts future states in the world model's latent space and uses them to select the next subgoal. The frozen world model scores actions toward that subgoal, and FCP replans as new observations arrive.

## Full-task PushT

![Full-task PushT comparison](media/previews/pusht_full.gif)

## More comparisons

| Environment | Animated comparison |
| --- | --- |
| **PushT H150** | <img src="media/previews/pusht_H150.gif" alt="PushT H150 segmented comparison" width="650"> |
| **Cube H150** | <img src="media/previews/cube_H150.gif" alt="Cube H150 comparison" width="650"> |
| **LIBERO Scene2** | <img src="media/previews/scene2.gif" alt="Scene2 comparison" width="650"> |
| **LIBERO Caddy** | <img src="media/previews/caddy.gif" alt="Caddy comparison" width="650"> |
| **AntMaze D3** | <img src="media/previews/antmaze_D3.gif" alt="AntMaze D3 comparison" width="650"> |
| **FourRoom H125** | <img src="media/previews/fourroom_H125.gif" alt="FourRoom H125 comparison" width="650"> |

## Closed-loop success rates

Success rates (%) use the released FCP checkpoints and their matched priors. Segmented means average the six horizons from 25 to 150 steps. Values after ± are sample standard deviations across evaluation groups.

| Method | PushT full task | PushT segmented mean | Cube segmented mean | LIBERO Scene2 | LIBERO Caddy | AntMaze D3 | FourRoom H75 | FourRoom H100 | FourRoom H125 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCP-True | **86.82** | **90.72 ± 0.67** | **90.22 ± 1.77** | 61.33 ± 3.06 | 66.00 ± 5.29 | **97.33 ± 1.15** | **100.00** | **100.00** | 48.00 |
| FCP-Generated | 83.18 | 90.11 ± 1.42 | 89.61 ± 0.51 | **70.67 ± 1.15** | **74.67 ± 1.15** | 89.33 ± 4.62 | 84.00 | 90.00 | **52.00** |
| SAGE | 46.59 | 74.72 ± 1.54 | 82.00 ± 0.58 | 48.67 ± 1.15 | 58.00 ± 7.21 | — | — | — | — |
| H-JEPA | 35.00 | 48.22 ± 0.92 | 46.89 ± 2.47 | — | — | 70.00 ± 8.00 | 96.00 | 26.00 | 0.00 |
| HWM | 34.77 | 52.61 ± 1.13 | 52.39 ± 1.42 | — | — | 36.00 ± 3.46 | **100.00** | 4.00 | 0.00 |

AntMaze and FourRoom use the retrained SAGE action priors paired with the released FCP models.

## Checkpoints

Use the same frozen lower model and FCG for both variants. Pair each FGSS with its corresponding action prior.

### Frozen lower models

| Environment | Lower model used | Checkpoint | Companion files |
|---|---|---|---|
| PushT | Published LeWM | [weights.pt](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/weights.pt) | [Config](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/config.json) |
| Cube | Published LeWM | [weights.pt](https://huggingface.co/quentinll/lewm-cube/blob/b0747c5002e86d2ce8f3cd8178004b97524c587d/weights.pt) | [Config](https://huggingface.co/quentinll/lewm-cube/blob/b0747c5002e86d2ce8f3cd8178004b97524c587d/config.json) |
| LIBERO Scene2 | Frozen SAGE world model | [weights.pt](https://huggingface.co/CLTRAY/SAGE/blob/d81d637d72884d8ef0ad80813d37c583ebfd33c8/libero_scene2/world_model/weights.pt) | [Config](https://huggingface.co/CLTRAY/SAGE/blob/d81d637d72884d8ef0ad80813d37c583ebfd33c8/libero_scene2/world_model/config.json) |
| LIBERO Caddy | Frozen SAGE world model | [weights.pt](https://huggingface.co/CLTRAY/SAGE/blob/d81d637d72884d8ef0ad80813d37c583ebfd33c8/libero_caddy/world_model/weights.pt) | [Config](https://huggingface.co/CLTRAY/SAGE/blob/d81d637d72884d8ef0ad80813d37c583ebfd33c8/libero_caddy/world_model/config.json) |
| AntMaze D3 | FCP retrained LeWM | [lower_E20.pt](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/lower_model/lower_E20.pt) | [Config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/lower_model/config.json) · [Normalizer](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/lower_model/train_normalizer.json) |
| FourRoom | FCP retrained LeWM | [fourroom_lewm.ckpt](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/lower_model/fourroom_lewm.ckpt) | [Config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/lower_model/config.json) · [Normalizer](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/lower_model/normalizer.pt) |

The AntMaze D3 and FourRoom downloads include the normalization statistics used in evaluation. The two models were retrained using the H-JEPA implementation. [Source revision](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5).

### FCP and action priors

FCG is shared by FCP-True and FCP-Generated within each environment. Use the upper model config and the prior metadata alongside the selected weights.

| Environment | FCG | FGSS | Action prior | Configuration |
|---|---|---|---|---|
| PushT | [Shared](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/fcg.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/fgss_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/fgss_generated.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/action_prior_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/action_prior_generated.safetensors) | [Upper config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/config.json) · [Prior metadata](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/pusht/action_priors.json) |
| Cube | [Shared](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/fcg.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/fgss_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/fgss_generated.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/action_prior_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/action_prior_generated.safetensors) | [Upper config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/config.json) · [Prior metadata](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/cube/action_priors.json) |
| LIBERO Scene2 | [Shared](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/fcg.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/fgss_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/fgss_generated.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/action_prior_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/action_prior_generated.safetensors) | [Upper config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/config.json) · [Prior metadata](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/scene2/action_priors.json) |
| LIBERO Caddy | [Shared](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/fcg.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/fgss_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/fgss_generated.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/action_prior_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/action_prior_generated.safetensors) | [Upper config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/config.json) · [Prior metadata](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/caddy/action_priors.json) |
| AntMaze D3 | [Shared](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/fcg.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/fgss_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/fgss_generated.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/action_prior_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/action_prior_generated.safetensors) | [Upper config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/config.json) · [Prior metadata](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/antmaze/action_priors.json) |
| FourRoom | [Shared](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/fcg.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/fgss_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/fgss_generated.safetensors) | [True](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/action_prior_true.safetensors) · [Generated](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/action_prior_generated.safetensors) | [Upper config](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/config.json) · [Prior metadata](https://huggingface.co/stardust8979/FCP-weights/blob/fb370f8f99e46183af641b6639f5dc85974ea067/fourroom/action_priors.json) |

## Data and splits

### Original demonstrations and benchmark tasks

| Environment | Demonstrations | Evaluation inputs |
|---|---|---|
| PushT | [`pusht_expert_train.h5.zst`](https://huggingface.co/datasets/quentinll/lewm-pusht/blob/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst) | [Full440 task data and index](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/00ccb6c343e053aac5c079bb96e9ee396e8aef9b/pusht) |
| Cube | [`cube_single_expert.tar.zst`](https://huggingface.co/datasets/quentinll/lewm-cube/blob/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst) | [Fixed episode IDs and start frames by seed and horizon](fcp/closed_loop/cube_tasks.json) |
| Scene2 | `libero_scene2_train.tar.part*` and `libero_scene2_validation.tar`. [SAGE training archives](https://huggingface.co/datasets/CLTRAY/SAGE-data/tree/7e87e5b0fee414a70945a310d08e18d2f92cffba/training) | [Scene2 evaluation archive](https://huggingface.co/datasets/CLTRAY/SAGE-data/blob/7e87e5b0fee414a70945a310d08e18d2f92cffba/evaluation/libero_scene2_evaluation.tar) |
| Caddy | `libero_caddy_train.tar.part*` and `libero_caddy_validation.tar`. [SAGE training archives](https://huggingface.co/datasets/CLTRAY/SAGE-data/tree/7e87e5b0fee414a70945a310d08e18d2f92cffba/training) | [Caddy evaluation archive](https://huggingface.co/datasets/CLTRAY/SAGE-data/blob/7e87e5b0fee414a70945a310d08e18d2f92cffba/evaluation/libero_caddy_evaluation.tar) |
| AntMaze | [Train HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/visual_antmaze_medium_explore_stitch_train.h5) · [Val HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/visual_antmaze_medium_stitch_val.h5) | [Original H-JEPA D3 tasks](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/eval_trajs/ant/expert_grid_d3_n50.pt) |
| FourRoom | [Train HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/fourroom_tp35_d1.h5) · [Val HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/fourroom_tp35_d1_val.h5) | [Original H-JEPA H75 tasks](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt) · [FCP H100 and H125 task data](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/00ccb6c343e053aac5c079bb96e9ee396e8aef9b/fourroom) |

Use the [SAGE extraction instructions](https://huggingface.co/datasets/CLTRAY/SAGE-data#extract) for the multipart LIBERO training archives.

### Episode partitions

The CSV files record the exact episode membership. For randomized partitions and validation subsets, the table also gives the selection seed.

| Environment | Training / validation / test partition | Split procedure and seed |
|---|---|---|
| PushT | 14,948 / 1,868 / 1,869 episodes | SAGE 80/10/10 episode split, seed 42. Exact membership in [PushT split CSV](data/splits/pusht_episode_split.csv). The 440 full-task evaluation set is selected from the established validation pool and is separately pinned in the evaluation-task archive. |
| Cube | 8,000 / 1,000 / 1,000 episodes | One episode-level 80/10/10 permutation using NumPy `default_rng(42)`. Exact membership in [Cube split CSV](data/splits/cube_episode_split.csv). |
| LIBERO Scene2 | 8,000 / 1,000 episodes | Uses the fixed SAGE training and validation membership. [Scene2 split CSV](data/splits/scene2_train_val_episode_index.csv) records the partition, episode index, frame offset, and length. |
| LIBERO Caddy | 8,000 / 1,000 episodes | Uses the fixed SAGE training and validation membership. [Caddy split CSV](data/splits/caddy_train_val_episode_index.csv) records the partition, episode index, frame offset, and length. |
| AntMaze D3 | all 25,000 training episodes / 500 selected validation episodes | Train uses the complete `visual_antmaze_medium_explore_stitch_train.h5`. Validation candidates come from `visual_antmaze_medium_stitch_val.h5`. Seed 42 permutes eligible complete episodes, which are accumulated to the 201,000-transition budget, choosing the closer total at the boundary. The selected episode IDs are in [AntMaze split CSV](data/splits/antmaze_episode_split.csv). |
| FourRoom | 2,494 / 100 episodes | Whole episodes selected with seed 0 from `fourroom_tp35_d1.h5` and `fourroom_tp35_d1_val.h5`, subject to 1,000,000 and 40,000 transition budgets. Exact membership is in [FourRoom split CSV](data/splits/fourroom_episode_split.csv). The upper validation windows use 1,000 windows per offset without replacement with `SeedSequence([42, H, 20261101])`. |

## Training seeds

The values below are the defaults of the public FCG and FGSS trainer. CLI flags `--seed`, `--init-seed`, `--noise-seed`, and `--val-seed` override these defaults and are saved in each training checkpoint. PushT and FourRoom use `seed + 1009 * epoch` for the permutation with zero-based epochs. Cube uses NumPy `default_rng(seed + epoch)`. The other environments use a CPU `torch.randperm` with `seed + epoch`.

| Environment | FCG / FGSS shuffle seed | FCG / FGSS initialization seed | Training noise seed (FCG / FGSS) | Fixed validation-noise seed |
|---|---:|---:|---:|---:|
| PushT | 148 / 148 | 314159 / 314159 | 500000151 / 500000151 | 77148 |
| Cube | 148 / 148 | 314159 / 314159 | 148 / 148 | 77148 |
| Scene2 | 217 / 218 | 314159 / 314160 | 1000220 / 1000221 | 20261027 |
| Caddy | 217 / 218 | 314159 / 314160 | 1000220 / 1000221 | 20261027 |
| AntMaze D3 | 42 / 42 | 314159 / 314159 | 500000045 / 500000045 | 1209 |
| FourRoom | 44 / 44 | 314159 / 314160 | 44000132 / 44000132 | 77148 |

The AntMaze and FourRoom lower LeWM training seeds are 42 and 44, respectively. The prior training seeds are 149 for PushT, Cube, AntMaze and FourRoom and 218 for Scene2 and Caddy. LIBERO prior validation sampling uses seed 1209. The following cache seeds are the public CLI defaults.

| Environment | FCG Train / Val cache seed | Action-prior Train / Val cache seed |
|---|---:|---:|
| PushT | 1209 / 77148 | 149000000 / 150000000 |
| Cube | 148 / 77148 | 149000000 / 150000000 |
| Scene2, Caddy | 1020261034 / 20261027 | 9026102900 / 9126102900 |
| AntMaze | 1209 / 1209 | 1209 / 1209 |
| FourRoom | 1209 / 77148 | 1209 / 1209 |

## Quick start

```bash
git clone https://github.com/TexasOmertosa/FCP.git
cd FCP
python -m pip install -r requirements.txt
```

Use an encoded `test.npz` with `history` and `goal` arrays. Use Python 3.10 or 3.11 for the core commands. Include `subgoal` and `future` to measure prediction error. The commands download the matching released weights. Replace `true` with `generated` to use the other training variant.

```bash
python -m fcp.generate --environment pusht --variant true \
  --data test.npz --samples 8 --output subgoals.npz
python -m fcp.evaluate --environment pusht --variant true \
  --data test.npz --samples 1 8
```

## Training

### Train FCG and FGSS

Encode each demonstration split with the matching frozen world model. Save `latents` and `episode_offsets` in `train_episodes.npz` and `val_episodes.npz`. Use the shape `[total_frames, D]` for latents. The offsets mark each episode start and the final frame count. Save an integer array of start and goal frame indices with shape `[N, 2]` for each split in `train_pairs.npy` and `val_pairs.npy`. Use the same preparation step for your test split.

The standard window constructor uses start frame `t`, goal frame `g`, subgoal `u = min(t + 10, g)`, and future frames `u + (j * (g - u) + 2) // 4` for `j = 1, 2, 3`. Pass `--target-indices` for runs with saved supervision frame indices. PushT, Cube and AntMaze use the current frame. Scene2, Caddy and FourRoom use offsets −10, −5 and 0, clipped to the episode start. LIBERO concatenates external-view latents before wrist-view latents. Rebuilding an exact training window set requires the original start/goal pair selection in addition to the episode split.

Train one future-context generator (FCG), then fit a future-guided subgoal solver (FGSS) for each condition. True uses the demonstrated future during training. Generated uses one prediction per window from the frozen FCG, cached before training.

```bash
for split in train val; do
  python -m fcp.prepare --environment pusht \
    --episodes "${split}_episodes.npz" --pairs "${split}_pairs.npy" --output "${split}.npz"
done

python -m fcp.train --environment pusht --stage fcg \
  --train train.npz --val val.npz --output fcg.pt

for split in train val; do
  python -m fcp.generate --environment pusht --output-kind future --split "$split" \
    --checkpoint fcg.pt --data "${split}.npz" --output "${split}_future.npz"
done

python -m fcp.train --environment pusht --stage fgss --condition true \
  --fcg-checkpoint fcg.pt --val-future-cache val_future.npz \
  --train train.npz --val val.npz --output true.pt
python -m fcp.train --environment pusht --stage fgss --condition generated \
  --fcg-checkpoint fcg.pt --future-cache train_future.npz --val-future-cache val_future.npz \
  --train train.npz --val val.npz --output generated.pt
```

Pass `--checkpoint true.pt` or `--checkpoint generated.pt` to the generation or evaluation command to use the corresponding locally trained solver.

The public trainer uses AdamW with learning rate `1e-3`, weight decay `1e-4`, five warm-up epochs, cosine decay to `1e-6`, and gradient clipping at 1. The default global batch is 65,536. The public CLI defaults to 200 epochs, with 300 for Generated FGSS on PushT and AntMaze. For four GPUs, replace `python -m fcp.train` with `torchrun --standalone --nproc-per-node=4 -m fcp.train`. Both FGSS variants are selected by prediction error on generated validation conditions.

### Train the action prior

Add frame-aligned `actions` in the simulator's coordinates before normalization. PushT, Cube, AntMaze and FourRoom also use `lowdim` arrays of width 11, 28, 27 and 2, respectively. The prior uses three history frames and predicts ten actions in two tokens of five. Supply `upper_latents` when the upper and prior histories use different representations. Cache eight-draw subgoal targets from the frozen planner, then train its action prior on the aligned expert actions. The example uses the True solver.

```bash
for split in train val; do
  python -m priors.prepare --environment pusht \
    --episodes "${split}_episodes.npz" --pairs "${split}_pairs.npy" --output "prior_rows/$split"
  python -m priors.cache --environment pusht --variant true --checkpoint true.pt \
    --data "prior_rows/$split" --split "$split" --output "prior_cache/true_$split"
done

python -m priors.train --environment pusht --variant true \
  --train prior_rows/train --val prior_rows/val \
  --train-cache prior_cache/true_train --val-cache prior_cache/true_val \
  --output prior_runs/pusht_true
```

Repeat the cache and training steps for Generated with `--variant generated`, `--checkpoint generated.pt` on the cache command, and separate output directories. To deploy the trained True pair in PushT, add `--checkpoint true.pt --prior-root prior_runs/pusht_true` to the closed-loop command.

Training uses AdamW for 30 epochs at a constant learning rate of `1e-4`, weight decay of `1e-4`, and gradient clipping at 1. The loss is trajectory negative log likelihood plus 0.05 times the lowest per-mode action L1 error. Each environment uses an eight-component mixture with a width-512, three-layer Transformer decoder and eight attention heads.

| Environment | Global batch | Seed | Validation selection |
|---|---:|---:|---|
| PushT, Cube, AntMaze, FourRoom | 128 | 149 | Minimum trajectory NLL |
| Scene2, Caddy | 1,024 | 218 | Minimum normalized action L1 over 128 samples per row, with validation seed 1209 |

For Scene2 or Caddy distributed training, replace `python -m priors.train` with `torchrun --standalone --nproc-per-node=4 -m priors.train`. The batch size is global. By default, action and low-dimensional statistics are fitted from Train rows. Supply `--normalization stats.json` to reuse the exact Train-only scaler for an existing protocol. The schema matches the `normalization` field in the published environment's `action_priors.json`.

## Evaluation seeds

| Environment | Fixed evaluation tasks and seeds |
|---|---|
| PushT full task | 440 validation-derived tasks, evaluation seed 42. |
| PushT segmented | H = 25, 50, 75, 100, 125, 150. Seeds 22, 32 and 42, with 100 tasks per horizon and seed. |
| Cube | H = 25, 50, 75, 100, 125, 150. Seeds 22, 32 and 42, with 100 tasks per horizon and seed. |
| Scene2 | 150 fixed queries, three groups of 50 with group seeds 8501, 8502 and 8503. Each query has a recorded physical reset seed. [Query manifest](fcp/closed_loop/libero_tasks.json) |
| Caddy | 150 fixed queries in three groups of 50 with explicit physical reset seeds. [Query manifest](fcp/closed_loop/libero_tasks.json) |
| AntMaze D3 | Original H-JEPA D3 tasks, evaluation seeds 42, 43 and 44, with 50 tasks per seed. |
| FourRoom | Fixed H75/H100/H125 task sets, 50 tasks per horizon, base seed 42. The H100 and H125 files are in the evaluation archive. |

The LIBERO evaluation uses PyTorch seed 42. For action proposals it uses `base + 1009 * query_index + 17 * elapsed`, with base 20260829 for Scene2 and 42 for Caddy. The [model registry](fcp/selected_models.json) records complete-draw counts and planner settings. The open-loop evaluation CLI defaults to seed 1209.

## Evaluation

### Simulator versions

Training and open-loop evaluation use encoded windows. Closed-loop evaluation uses the selected benchmark's native simulator for resets, observations, and action execution. Keep separate Python environments for incompatible simulator stacks, especially the different MuJoCo versions.

| Environment | Native environment | Simulator stack |
|---|---|---|
| PushT | Stable World-Model PushT | Stable World-Model 0.1.1, Pymunk 7.3.0, Pygame 2.6.1, Gymnasium 1.3.0 |
| Cube | Stable World-Model OGBench Cube | Stable World-Model 0.1.1, OGBench 1.2.1, MuJoCo 3.12.0, Gymnasium 1.3.0 |
| Scene2 | LIBERO Scene2 | LIBERO 0.1.0, robosuite 1.4.0, MuJoCo 2.3.7 |
| Caddy | LIBERO Caddy | LIBERO 0.1.0, robosuite 1.4.0, MuJoCo 2.3.7 |
| AntMaze D3 | H-JEPA Visual AntMaze | Stable World-Model 0.1.1, MuJoCo 3.14.0, Gymnasium 1.2.3 |
| FourRoom | H-JEPA FourRoom Distractors | Gymnasium 1.2.3, `swm/FourRoomDistractors-v0` from the H-JEPA source snapshot at `97054ec` |

PushT uses Pymunk for 2D physics. FourRoom implements its own 2D dynamics in the environment. For Cube, LIBERO, and AntMaze, match the MuJoCo version of the selected benchmark. See the upstream [Stable World-Model](https://github.com/galilai-group/stable-worldmodel), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [H-JEPA](https://github.com/kevinghst/H-JEPA/commit/97054ecaff83e13839195ec4b92d74c08a0127c5) projects for native environment sources.

### CUDA numerical settings

Parameters, velocity losses, Euler states, and averages of complete draws use FP32. The table gives the upper-model forward and the prior train/Val precision. CUDA attention and TF32 settings are implemented in [`fcp/numerics.py`](fcp/numerics.py).

| Environment | Upper train | Upper deploy | Upper for action-prior cache | Prior train / Val |
|---|---|---|---|---|
| PushT | BF16 / automatic SDPA | BF16 / efficient self, Flash cross | BF16 / efficient self, Flash cross | BF16 / BF16, Math |
| Cube | BF16 / efficient self, Flash cross | FP32 / native MHA and SDPA | FP32 / native MHA and SDPA | FP32 / FP32, native MHA and SDPA |
| Scene2 | BF16 / Math | FP32 / Math | BF16 / Math | BF16 / FP32, Math |
| Caddy | BF16 / Math | BF16 / Math | BF16 / Math | BF16 / FP32, Math |
| AntMaze | BF16 / Math | BF16 / Math | BF16 / Math | FP32 / FP32, Math |
| FourRoom | BF16 / Flash | BF16 / Flash | BF16 / Flash | FP32 / FP32, Math |

The CPU path uses FP32. Use the specified CUDA and simulator versions for comparisons with the archived GPU runs.

### Native PushT evaluation

Install the packages marked for PushT in [requirements.txt](requirements.txt) in a separate environment. The original full-task evaluation used Python 3.10, PyTorch 2.13.0 with CUDA 13.0, and an NVIDIA A100.

```bash
sed -n 's/^# pusht: //p' requirements.txt | xargs python -m pip install -r requirements.txt
python -m fcp.closed_loop.pusht --variant true --trials 193 --output runs/pusht_trial193.json
```

Omit `--trials` for all 440 tasks. Use `--variant generated` for the other configuration. The runner downloads the pinned task set, FCP weights, matched action prior, and official LeWM checkpoint. It preserves the original 16-task evaluation batches, including when reporting a single trial. It averages eight complete FCG→FGSS draws and uses the prior as the first bank in 30 rounds of CEM with 300 candidates and 30 elites. The output gives the first step above 0.85 coverage and the first native pose match. `--trace` includes the executed actions and the upper subgoals.

An A100 rerun of both full440 variants reproduced all 880 archived first-hit records. FCP-True completed 382/440 tasks (86.82%) and FCP-Generated completed 366/440 tasks (83.18%).

### Native Cube and LIBERO evaluation

Use a separate Python 3.10 environment for each simulator. The native Cube and LIBERO runners were checked on an A100 with PyTorch 2.13.0 and CUDA 13.0. Extract the [evaluation data](#original-demonstrations-and-benchmark-tasks) first.

```bash
# Run in the Cube environment.
sed -n 's/^# cube: //p' requirements.txt | xargs python -m pip install -r requirements.txt
```

For Cube, run each horizon (25, 50, 75, 100, 125, 150) with each evaluation seed (22, 32, 42) and variant.

```bash
python -m fcp.closed_loop.cube --variant true --horizon 25 --seed 22 \
  --dataset /path/to/cube_single_expert.h5 --output runs/cube_true_H25_seed22.json
```

Each Cube cell evaluates 100 fixed tasks in ten-task batches. Use `--chunk 0` to check the first ten tasks. The cell counts native zero-step successes and uses an action budget of `2H`. The last 30 actions target the final observation. The last search plans a ten-action segment. [Fixed Cube task manifest](fcp/closed_loop/cube_tasks.json)

For LIBERO, use the extracted suite evaluation directory containing `episodes/`. Use the SAGE and LIBERO source revisions below. The runner uses the pinned SAGE lower model and applies the recorded per-query OpenBLAS kernel and reset seeds. Set `MUJOCO_EGL_DEVICE_ID` to the physical rendering device when restricting CUDA visibility.

```bash
# Run in the LIBERO environment.
sed -n 's/^# libero: //p' requirements.txt | xargs python -m pip install -r requirements.txt
git clone https://github.com/PKU-ML/SAGE.git
git -C SAGE checkout fbdb018489e2cdf27f0a8ad692746db4be596b6e
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

python -m fcp.closed_loop.libero --environment scene2 --variant true \
  --dataset /path/to/libero_scene2_evaluation --sage-root ./SAGE \
  --libero-root ./LIBERO --output runs/scene2_true.json
```

Use `--environment caddy` with the Caddy archive. Use `--trials 1` to check one query. At each decision the matched prior proposes 64 plans, including its top mode. The frozen two-view lower model scores the plans once. Native success stops execution immediately. The 150-query summaries report success and the sample standard deviation over three groups of 50. The native PushT, Cube, and LIBERO commands take `--checkpoint` and `--prior-root` to evaluate a locally trained pair and its matching prior. [Fixed Scene2 and Caddy query manifests](fcp/closed_loop/libero_tasks.json)

### AntMaze and FourRoom evaluation

Use the fixed navigation tasks and the retrained LeWM checkpoints with their normalizers and matched priors. Run the public entry point with your navigation backend in place of `module:factory`. The backend supplies the simulator and the protocol-specific controller.

```bash
python -m fcp.evaluate --environment antmaze --variant true --backend module:factory
```

The factory receives `(environment, variant, backend_config, device)` and returns an object with the following methods.

- `tasks()` for the fixed task set and `reset(task)` for the goal latent, action budget, and initial success flag.
- `history()` for the latest encoded observation or history.
- `propose(target, goal, horizon, count, terminal, rng)` for the matching prior's candidate actions in planner coordinates. Include the prior mode among the LIBERO candidates.
- `rollout(actions)` for the frozen world model's terminal latent per candidate. FourRoom's projected-gradient planner requires differentiable rollouts and bounded actions in `[-1,1]`.
- `actions_for_execution(actions)` to convert the chosen action segment to the simulator's coordinates, followed by `step(action)` for the success and termination flags.

### Release verification

The public GitHub revision `f41ee0d` was cloned into a fresh directory and exercised with its published checkpoint and task downloads on an A100. Checks completed:

- PushT True, full-task query 193: the runner loaded all assets and returned the archived task outcome (success in 110 actions).
- Cube True, H25, seed 22, first 10-task batch: the runner loaded the public task and weight assets and completed all ten tasks.
- Scene2 True query 1 and Caddy True query 28: both native LIBERO runners loaded the fixed query and simulator assets and completed. Scene2 succeeded in 226 actions. Caddy reached its 370-action budget without success, matching the recorded per-query outcome.
- A two-window PushT open-loop evaluation was run twice with the same seed. Reported MSEs were identical. A one-epoch FCG/FGSS-True/FGSS-Generated training smoke run was also repeated: all three resulting checkpoints matched between runs.

The release check covers representative native evaluation tasks and repeated short training runs. The LIBERO action traces diverged from their archived counterparts after the first segment.

## Repository layout

```text
FCP/
├── fcp/                  # FCG/FGSS models, data, training, generation, and evaluation
│   └── closed_loop/      # Native benchmark evaluation and action search
├── priors/               # Action-prior model, data, cache, training, and weights
├── data/splits/          # Episode membership for training and validation
└── media/                # Project videos and animated previews
```
