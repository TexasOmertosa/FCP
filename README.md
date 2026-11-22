# Imagine Beyond, Plan the Next

**Future-Context Planning with Latent World Models**

[![FCP weights and action priors](https://img.shields.io/badge/Checkpoints-Hugging%20Face-ffc84a?logo=huggingface)](https://huggingface.co/stardust8979/FCP-weights) [![Evaluation datasets](https://img.shields.io/badge/Data-Evaluation-278060)](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks) [![Reproduction guide](https://img.shields.io/badge/Guide-Reproduction-315caa)](USAGE.md) [![Original downloads](https://img.shields.io/badge/Assets-Original%20sources-5c5379)](ASSETS.md)

## Overview

FCP predicts future states in the world model's latent space and uses them to select the next subgoal. The frozen world model scores actions toward that subgoal, and FCP replans as new observations arrive.

## Full-task PushT

![Full-task PushT comparison with the goal on the left](media/previews/pusht_full.gif)

## More comparisons

Each comparison uses the same start and goal for the displayed methods. The goal is on the left. After FCP reaches the goal, its panel holds the success frame while the baselines continue for **5 seconds**.

| Environment | Animated comparison |
| --- | --- |
| **PushT H150** | <img src="media/previews/pusht_H150.gif" alt="PushT H150 segmented comparison" width="650"> |
| **Cube H150** | <img src="media/previews/cube_H150.gif" alt="Cube H150 comparison" width="650"> |
| **LIBERO Scene2** | <img src="media/previews/scene2.gif" alt="Scene2 comparison" width="650"> |
| **LIBERO Caddy** | <img src="media/previews/caddy.gif" alt="Caddy comparison" width="650"> |
| **AntMaze D3** | <img src="media/previews/antmaze_D3.gif" alt="AntMaze D3 comparison" width="650"> |
| **FourRoom H125** | <img src="media/previews/fourroom_H125.gif" alt="FourRoom H125 comparison" width="650"> |

## Closed-loop success rates

Success rates (%) follow the paper’s main comparison. Segmented means average the six horizons from 25 to 150 steps. Values after ± are sample standard deviations across evaluation groups.

| Method | PushT full task | PushT segmented mean | Cube segmented mean | LIBERO Scene2 | LIBERO Caddy | AntMaze D3 | FourRoom H75 | FourRoom H100 | FourRoom H125 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCP-True | **86.82** | **90.72 ± 0.67** | **90.22 ± 1.77** | 61.33 ± 3.06 | 66.00 ± 5.29 | **96.00 ± 2.00** | **100.00** | **100.00** | 54.00 |
| FCP-Generated | 83.18 | 90.11 ± 1.42 | 89.61 ± 0.51 | **70.67 ± 1.15** | **74.67 ± 1.15** | 89.33 ± 4.62 | 80.00 | 98.00 | **56.00** |
| SAGE | 46.59 | 74.72 ± 1.54 | 82.00 ± 0.58 | 48.67 ± 1.15 | 58.00 ± 7.21 | — | — | — | — |
| H-JEPA | 35.00 | 48.22 ± 0.92 | 46.89 ± 2.47 | — | — | 70.00 ± 8.00 | 96.00 | 26.00 | 0.00 |
| HWM | 34.77 | 52.61 ± 1.13 | 52.39 ± 1.42 | — | — | 36.00 ± 3.46 | **100.00** | 4.00 | 0.00 |

## Quick start

```bash
git clone https://github.com/TexasOmertosa/FCP.git
cd FCP
python -m pip install -r requirements.txt
```

Use an encoded `test.npz` with `history` and `goal` arrays. Include `subgoal` and `future` to measure prediction error. The commands download the matching released weights. Replace `true` with `generated` to use the other training variant.

```bash
python -m fcp.generate --environment pusht --variant true \
  --data test.npz --samples 8 --output subgoals.npz
python -m fcp.evaluate --environment pusht --variant true \
  --data test.npz --samples 1 8
```

To run native PushT with the released checkpoints and tasks, install the packages marked for PushT in a separate Python environment. Omit `--trials` to evaluate the full set.

```bash
sed -n 's/^# pusht: //p' requirements.txt | xargs python -m pip install -r requirements.txt
python -m fcp.closed_loop.pusht --variant true --trials 193
```

## Training

### Train FCG and FGSS

Encode each demonstration split with the matching frozen world model. Save `latents` and `episode_offsets` in `train_episodes.npz` and `val_episodes.npz`. The offsets mark each episode start and the final frame count. Save the start and goal frame indices for each split in `train_pairs.npy` and `val_pairs.npy`. Use the same preparation step for your test split. [Frozen models and demonstration data](ASSETS.md)

Train one future-context generator (FCG), then fit a future-guided subgoal solver (FGSS) for each condition. True uses the demonstrated future during training. Generated uses predictions from the frozen FCG.

```bash
for split in train val; do
  python -m fcp.prepare --environment pusht \
    --episodes "${split}_episodes.npz" --pairs "${split}_pairs.npy" --output "${split}.npz"
done

python -m fcp.train --environment pusht --stage fcg \
  --train train.npz --val val.npz --output fcg.pt

for variant in true generated; do
  python -m fcp.train --environment pusht --stage fgss --condition "$variant" \
    --fcg-checkpoint fcg.pt --train train.npz --val val.npz --output "${variant}.pt"
done
```

Pass `--checkpoint true.pt` or `--checkpoint generated.pt` to the generation or evaluation command to use the corresponding locally trained solver.

### Train the action prior

Add frame-aligned `actions` and `lowdim` arrays to the PushT episode files. Cache eight-draw subgoal targets from the frozen planner, then train its action prior on the aligned expert actions. The example uses the True solver.

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

[Input formats and simulator versions](USAGE.md) · [Prior dimensions and training settings](priors/README.md)

## Repository layout

```text
FCP/
├── fcp/                  # FCG/FGSS models, data, training, generation, and evaluation
│   └── closed_loop/      # Action search and native PushT evaluation
├── priors/               # Action-prior model, data, cache, training, and weights
├── media/                # Project videos and animated previews
├── USAGE.md              # Commands and simulator setup
└── ASSETS.md             # Checkpoint and dataset sources
```
