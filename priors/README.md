# Action-prior training

Train a separate prior for each FCP variant. Keep that variant's upper weights frozen. Each training or validation row pairs its mean subgoal from eight complete FCG→FGSS draws with the corresponding expert action chunk. Both FCP variants use the same demonstration actions.

## Prepare aligned demonstrations

For each split, use the episode NPZ and the start/goal pairs described in the [data preparation guide](../USAGE.md#data-preparation). Add two arrays to the NPZ.

- `actions` has one row per frame. Use the benchmark's action coordinates before the prior's normalization. The action at frame `t` leads to frame `t+1`.
- `lowdim` contains the benchmark's frame-aligned low-dimensional inputs for PushT, Cube, AntMaze, and FourRoom. It has 11, 28, 27, and 2 columns, respectively. Scene2 and Caddy use visual inputs.

The prepared prior input contains three history frames, the far goal, and ten supervised actions beginning at the window start. The ten actions form two five-action tokens. When prior history and upper-model history come from different representations, include frame-aligned `upper_latents` to populate the separate upper history array.

```bash
python -m priors.prepare --environment pusht --episodes train_episodes.npz \
  --pairs train_pairs.npy --output prior_rows/train
python -m priors.prepare --environment pusht --episodes val_episodes.npz \
  --pairs val_pairs.npy --output prior_rows/val
```

## Cache subgoals and fit the prior

```bash
python -m priors.cache --environment pusht --variant true --split train \
  --data prior_rows/train --output prior_cache/true_train
python -m priors.cache --environment pusht --variant true --split val \
  --data prior_rows/val --output prior_cache/true_val
python -m priors.train --environment pusht --variant true \
  --train prior_rows/train --val prior_rows/val \
  --train-cache prior_cache/true_train --val-cache prior_cache/true_val \
  --output prior_runs/pusht_true
```

The cache uses the selected release by default. Pass `--checkpoint <fgss.pt>` to cache conditions from locally trained FCP weights from `python -m fcp.train`. Repeat the cache and training commands with `--variant generated` and separate output paths for the other FCP pair. Each cache records its split, data checksums, upper-weight checksum, seed, and complete draw count.

Training uses AdamW for 30 epochs at a constant learning rate of `1e-4`, weight decay of `1e-4`, and gradient clipping at 1. The loss is trajectory negative log likelihood plus 0.05 times the lowest per-mode action L1 error. Each environment uses an eight-component mixture with a width-512, three-layer Transformer decoder and eight attention heads.

| Environment | Global batch | Seed | Validation selection |
|---|---:|---:|---|
| PushT, Cube, AntMaze, FourRoom | 128 | 149 | Minimum trajectory NLL |
| Scene2, Caddy | 1,024 | 218 | Minimum normalized action L1 over 128 samples per row, with validation seed 1209 |

For Scene2 or Caddy distributed training, replace `python -m priors.train` with `torchrun --standalone --nproc-per-node=4 -m priors.train`. The batch size is global. By default, action and low-dimensional statistics are fitted from Train rows. Supply `--normalization stats.json` to reuse the exact Train-only scaler for an existing protocol. The schema matches the `normalization` field in the published environment's `action_priors.json`.

The selected prior is saved as `best.safetensors` with `best.json`. For native PushT, pass `--prior-root prior_runs/pusht_true` to the benchmark runner. Add `--checkpoint <fgss.pt>` when the prior was trained against local upper weights.

The prior architecture adapts the SAGE Transformer mixture. [SAGE](https://github.com/PKU-ML/SAGE) · [SAGE license](LICENSE-SAGE)
