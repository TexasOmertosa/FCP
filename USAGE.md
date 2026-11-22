# FCP training, generation, and evaluation

Each environment shares one future-context generator (FCG) and has two future-guided subgoal solvers (FGSS). FCP-True trains FGSS on demonstrated futures. FCP-Generated trains FGSS on frozen FCG predictions. Both use FCG predictions at deployment.

| Environment | CLI key | Shared FCG | True FGSS | Generated FGSS |
|---|---|---:|---:|---:|
| PushT | `pusht` | 191 | 139 | 280 |
| Cube | `cube` | 163 | 114 | 78 |
| LIBERO Scene2 | `scene2` | 187 | 198 | 185 |
| LIBERO Caddy | `caddy` | 154 | 197 | 178 |
| AntMaze D3 | `antmaze` | 175 | 181 | 275 |
| FourRoom | `fourroom` | 200 | 197 | 195 |

[Checkpoint and data sources](ASSETS.md) · [Evaluation task data](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks)

For fixed data split manifests, exact training/evaluation seeds, artifact revisions, and the fresh-checkout verification scope, see [Reproducibility](REPRODUCIBILITY.md).

## Repository layout

| Module | Purpose |
|---|---|
| [`fcp/model.py`](fcp/model.py) | FCG and FGSS flow networks, training objectives, and Euler integration. |
| [`fcp/data.py`](fcp/data.py), [`fcp/prepare.py`](fcp/prepare.py) | Construct and load windows from frozen demonstration latents. |
| [`fcp/train.py`](fcp/train.py), [`fcp/generate.py`](fcp/generate.py) | Train the two upper stages or generate future contexts and subgoals. |
| [`fcp/evaluate.py`](fcp/evaluate.py), [`fcp/metrics.py`](fcp/metrics.py) | Evaluate prediction error or run a supplied closed-loop adapter. |
| [`fcp/closed_loop/`](fcp/closed_loop) | Native PushT, Cube, Scene2, and Caddy evaluation and a general controller interface. |
| [`fcp/release.py`](fcp/release.py), [`fcp/selected_models.json`](fcp/selected_models.json) | Resolve asset revisions and benchmark settings. |
| [`priors/`](priors) | Build expert action windows, cache FCP conditions, and train the matching action prior. |

## Data preparation

Install `requirements.txt` in Python 3.10 or 3.11. Encode each split with the environment's frozen encoder. Create a file `episodes.npz` containing `latents` with shape `[total_frames, D]` and `episode_offsets` with the start of every episode plus the final frame count. Supply `pairs.npy` with integer start and goal row indices `[N, 2]` for that split.

```bash
python -m fcp.prepare --environment pusht --episodes episodes.npz --pairs pairs.npy --output train.npz
```

Use separate episode arrays and index pairs for validation and test. The public benchmark task data provide fixed evaluation inputs. Training requires demonstration data, split indices, and the corresponding frozen encoder. [Original downloads](ASSETS.md)

The resulting encoded NPZ windows contain `history` in `[N,D]` or `[N,T,D]`, `goal` and `subgoal` in `[N,D]`, and `future` in `[N,3,D]`. Generation reads `history` and `goal`. In the common window constructor, a start `t` and goal `g` give subgoal index `u=min(t+10,g)` and future indices `u+(j*(g-u)+2)//4` for `j=1,2,3`. Three-frame histories use offsets −10, −5, and 0, clipped to the episode start. The LIBERO latent order is external view followed by wrist view. PushT, Cube and AntMaze use the current observation only. Scene2, Caddy and FourRoom use three observations. For a run with saved labels, pass its exact frame indices via `--target-indices` or use its encoded NPZ directly.

## Train, generate, and test

Train a shared FCG first. Freeze it and cache one generated future per window. Train FGSS-True on demonstrated futures and FGSS-Generated on the fixed Train cache. Select both FGSS checkpoints by prediction error under generated validation conditions and share the Val cache across variants.

```bash
python -m fcp.train --environment pusht --stage fcg --train train.npz --val val.npz --output fcg.pt
python -m fcp.generate --environment pusht --output-kind future --split train --checkpoint fcg.pt --data train.npz --output train_future.npz
python -m fcp.generate --environment pusht --output-kind future --split val --checkpoint fcg.pt --data val.npz --output val_future.npz
python -m fcp.train --environment pusht --stage fgss --condition true --fcg-checkpoint fcg.pt --val-future-cache val_future.npz --train train.npz --val val.npz --output true.pt
python -m fcp.train --environment pusht --stage fgss --condition generated --fcg-checkpoint fcg.pt --future-cache train_future.npz --val-future-cache val_future.npz --train train.npz --val val.npz --output generated.pt

python -m fcp.generate --environment pusht --variant generated --data test.npz --output subgoals.npz
python -m fcp.evaluate --environment pusht --variant generated --data test.npz --samples 1 8
```

Use `--checkpoint generated.pt` with the last two commands for a locally fitted solver. The default is the selected release from the pinned Hugging Face revision. The trainer checks the cache against the encoded split and FCG checkpoint. For four-GPU training, replace `python -m fcp.train` with `torchrun --standalone --nproc-per-node=4 -m fcp.train`. The batch size is global and defaults to 65,536. The upper optimizer uses AdamW with learning rate `1e-3`, weight decay `1e-4`, five warm-up epochs, cosine decay to `1e-6`, and gradient clipping at 1. Use the original splits, frozen latents, and training records when comparing checkpoints from scratch. The released weights give a fixed starting point for evaluating the reported models.

### Evaluation seeds

| Environment | Native evaluation |
|---|---|
| PushT | 42 for full440. 22, 32, 42 for the six segmented horizons, with 100 tasks per seed and horizon. |
| Cube | 22, 32, 42, with 100 tasks per seed and horizon. |
| Scene2 | Groups 8501, 8502, 8503, 50 tasks each. The fixed query manifest records individual physical reset seeds. |
| Caddy | Three fixed groups of 50 queries. The fixed query manifest records individual physical reset seeds. |
| AntMaze | 42, 43, 44, with 50 D3 tasks per seed. |
| FourRoom | Base seed 42, with 50 tasks each at H75, H100 and H125. |

The open-loop evaluation CLI defaults to seed 1209.

## Train the action prior

Follow the [action-prior training guide](priors/README.md) after fitting FCG and FGSS. The prior receives fixed eight-draw subgoal targets and is fitted separately for each FCP variant.

## Simulator versions

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

## Native PushT evaluation

Install the packages marked for PushT in [requirements.txt](requirements.txt) in a separate environment. The original full-task evaluation used Python 3.10, PyTorch 2.13.0 with CUDA 13.0, and an NVIDIA A100.

```bash
sed -n 's/^# pusht: //p' requirements.txt | xargs python -m pip install -r requirements.txt
python -m fcp.closed_loop.pusht --variant true --trials 193 --output runs/pusht_trial193.json
```

Omit `--trials` for all 440 tasks. Use `--variant generated` for the other configuration. The runner downloads the pinned task set, FCP weights, matched action prior, and official LeWM checkpoint. It preserves the original 16-task evaluation batches, including when reporting a single trial. It averages eight complete FCG→FGSS draws and uses the prior as the first bank in 30 rounds of CEM with 300 candidates and 30 elites. The output gives the first step above 0.85 coverage and the first native pose match. `--trace` includes the executed actions and the upper subgoals. [Frozen PushT LeWM checkpoint](https://huggingface.co/quentinll/lewm-pusht/tree/22b330c28c27ead4bfd1888615af1340e3fe9052)

An A100 rerun of both full440 variants reproduced all 880 archived first-hit records. FCP-True completed 382/440 tasks (86.82%) and FCP-Generated completed 366/440 tasks (83.18%).

## Native Cube and LIBERO evaluation

Use a separate Python 3.10 environment for each simulator. The native Cube and LIBERO runners were checked on an A100 with PyTorch 2.13.0 and CUDA 13.0. Extract the [evaluation data](ASSETS.md#original-demonstrations-and-benchmark-tasks) first.

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

In the release check, the ten Cube H25 tasks in the first seed-22 batch matched the archived first-hit steps. The tested Scene2 and Caddy queries matched their archived outcomes and execution lengths. Their action traces diverged after the first segment. The release check covered one LIBERO query per suite.

## Navigation weights and evaluation

The released AntMaze and FourRoom checkpoints are paired with individually retrained SAGE Transformer-GMM priors from November 22. The table gives the archived native reruns for those exact priors.

| Variant | AntMaze D3 | FourRoom H75 | FourRoom H100 | FourRoom H125 |
|---|---:|---:|---:|---:|
| FCP-True | 146/150, 97.33 ± 1.15% | 50/50, 100% | 50/50, 100% | 24/50, 48% |
| FCP-Generated | 134/150, 89.33 ± 4.62% | 42/50, 84% | 45/50, 90% | 26/50, 52% |

Download the retrained LeWM checkpoint and normalizer for each navigation environment. Evaluating the native protocol also requires the corresponding navigation controller. See the backend interface below. [Navigation lower models](ASSETS.md#frozen-lower-models) · [FCP and action priors](ASSETS.md#fcp-and-action-priors)

## Connect other simulators

The general closed-loop entry point is `python -m fcp.evaluate --environment <key> --variant <variant> --backend module:factory`. This interface lets a caller supply a simulator, frozen encoder and dynamics, native task reset and success rules, and action-prior inference. Use the native commands above for the corresponding paper evaluation protocols. [Asset sources](ASSETS.md)

The backend factory receives `(environment, variant, backend_config, device)` and implements the following methods.

- `tasks()` for the fixed task set and `reset(task)` for the goal latent, action budget, and initial success flag.
- `history()` for the latest encoded observation or history.
- `propose(target, goal, horizon, count, terminal, rng)` for the matching prior's candidate actions in planner coordinates. Include the prior mode among the LIBERO candidates.
- `rollout(actions)` for the frozen world model's terminal latent per candidate. FourRoom's projected-gradient planner requires differentiable rollouts and bounded actions in `[-1,1]`.
- `actions_for_execution(actions)` to convert the chosen action segment to the simulator's coordinates, followed by `step(action)` for the success and termination flags.
