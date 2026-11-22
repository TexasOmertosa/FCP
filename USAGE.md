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

## Repository layout

| Module | Purpose |
|---|---|
| [`fcp/model.py`](fcp/model.py) | FCG and FGSS flow networks, training objectives, and Euler integration. |
| [`fcp/data.py`](fcp/data.py), [`fcp/prepare.py`](fcp/prepare.py) | Construct and load windows from frozen demonstration latents. |
| [`fcp/train.py`](fcp/train.py), [`fcp/generate.py`](fcp/generate.py) | Train the two upper stages or generate future contexts and subgoals. |
| [`fcp/evaluate.py`](fcp/evaluate.py), [`fcp/metrics.py`](fcp/metrics.py) | Evaluate prediction error or run a supplied closed-loop adapter. |
| [`fcp/closed_loop/`](fcp/closed_loop) | Action search and native PushT full-task evaluation. |
| [`fcp/release.py`](fcp/release.py), [`fcp/selected_models.json`](fcp/selected_models.json) | Resolve asset revisions and benchmark settings. |
| [`priors/`](priors) | Build expert action windows, cache FCP conditions, and train the matching action prior. |

## Data preparation

Install `requirements.txt` in Python 3.10 or 3.11. Encode each split with the environment's frozen encoder. Create a file `episodes.npz` containing `latents` with shape `[total_frames, D]` and `episode_offsets` with the start of every episode plus the final frame count. Supply `pairs.npy` with integer start and goal row indices `[N, 2]` for that split.

```bash
python -m fcp.prepare --environment pusht --episodes episodes.npz --pairs pairs.npy --output train.npz
```

Use separate episode arrays and index pairs for validation and test. The public benchmark task data provide fixed evaluation inputs. Training requires demonstration data, split indices, and the corresponding frozen encoder. [Original downloads](ASSETS.md)

The resulting encoded NPZ windows contain `history` in `[N,D]` or `[N,T,D]`, `goal` and `subgoal` in `[N,D]`, and `future` in `[N,3,D]`. Generation only reads `history` and `goal`. For a window starting at `t` and ending at `g`, use `u=min(t+10,g)` for the subgoal and `u+(j*(g-u)+2)//4` for future indices `j=1,2,3`. Three-frame histories use offsets −10, −5, and 0, clipped to the episode start. The LIBERO latent order is external view followed by wrist view. PushT, Cube and AntMaze use the current observation only. Scene2, Caddy and FourRoom use three observations.

## Train, generate, and test

Train a shared FCG first and reuse that checkpoint for both FGSS variants.

```bash
python -m fcp.train --environment pusht --stage fcg --train train.npz --val val.npz --output fcg.pt
python -m fcp.train --environment pusht --stage fgss --condition true --fcg-checkpoint fcg.pt --train train.npz --val val.npz --output true.pt
python -m fcp.train --environment pusht --stage fgss --condition generated --fcg-checkpoint fcg.pt --train train.npz --val val.npz --output generated.pt

python -m fcp.generate --environment pusht --variant generated --data test.npz --output subgoals.npz
python -m fcp.evaluate --environment pusht --variant generated --data test.npz --samples 1 8
```

Use `--checkpoint generated.pt` with the last two commands for a locally fitted solver. Otherwise the commands download the selected release from its pinned Hugging Face revision. To cache Generated training conditions, use `python -m fcp.generate --output-kind future --checkpoint fcg.pt --data train.npz` and pass its output to `python -m fcp.train --future-cache`. The trainer checks the cache against the exact training split and FCG checkpoint.

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

Omit `--trials` for all 440 tasks. Use `--variant generated` for the other FCP configuration. The runner downloads the pinned task set, FCP weights, action prior, and official LeWM checkpoint. It loads each task's recorded start and goal, averages eight complete FCG→FGSS draws, scores the prior's action proposals with LeWM, and replans after each executed action segment. The output includes the first step with block coverage above 0.85 and the first native pose match. The native runner replans from the current observations. In the release check, trial 193 succeeded in 94 actions on an A100. The archived video records 110 actions for that trial. [Frozen PushT LeWM checkpoint](https://huggingface.co/quentinll/lewm-pusht/tree/22b330c28c27ead4bfd1888615af1340e3fe9052)

## Connect other simulators

The general closed-loop entry point is `python -m fcp.evaluate --environment <key> --variant <variant> --backend module:factory`. Cube, Scene2, Caddy, AntMaze, and FourRoom require a native adapter that supplies the matching frozen encoder and dynamics, exact task reset and success rules, and action-prior inference. The benchmark settings in [`fcp/selected_models.json`](fcp/selected_models.json) provide the published search sizes, evaluation seeds, and action budgets. The source links give the exact checkpoints and task inputs. [Download checkpoints and data](ASSETS.md)

The backend factory receives `(environment, variant, backend_config, device)` and implements the following methods.

- `tasks()` for the fixed task set and `reset(task)` for the goal latent, action budget, and initial success flag.
- `history()` for the latest encoded observation or history.
- `propose(target, goal, horizon, count, terminal, rng)` for the matching prior's candidate actions in planner coordinates. Include the prior mode among the LIBERO candidates.
- `rollout(actions)` for the frozen world model's terminal latent per candidate. FourRoom's projected-gradient planner requires differentiable rollouts and bounded actions in `[-1,1]`.
- `actions_for_execution(actions)` to convert the chosen action segment to the simulator's coordinates, followed by `step(action)` for the success and termination flags.
