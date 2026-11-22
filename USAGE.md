# FCP training, generation, and evaluation

Training, generation, and evaluation code for the twelve released FCP configurations. Within each environment, FCP-True trains the solver on demonstrated futures and FCP-Generated trains it on frozen FCG predictions. Both use generated futures at inference.

| Environment | CLI key | Shared FCG | True FGSS | Generated FGSS |
|---|---|---:|---:|---:|
| PushT | `pusht` | 191 | 139 | 280 |
| Cube | `cube` | 163 | 114 | 78 |
| LIBERO Scene2 | `scene2` | 187 | 198 | 185 |
| LIBERO Caddy | `caddy` | 154 | 197 | 178 |
| AntMaze D3 | `antmaze` | 175 | 181 | 275 |
| FourRoom | `fourroom` | 200 | 197 | 195 |

[Model weights](https://huggingface.co/stardust8979/FCP-weights) · [Evaluation task data](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks)

## Train, generate, and test

Use Python 3.11+ and install `requirements.txt`. Supply encoded NPZ windows with `history` in `[N,D]` or `[N,T,D]`, `goal` and `subgoal` in `[N,D]`, and `future` in `[N,3,D]`. Generation only reads `history` and `goal`. For a window starting at `t` and ending at `g`, use `u=min(t+10,g)` for the subgoal and `u+(j*(g-u)+2)//4` for future indices `j=1,2,3`. Three-frame histories use offsets −10, −5, and 0, clipped to the episode start. The LIBERO latent order is external view followed by wrist view.

```bash
python train.py --environment pusht --stage fcg --train train.npz --val val.npz --output fcg.pt
python train.py --environment pusht --stage fgss --condition true --fcg-checkpoint fcg.pt --train train.npz --val val.npz --output true.pt
python train.py --environment pusht --stage fgss --condition generated --fcg-checkpoint fcg.pt --train train.npz --val val.npz --output generated.pt

python generate.py --environment pusht --variant generated --data test.npz --output subgoals.npz
python test.py --environment pusht --variant generated --data test.npz --samples 1 8
```

Use `--checkpoint generated.pt` with the last two commands for a locally fitted solver. Otherwise the commands download the selected release from its pinned Hugging Face revision. To cache Generated training conditions, use `generate.py --output-kind future --checkpoint fcg.pt --data train.npz` and pass its output to `train.py --future-cache`. The trainer checks the cache against the exact training split and FCG checkpoint.

## Closed-loop evaluation

Run `test.py --environment <key> --variant <variant> --backend module:factory`. The released setup averages eight complete FCG→FGSS draws and uses environment-specific action search. `selected_models.json` gives the defaults. The factory receives `(environment, variant, backend_config, device)` and connects the caller's simulator, frozen encoder and dynamics, and action prior. It supplies:

- `tasks()` for the fixed task set and `reset(task)` returning `(goal_latent, budget, initial_success)`.
- `history()` for the latest encoded observation or history.
- `propose(target, goal, horizon, count, terminal, rng)` to produce the matching prior's candidate action sequences in planner coordinates, with the prior mode included for LIBERO.
- `rollout(actions)` to score candidates through frozen dynamics. FourRoom requires differentiable rollouts and bounded actions in `[-1,1]`.
- `actions_for_execution(actions)` to convert and clip the selected segment in native coordinates and `step(action)` returning `(success, done)`.

The standalone upper trainer and evaluator expose the method and selected checkpoints. Use the matching action-prior file and normalization from the model repository with the original environment backend.
