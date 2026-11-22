# Imagine Beyond, Plan the Next

**Future-Context Planning with Latent World Models**

FCP predicts a future task context and uses it to select the next subgoal for closed-loop visual planning.

[Model weights](https://huggingface.co/stardust8979/FCP-weights) · [Evaluation datasets](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks) · [Training, generation, and evaluation](USAGE.md)

## Full-task PushT

[![Full-task PushT with the goal at left. FCP-True succeeds while SAGE and H-JEPA fail.](media/previews/pusht_full.gif)](media/pusht_full.mp4)

FCP-True completes the task in **110 actions**. SAGE and H-JEPA reach the **300-action** budget without completing the same task. [Watch full video](media/pusht_full.mp4)

## More environments

The videos show selected trials with the same initial state and goal for the compared methods. The left panel shows the goal. The AntMaze and FourRoom examples use the latest released navigation priors.

| Environment and outcome | Full comparison video |
| --- | --- |
| **Cube H150**<br>FCP-True succeeds. SAGE and H-JEPA fail. | [<img src="media/previews/cube_H150.jpg" alt="Cube H150 final observations" width="450">](media/cube_H150.mp4) |
| **LIBERO Scene2**<br>FCP-True succeeds. SAGE fails. | [<img src="media/previews/scene2.jpg" alt="Scene2 final observations" width="450">](media/scene2.mp4) |
| **LIBERO Caddy**<br>FCP-True succeeds. SAGE fails. | [<img src="media/previews/caddy.jpg" alt="Caddy final observations" width="450">](media/caddy.mp4) |
| **AntMaze D3**<br>FCP-True succeeds at 213 actions. H-JEPA fails. | [<img src="media/previews/antmaze_D3.jpg" alt="AntMaze D3 final observations" width="450">](media/antmaze_D3.mp4) |
| **FourRoom H125**<br>FCP-True succeeds at 229 actions. H-JEPA fails. | [<img src="media/previews/fourroom_H125.jpg" alt="FourRoom H125 final observations" width="450">](media/fourroom_H125.mp4) |

## Downloads

- FCP-True and FCP-Generated upper weights for all six environments, with one shared FCG, two FGSS checkpoints, and both matching action priors per environment. [Hugging Face model repository](https://huggingface.co/stardust8979/FCP-weights)
- 440 full-task PushT trajectories, 50 AntMaze D3 tasks, and 50 FourRoom tasks each for H100 and H125. [Hugging Face dataset repository](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks)
