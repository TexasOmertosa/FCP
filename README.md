# Imagine Beyond, Plan the Next

**Future-Context Planning with Latent World Models**

[![FCP weights and action priors](https://img.shields.io/badge/Checkpoints-Hugging%20Face-ffc84a?logo=huggingface)](https://huggingface.co/stardust8979/FCP-weights) [![Evaluation datasets](https://img.shields.io/badge/Data-Evaluation-278060)](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks) [![Reproduction guide](https://img.shields.io/badge/Guide-Reproduction-315caa)](USAGE.md) [![Original downloads](https://img.shields.io/badge/Assets-Original%20sources-5c5379)](ASSETS.md)

## Overview

FCP predicts future states in the world model's latent space and uses them to select the next subgoal. The frozen world model scores actions toward that subgoal, and FCP replans as new observations arrive.

## Closed-loop comparisons

<table>
  <tr>
    <td align="center" width="50%">
      <a href="media/pusht_full.mp4"><img src="media/previews/pusht_full.gif" alt="Full-task PushT comparison" width="410"></a><br>
      <strong>PushT · Full task</strong> · <a href="media/pusht_full.mp4">MP4</a>
    </td>
    <td align="center" width="50%">
      <a href="media/cube_H150.mp4"><img src="media/previews/cube_H150.gif" alt="Cube H150 comparison" width="410"></a><br>
      <strong>Cube · H150</strong> · <a href="media/cube_H150.mp4">MP4</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="media/scene2.mp4"><img src="media/previews/scene2.gif" alt="LIBERO Scene2 comparison" width="410"></a><br>
      <strong>LIBERO Scene2</strong> · <a href="media/scene2.mp4">MP4</a>
    </td>
    <td align="center" width="50%">
      <a href="media/caddy.mp4"><img src="media/previews/caddy.gif" alt="LIBERO Caddy comparison" width="410"></a><br>
      <strong>LIBERO Caddy</strong> · <a href="media/caddy.mp4">MP4</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="media/antmaze_D3.mp4"><img src="media/previews/antmaze_D3.gif" alt="AntMaze D3 comparison" width="410"></a><br>
      <strong>AntMaze · D3</strong> · <a href="media/antmaze_D3.mp4">MP4</a>
    </td>
    <td align="center" width="50%">
      <a href="media/fourroom_H125.mp4"><img src="media/previews/fourroom_H125.gif" alt="FourRoom H125 comparison" width="410"></a><br>
      <strong>FourRoom · H125</strong> · <a href="media/fourroom_H125.mp4">MP4</a>
    </td>
  </tr>
</table>

The goal is on the left. Methods in each clip start from the same initial state. In the full PushT clip, FCP succeeds after **110 actions**, and SAGE and H-JEPA reach the **300-action** limit. Each other clip continues for **20 video frames** after FCP reaches the goal.

## Repository layout

```text
FCP/
├── fcp.py                 # Future generator and subgoal solver
├── prepare.py             # Build windows from frozen demonstration latents
├── train.py               # Fit the two upper stages
├── generate.py            # Generate future contexts or subgoals
├── priors/                # Prepare, cache, and train the matching action prior
├── benchmarks/pusht.py    # Native PushT full-task evaluation
├── test.py                # Prediction error or evaluation with a native adapter
└── reproduce.py          # Check released weights and priors on reference windows
```

## Quick start

```bash
git clone https://github.com/TexasOmertosa/FCP.git
cd FCP
python -m pip install -r requirements.txt
python generate.py --environment pusht --variant true \
  --data examples/pusht_window.npz --output subgoals.npz --device cpu
```

The first run downloads the matching weights. To check the reference predictions and action priors for all 12 configurations, run `python reproduce.py --verify-priors`. Use `--environment pusht` to check only PushT.

## Training and evaluation

Train the future generator first. Reuse it to fit the True solver with demonstrated futures or the Generated solver with predicted futures. Then train an action prior for the selected upper weights. [Upper-model training](USAGE.md#train-generate-and-test) · [Action-prior training](priors/README.md)

For closed-loop evaluation, install the simulator stack for your benchmark and use its matching frozen world model. [Native PushT](USAGE.md#native-pusht-evaluation) · [Simulator versions](USAGE.md#simulator-versions) · [Original LeWM, SAGE, and H-JEPA assets](ASSETS.md)
