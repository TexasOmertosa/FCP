# Checkpoints and datasets

Use the frozen checkpoint that produced the demonstration latents for each environment. The weight links below are pinned to a revision. The SHA-256 values identify the exact files used by the FCP experiments.

## FCP release

| Asset | Source |
|---|---|
| FCG, FGSS-True, FGSS-Generated, and matched action priors for all six environments | [FCP weights](https://huggingface.co/stardust8979/FCP-weights/tree/2727a680b2d45ce7cca1768ff272f4614245afd4) · [Checksums](https://huggingface.co/stardust8979/FCP-weights/blob/2727a680b2d45ce7cca1768ff272f4614245afd4/checksums.json) |
| PushT full440, AntMaze D3, FourRoom H100 and H125 | [FCP evaluation tasks](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/f5cd0e6c8fc662933f549033d850dc31fcd5935d) |

## Frozen lower models

Download both `weights.pt` and the adjacent `config.json` for the four published lower models.

| Environment | Checkpoint source | SHA-256 of weights |
|---|---|---|
| PushT | [Official LeWM PushT](https://huggingface.co/quentinll/lewm-pusht/tree/22b330c28c27ead4bfd1888615af1340e3fe9052) | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| Cube | [Official LeWM Cube](https://huggingface.co/quentinll/lewm-cube/tree/b0747c5002e86d2ce8f3cd8178004b97524c587d) | `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89` |
| Scene2 | [SAGE Scene2 world model](https://huggingface.co/CLTRAY/SAGE/tree/d81d637d72884d8ef0ad80813d37c583ebfd33c8/libero_scene2/world_model) | `a62b226f1875dfdc31f93ba12ed50e3d4a53ecac47cce34c1db8c68e42fee0ff` |
| Caddy | [SAGE Caddy world model](https://huggingface.co/CLTRAY/SAGE/tree/d81d637d72884d8ef0ad80813d37c583ebfd33c8/libero_caddy/world_model) | `c5b6b3cd3384b5e4914942ffb762ad832d38d89436be11ef5446c8abb80c4378` |

The AntMaze and FourRoom lower models were retrained with the H-JEPA code. The exact files are in the project archive and require access to that archive. The public H-JEPA repository provides the training code, data, and upstream checkpoints. [H-JEPA source](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5) · [H-JEPA public checkpoints](https://huggingface.co/jepa-world-models/h-jepa/tree/08fc8a21208671a87741f2960aeba9b5db4420e0)

| Environment | Project archive path | SHA-256 |
|---|---|---|
| AntMaze | `/mnt/planner/antmaze_best78_native_A100_reproduction_20261030/inputs/lower_E20.pt` | `525b8e595a6791433de35b4b49f95054c3ed4d1dadea9c0b6d4a51af9ea3801a` |
| FourRoom | `/mnt/cfs/sfxxdd/planner/data/hjepa_fourroom_official/ckpts/fourroom/fourroom_hjepa3_isolated_l1_cross3_h100_20261101/seed44/fourroom_hjepa3_isolated_l1_object.ckpt` | `e8144cd19633daa94d3f0138e13e27d3ff3958be6068bec2c95cde43a4a691bc` |

## Original demonstrations and benchmark tasks

| Environment | Demonstrations | Evaluation inputs |
|---|---|---|
| PushT | [`pusht_expert_train.h5.zst`](https://huggingface.co/datasets/quentinll/lewm-pusht/blob/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst) | [Full440 task data and index](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/f5cd0e6c8fc662933f549033d850dc31fcd5935d/pusht) |
| Cube | [`cube_single_expert.tar.zst`](https://huggingface.co/datasets/quentinll/lewm-cube/blob/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst) | Extract the segmented windows from the same demonstrations using the recorded seeds and offsets. |
| Scene2 | `libero_scene2_train.tar.part*` and `libero_scene2_validation.tar`. [SAGE training archives](https://huggingface.co/datasets/CLTRAY/SAGE-data/tree/7e87e5b0fee414a70945a310d08e18d2f92cffba/training) | [Scene2 evaluation archive](https://huggingface.co/datasets/CLTRAY/SAGE-data/blob/7e87e5b0fee414a70945a310d08e18d2f92cffba/evaluation/libero_scene2_evaluation.tar) |
| Caddy | `libero_caddy_train.tar.part*` and `libero_caddy_validation.tar`. [SAGE training archives](https://huggingface.co/datasets/CLTRAY/SAGE-data/tree/7e87e5b0fee414a70945a310d08e18d2f92cffba/training) | [Caddy evaluation archive](https://huggingface.co/datasets/CLTRAY/SAGE-data/blob/7e87e5b0fee414a70945a310d08e18d2f92cffba/evaluation/libero_caddy_evaluation.tar) |
| AntMaze | [Train HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/visual_antmaze_medium_explore_stitch_train.h5) · [Val HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/visual_antmaze_medium_stitch_val.h5) | [Original H-JEPA D3 tasks](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/eval_trajs/ant/expert_grid_d3_n50.pt) · [FCP D3 task states](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/f5cd0e6c8fc662933f549033d850dc31fcd5935d/antmaze) |
| FourRoom | [Train HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/fourroom_tp35_d1.h5) · [Val HDF5](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/fourroom_tp35_d1_val.h5) | [Original H-JEPA H75 tasks](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt) · [FCP H100 and H125 task data](https://huggingface.co/datasets/stardust8979/FCP-evaluation-tasks/tree/f5cd0e6c8fc662933f549033d850dc31fcd5935d/fourroom) |

Use the [SAGE extraction instructions](https://huggingface.co/datasets/CLTRAY/SAGE-data#extract) for the multipart LIBERO training archives.
