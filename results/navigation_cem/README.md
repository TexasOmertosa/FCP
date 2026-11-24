# Navigation: matched SAGE prior + CEM

Both variants use eight complete independent FCG→FGSS draws and the mean immediate subgoal. CEM uses 300 candidates, 30 rounds and 30 elites. The SAGE Transformer-GMM8 priors are unchanged; there is no PRISM prior or product-of-Gaussians fusion.

| Model | AntMaze D3 | FourRoom H75 | FourRoom H100 | FourRoom H125 |
|---|---:|---:|---:|---:|
| FCP-True | 146/150 = 97.33 ± 1.15% | 49/50 = 98% | 50/50 = 100% | 26/50 = 52% |
| FCP-Generated | 134/150 = 89.33 ± 4.62% | 44/50 = 88% | 45/50 = 90% | 27/50 = 54% |

AntMaze evaluates seeds 42/43/44 with 50 fixed D3 tasks each; ± is the sample standard deviation across those evaluation groups, not training seeds. Its budget is 350, normal execution 10, last30 final-goal control, terminal 5+5. Its upper models are current-only A175/B181 (True) and A175/B275 (Generated).

FourRoom evaluates 50 fixed tasks per horizon, seed42, budgets210/280/350. It uses history[-10,-5,0], no mask, A200/B197 (True) or A200/B195 (Generated), normal10/last30/terminal10. Physical actions remain bounded to[-1,1] on every CEM candidate; costs preserve the native last-two-prediction reduction and intermediate weight. It does not use terminal5+5.

AntMaze counts come from the original SAGE-prior CEM run; only its AntMaze rows are selected. FourRoom counts come from the new CEM run with 300 native tasks and 300 strict physical replays, all verified. The original run also contains historical FourRoom GradientSolver rows; those are not the selected CEM results. Archive paths inside receipts identify original provenance, not paths required on another machine. Results hashes must match their COMPLETE receipts.

`PRIOR_REMOTE_VERIFIED.json` records anonymous remote LFS SHA256 checks for all four exported priors at the pinned Hugging Face revision. AntMaze selected prior epochs are11/8, FourRoom30/30; safetensors export hashes differ from source PyTorch container hashes, while all tensors were preserved by the recorded export. No prior retraining or new weight upload was needed for the CEM switch.

These are FCP results. Baseline methods and other environments retain their existing evaluation protocols; LIBERO's prior candidate ranking is not relabeled as CEM. The general Python backend interface requires an external native navigation controller; these receipts do not claim that a new all-in-one native navigation runner was added.
