#!/usr/bin/env python3
"""Generate FourRoom H100 or H125 evaluation trajectories with H-JEPA."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


def load_generator(root):
    path = root.resolve() / "h_jepa/scripts/generate_fourroom_eval_tasks.py"
    sys.path.insert(0, str(root.resolve()))
    spec = importlib.util.spec_from_file_location("hjepa_fourroom_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def screen_long_rollout(env, start, goal, budget):
    """Check ego geometry before rendering the full distractor rollout."""
    position = torch.tensor(start, dtype=torch.float32)
    goal_tensor = torch.as_tensor(goal, dtype=torch.float32)
    speed = float(env.variation_space["agent"]["speed"].value.item())
    positions = [position.numpy().copy()]
    for _ in range(budget):
        current = position.numpy()
        direction = env.expert_waypoint(current, goal, 10.5) - current
        norm = float(np.linalg.norm(direction))
        action = direction / norm if norm > 1e-8 else np.zeros(2, np.float32)
        position = env._apply_collisions(position, position + torch.as_tensor(action) * speed)
        positions.append(position.numpy().copy())
        if float(torch.norm(position - goal_tensor)) < 16:
            return positions
    return None


def sample_tasks(official, world, options, horizon, seed, count, max_attempts, budget):
    rng = np.random.default_rng(seed)
    accepted = 0
    for attempt in range(1, max_attempts + 1):
        task_seed = int(rng.integers(0, 1_000_000_000))
        world.reset(seed=task_seed, options=options)
        env = official._unwrapped_env(world)
        try:
            start, goal, _, _ = official._sample_crossing_task(env, rng, 3)
        except official.NoRoomPairError:
            continue
        if horizon == 125:
            positions = screen_long_rollout(env, start, goal, budget)
            if positions is None or len(positions) <= horizon:
                continue
        steps = official._rollout_full_task(world, task_seed, options, start, goal, budget)
        if horizon == 125:
            if steps is None:
                raise RuntimeError("The expert rollout did not match the geometry screen.")
            np.testing.assert_allclose(
                [step["xy"] for step in steps], positions, rtol=0, atol=1e-5
            )
        if steps is None or len(steps) <= horizon:
            continue
        if horizon == 100 and len(steps) == horizon + 1:
            lo, hi = 0, horizon
        else:
            windows = official._valid_crossing_windows(env, steps, horizon, 3)
            if not windows:
                continue
            lo, hi = windows[int(rng.integers(len(windows)))]
        yield {
            "attempt": attempt,
            "seed": task_seed,
            "sampled_start_xy": start.tolist(),
            "sampled_goal_xy": goal.tolist(),
            "source_window": [lo, hi],
        }, steps
        accepted += 1
        if accepted == count:
            return
    raise RuntimeError(f"Accepted {accepted}/{count} tasks after {max_attempts} attempts.")


def replay_tasks(official, world, options, tasks, budget):
    for row in tasks:
        metadata = row["metadata"]
        steps = official._rollout_full_task(
            world, metadata["seed"], options, metadata["sampled_start_xy"],
            metadata["sampled_goal_xy"], budget,
        )
        if steps is None or len(steps) - 1 != metadata["source_action_count"]:
            raise RuntimeError(f"Expert rollout changed for task {row['task_index']}.")
        yield metadata, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hjepa-root", type=Path, required=True, help="Pinned H-JEPA checkout.")
    parser.add_argument("--horizon", type=int, choices=(100, 125), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 path.")
    parser.add_argument("--manifest", type=Path, help="Replay the released task JSON instead of sampling.")
    parser.add_argument("--seed", type=int, default=3072, help="Sampling seed for new tasks.")
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=20_000)
    args = parser.parse_args()
    if min(args.num_tasks, args.max_attempts) < 1:
        parser.error("Use positive task and attempt counts.")
    manifest_out = args.output.with_suffix(".json")
    if args.output.suffix != ".h5" or args.output.exists() or manifest_out.exists():
        parser.error("Provide a new .h5 path with no existing .json sidecar.")

    torch.set_num_threads(1)
    official = load_generator(args.hjepa_root)
    config = args.hjepa_root / "scripts/data/config/fourroom_tp35_d0to5.yaml"
    cfg = OmegaConf.load(config)
    cfg.world.min_distractors = cfg.world.max_distractors
    budget = int(cfg.world.max_episode_steps)
    options = official._base_options(cfg)
    seed, count = args.seed, args.num_tasks
    records = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        params = manifest["generation_parameters"]
        if (params["max_steps"], params["cross_n_rooms"], params["num_active_distractors"]) != (
            args.horizon, 3, 1
        ):
            parser.error("Manifest must describe the selected horizon, three doors, and one distractor.")
        records = manifest["tasks"]
        count, seed = len(records), params["seed"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_budget = {100: 280, 125: 350}[args.horizon]
    world = official._build_world(cfg)
    rows = []
    try:
        official._set_deterministic_expert(world, seed)
        tasks = (
            replay_tasks(official, world, options, records, budget) if records is not None
            else sample_tasks(official, world, options, args.horizon, seed, count, args.max_attempts, budget)
        )
        with TemporaryDirectory(dir=args.output.parent) as temp:
            temp = Path(temp)
            with h5py.File(temp / "tasks.h5", "w") as out:
                out.attrs.update(task_count=count, horizon=args.horizon,
                                 pixel_layout="NCHW", evaluation_action_budget=evaluation_budget)
                for index, (metadata, steps) in enumerate(tasks):
                    env = official._unwrapped_env(world)
                    lo, hi = metadata["source_window"]
                    if not (0 <= lo < hi < len(steps) and hi - lo == args.horizon):
                        raise ValueError(f"Invalid crop for task {index}.")
                    segment = steps[lo:hi + 1]
                    if any(not np.array_equal(s["goal_state"], segment[0]["goal_state"]) for s in segment):
                        raise ValueError(f"The expert goal changed during task {index}.")
                    if official._position_crossing_count(env, segment[0]["xy"], segment[-1]["xy"]) != 3:
                        raise ValueError(f"Task {index} must cross three doors.")
                    columns = official._saved_columns(steps[0].keys())
                    episode = official._make_variant_episode(env, segment, columns, index, 1)
                    if records is not None:
                        for key, endpoints in records[index]["endpoint_states"].items():
                            actual = episode[key].numpy()
                            expected = np.asarray([endpoints["initial"], endpoints["final"]], dtype=actual.dtype)
                            np.testing.assert_array_equal(actual[[0, -1]], expected)
                    group = out.create_group(f"{index:03d}")
                    for key, tensor in episode.items():
                        kwargs = {"compression": "gzip", "compression_opts": 1} if key == "pixels" else {}
                        group.create_dataset(key, data=tensor.numpy(), **kwargs)
                    rows.append({
                        "task_index": index,
                        "metadata": {**metadata, "source_action_count": len(steps) - 1},
                        "endpoint_states": {
                            key: {"initial": episode[key][0].tolist(), "final": episode[key][-1].tolist()}
                            for key in ("xy", "goal_state", "closed_door_idx")
                        },
                    })
                    print(f"Generated {index + 1}/{count} tasks", flush=True)
            result = {
                "generation_parameters": {
                    "seed": seed, "max_steps": args.horizon,
                    "cross_n_rooms": 3, "num_active_distractors": 1,
                },
                "task_count": count,
                "evaluation_action_budget": evaluation_budget,
                "tasks": rows,
            }
            (temp / "tasks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            (temp / "tasks.h5").rename(args.output)
            (temp / "tasks.json").rename(manifest_out)
    finally:
        world.close()


if __name__ == "__main__":
    main()
