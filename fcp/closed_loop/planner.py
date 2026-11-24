"""Execute the FCP closed loop with an environment's frozen control backend."""

from contextlib import nullcontext

import torch


def terminal_cost(backend, actions, target):
    if hasattr(backend, 'cost'):
        costs = backend.cost(actions, target)
        if costs.shape != (len(actions),):
            raise ValueError('The native cost must return one value per candidate.')
        return costs
    endpoints = backend.rollout(actions)
    if endpoints.shape != (len(actions), target.numel()):
        raise ValueError('The world model must return one terminal latent per candidate.')
    return (endpoints - target).square().mean(dim=-1)


def search(backend, target, goal, horizon, terminal, protocol, generator):
    with torch.no_grad():
        candidates = backend.propose(target, goal, horizon, protocol['candidates'],
                                     terminal, generator)
    if (candidates.ndim != 3 or candidates.shape[:2] != (protocol['candidates'], horizon)
            or candidates.device != target.device):
        raise ValueError('Expected the configured action bank [candidates, horizon, action_dim].')
    method = protocol['optimizer']

    with torch.no_grad():
        if method == 'rank':
            return candidates[terminal_cost(backend, candidates, target).argmin()]
        if method != 'cem':
            raise ValueError(f'Unknown planner {method}.')
        if protocol.get('action_bounds_required') and not hasattr(backend, 'bound_actions'):
            raise ValueError('This protocol requires backend.bound_actions for every CEM candidate.')
        if protocol.get('cost_backend_required') and not hasattr(backend, 'cost'):
            raise ValueError('This protocol requires the native backend.cost reduction.')
        for i in range(protocol['search_iterations']):
            if hasattr(backend, 'bound_actions'):
                candidates = backend.bound_actions(candidates)
            ordered = terminal_cost(backend, candidates, target).topk(protocol['elites'], largest=False)
            elites = candidates[ordered.indices]
            mean = elites.mean(dim=0)
            if i + 1 < protocol['search_iterations']:
                std = elites.std(dim=0).clamp_min(1e-6)
                eps = torch.randn(candidates.shape, device=target.device, generator=generator)
                candidates = mean + eps * std
                candidates[0] = mean
        return mean


@torch.no_grad()
def run_episode(model, backend, task, protocol, device, seed, samples, numerics=None):
    goal_array, budget, success = backend.reset(task)
    goal = torch.as_tensor(goal_array, dtype=torch.float32, device=device)
    if goal.ndim != 1 or budget < 1:
        raise ValueError('Reset must return a goal latent [D] and positive action budget.')
    noise = torch.Generator(device=device).manual_seed(seed)
    action_noise = torch.Generator(device=device).manual_seed(seed + 1_000_000_007)
    elapsed, done = 0, False
    while elapsed < budget and not (success or done):
        remaining = budget - elapsed
        terminal = remaining <= protocol['terminal_goal_steps']
        segment = protocol['final_segment'] if remaining <= 10 else protocol['segment']
        horizon = min(segment, remaining)
        if terminal:
            target = goal
        else:
            history = torch.as_tensor(backend.history(), dtype=torch.float32, device=device)
            precision = numerics.forward(device, model) if numerics else nullcontext()
            with precision:
                target = model.sample(history[None], goal[None], samples=samples, generator=noise)[0]
        actions = search(backend, target, goal, horizon, terminal, protocol, action_noise)
        executable = list(backend.actions_for_execution(actions))
        if len(executable) != horizon:
            raise ValueError('The executable action segment must match the search horizon.')
        for action in executable:
            success, done = backend.step(action)
            elapsed += 1
            if success or done or elapsed == budget:
                break
    return bool(success), elapsed
