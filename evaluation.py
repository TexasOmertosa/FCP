"""Shared held-out latent evaluation for checkpoint selection and the CLI."""

import torch


@torch.no_grad()
def evaluate(model, loader, device, counts, seed, future_only=False):
    model.eval()
    noise = torch.Generator(device=device).manual_seed(seed)
    totals = {count: 0.0 for count in counts}
    coordinates = 0
    for history, goal, subgoal, future in loader:
        history, goal = history.to(device), goal.to(device)
        target = (future if future_only else subgoal).to(device)
        predict = model.generate_future if future_only else model.sample
        draws = torch.stack([predict(history, goal, generator=noise)
                             for _ in range(max(counts))])
        for count in counts:
            prediction = draws[:count].mean(dim=0)
            totals[count] += (prediction.double() - target).square().sum().item()
        coordinates += target.numel()
    return {count: total / coordinates for count, total in totals.items()}
