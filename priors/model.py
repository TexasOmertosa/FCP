"""SAGE-style Transformer GMM over FCP action chunks."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _flatten_goal_latents(goal_latents: torch.Tensor) -> torch.Tensor:
    if goal_latents.ndim == 4:
        return goal_latents.flatten(1, 2)
    return goal_latents


def _time_features(values: torch.Tensor, max_value: int) -> torch.Tensor:
    values = values.float().view(-1, 1)
    x = values / max(float(max_value), 1.0)
    return torch.cat([x, torch.sin(math.pi * x), torch.cos(math.pi * x), torch.sin(2.0 * math.pi * x), torch.cos(2.0 * math.pi * x)], dim=-1)


class ActionPrior(nn.Module):
    """Decode a single shared mixture mode over all action tokens."""

    def __init__(
        self,
        latent_dim: int,
        lowdim_dim: int,
        action_dim: int = 10,
        max_plan_horizon: int = 7,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 3,
        num_modes: int = 8,
        dropout: float = 0.0,
        max_goal_offset: int = 200,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
        condition_on_offset: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.lowdim_dim = int(lowdim_dim)
        self.action_dim = int(action_dim)
        self.plan_horizon = int(max_plan_horizon)
        self.max_plan_horizon = int(max_plan_horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_modes = int(num_modes)
        self.depth = int(depth)
        self.max_goal_offset = int(max_goal_offset)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.visual_norm = nn.LayerNorm(self.latent_dim)
        self.visual_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.history_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.far_goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.length_encoder = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.offset_encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ) if condition_on_offset else None
        if self.lowdim_dim > 0:
            self.lowdim_encoder = nn.Sequential(
                nn.LayerNorm(self.lowdim_dim),
                nn.Linear(self.lowdim_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.lowdim_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        else:
            self.lowdim_encoder = None
            self.register_parameter("lowdim_type", None)

        self.length_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.offset_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) if condition_on_offset else None
        self.action_queries = nn.Parameter(
            torch.zeros(1, self.max_plan_horizon, self.hidden_dim)
        )
        self.action_pos = nn.Parameter(
            torch.zeros(1, self.max_plan_horizon, self.hidden_dim)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(depth))
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.logit_head = nn.Linear(self.hidden_dim, self.num_modes)
        self.mean_head = nn.Linear(self.hidden_dim, self.num_modes * self.action_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.num_modes * self.action_dim)

        for p in [
            self.history_type,
            self.goal_type,
            self.far_goal_type,
            self.length_type,
            self.offset_type,
            self.action_queries,
            self.action_pos,
        ]:
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)
        if self.lowdim_type is not None:
            nn.init.trunc_normal_(self.lowdim_type, std=0.02)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def _action_horizon_values(self, action_horizon, batch: int, device):
        if action_horizon is None:
            values = torch.full(
                (batch,), float(self.max_plan_horizon), device=device
            )
        elif torch.is_tensor(action_horizon):
            values = action_horizon.to(device=device, dtype=torch.float32).view(-1)
            if values.numel() == 1:
                values = values.expand(batch)
        else:
            values = torch.full((batch,), float(action_horizon), device=device)
        return values.clamp(1, self.max_plan_horizon)

    def _length_mask(self, hvals: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(self.max_plan_horizon, device=hvals.device)
        return steps[None, :] < hvals.long().clamp(1, self.max_plan_horizon)[:, None]

    def _offset_values(
        self,
        goal_offset_steps,
        subgoal_offset_steps,
        batch: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if goal_offset_steps is None:
            goal = torch.full(
                (batch,),
                float(self.max_goal_offset),
                device=device,
            )
        elif torch.is_tensor(goal_offset_steps):
            goal = goal_offset_steps.to(device=device, dtype=torch.float32).view(-1)
            if goal.numel() == 1:
                goal = goal.expand(batch)
        else:
            goal = torch.full((batch,), float(goal_offset_steps), device=device)
        if subgoal_offset_steps is None:
            subgoal = torch.clamp(goal, max=float(self.max_goal_offset))
        elif torch.is_tensor(subgoal_offset_steps):
            subgoal = subgoal_offset_steps.to(device=device, dtype=torch.float32).view(-1)
            if subgoal.numel() == 1:
                subgoal = subgoal.expand(batch)
        else:
            subgoal = torch.full((batch,), float(subgoal_offset_steps), device=device)
        return goal.clamp(1, self.max_goal_offset), subgoal.clamp(1, self.max_goal_offset)

    def _offset_features(
        self,
        goal_offset_steps,
        subgoal_offset_steps,
        batch: int,
        device,
    ) -> torch.Tensor:
        goal, subgoal = self._offset_values(
            goal_offset_steps,
            subgoal_offset_steps,
            batch,
            device,
        )
        denom = max(float(self.max_goal_offset), 1.0)
        ratio = subgoal / torch.clamp(goal, min=1.0)
        return self.offset_encoder(
            torch.stack([goal / denom, subgoal / denom, ratio], dim=-1)
        )

    def forward(
        self,
        history_latents,
        goal_latents,
        lowdim,
        action_horizon=None,
        far_goal_latents=None,
        goal_offset_steps=None,
        subgoal_offset_steps=None,
    ):
        goal_latents = _flatten_goal_latents(goal_latents)
        if far_goal_latents is None:
            far_goal_latents = goal_latents
        else:
            far_goal_latents = _flatten_goal_latents(far_goal_latents)
        batch = int(history_latents.size(0))
        history = (
            self.visual_proj(self.visual_norm(history_latents)) + self.history_type
        )
        goal = self.visual_proj(self.visual_norm(goal_latents)) + self.goal_type
        far_goal = (
            self.visual_proj(self.visual_norm(far_goal_latents))
            + self.far_goal_type
        )
        hvals = self._action_horizon_values(
            action_horizon, batch, history_latents.device
        )
        length_token = self.length_encoder(
            _time_features(hvals, self.max_plan_horizon)
        ).unsqueeze(1) + self.length_type
        memory_parts = [history, goal, far_goal, length_token]
        if self.offset_encoder is not None:
            offset = self._offset_features(goal_offset_steps, subgoal_offset_steps,
                                           batch, history_latents.device)
            memory_parts.append(offset.unsqueeze(1) + self.offset_type)
        if self.lowdim_encoder is not None:
            memory_parts.append(self.lowdim_encoder(lowdim).unsqueeze(1) + self.lowdim_type)
        memory = self.memory_norm(torch.cat(memory_parts, dim=1))
        query = (self.action_queries + self.action_pos).expand(batch, -1, -1)
        decoded = self.output_norm(self.decoder(query, memory))
        mask = self._length_mask(hvals).to(decoded.dtype)
        pooled = (decoded * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.logit_head(pooled)
        means = self.mean_head(decoded).view(
            batch, self.max_plan_horizon, self.num_modes, self.action_dim
        ).permute(0, 2, 1, 3).contiguous()
        log_stds = self.log_std_head(decoded).view_as(
            means.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3).contiguous()
        log_stds = log_stds.clamp(self.min_log_std, self.max_log_std)
        return {"logits": logits, "means": means, "log_stds": log_stds}

    def _slice_outputs(self, outputs, action_horizon=None):
        horizon = self.max_plan_horizon if action_horizon is None else int(action_horizon)
        horizon = max(1, min(horizon, self.max_plan_horizon))
        return {
            "logits": outputs["logits"],
            "means": outputs["means"][:, :, :horizon],
            "log_stds": outputs["log_stds"][:, :, :horizon],
        }

    def loss_terms(self, outputs, target):
        outputs = self._slice_outputs(outputs, target.size(1))
        means, log_stds = outputs['means'], outputs['log_stds']
        error = target[:, None] - means
        log_prob = -0.5 * (error * torch.exp(-log_stds)).square()
        log_prob = (log_prob - log_stds - 0.5 * math.log(2 * math.pi)).sum(dim=(-1, -2))
        nll = -torch.logsumexp(F.log_softmax(outputs['logits'], dim=-1) + log_prob, dim=-1).mean()
        best_l1 = error.abs().mean(dim=(-1, -2)).min(dim=1).values.mean()
        return nll, best_l1

    @torch.no_grad()
    def top_mode(self, history, goal, lowdim, **kwargs):
        result = self._slice_outputs(self(history, goal, lowdim, **kwargs), kwargs.get('action_horizon'))
        indices = result['logits'].argmax(dim=-1)
        return result['means'][torch.arange(len(indices), device=indices.device), indices]

    @torch.no_grad()
    def sample(
        self,
        history_latents,
        goal_latents,
        lowdim,
        num_samples: int,
        generator=None,
        action_horizon=None,
        far_goal_latents=None,
        goal_offset_steps=None,
        subgoal_offset_steps=None,
    ):
        outputs = self._slice_outputs(
            self(
                history_latents,
                goal_latents,
                lowdim,
                action_horizon,
                far_goal_latents,
                goal_offset_steps,
                subgoal_offset_steps,
            ),
            action_horizon,
        )
        probs = F.softmax(outputs["logits"], dim=-1)
        mode_ids = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
        gather = mode_ids[:, :, None, None].expand(
            -1, -1, outputs["means"].size(2), self.action_dim
        )
        means = outputs["means"].gather(1, gather)
        log_stds = outputs["log_stds"].gather(1, gather)
        noise = torch.randn(means.shape, dtype=means.dtype, device=means.device, generator=generator)
        return means + noise * torch.exp(log_stds)
