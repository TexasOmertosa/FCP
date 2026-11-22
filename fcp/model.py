"""Two-stage Future-Context Planning on frozen observation latents."""

import math
from contextlib import nullcontext

import torch
from torch import nn


class FlowBlock(nn.Module):
    def __init__(self, width, heads, ffn_width, inject_future):
        super().__init__()
        self.norms = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False) for _ in range(3)
        )
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(),
                                 nn.Linear(ffn_width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.future_projection = nn.Linear(width, width, bias=False) if inject_future else None
        self.attention_backends = (None, None)

    def forward(self, nodes, memory, time_embedding, future_embedding, near_tokens):
        shifts_scales_gates = self.modulation(time_embedding).chunk(9, dim=-1)
        for index, norm in enumerate(self.norms):
            shift, scale, gate = shifts_scales_gates[3 * index:3 * index + 3]
            hidden = norm(nodes) * (1 + scale[:, None]) + shift[:, None]
            if index < 2:
                backend = self.attention_backends[index]
                context = nullcontext() if backend is None else torch.nn.attention.sdpa_kernel(backend)
                if index == 0:
                    attention, key = self.self_attention, hidden
                else:
                    attention, key = self.cross_attention, memory
                with context:
                    update, _ = attention(hidden, key, key, need_weights=False)
            else:
                if self.future_projection is not None:
                    delta = 0.1 * self.future_projection(future_embedding)
                    hidden = torch.cat((hidden[:, :near_tokens] + delta[:, None],
                                        hidden[:, near_tokens:]), dim=1)
                update = self.ffn(hidden)
            nodes = nodes + gate[:, None] * update
        return nodes


class VelocityField(nn.Module):
    """A generates future nodes. B generates one subgoal from clean future nodes."""

    def __init__(self, latent_dim, history_frames, views, future_nodes,
                 width, depth, heads, ffn_width, *, solver):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_frames = history_frames
        self.views = views
        self.future_nodes = future_nodes
        self.solver = solver
        self.width = width
        view_dim = latent_dim // views
        query_nodes = future_nodes + int(solver)

        self.visual_projection = nn.Sequential(nn.Linear(view_dim, width), nn.LayerNorm(width))
        self.node_projection = nn.Linear(view_dim, width)
        self.node_position = nn.Parameter(torch.randn(1, query_nodes, views, width) * 0.02)
        self.history_position = (
            nn.Parameter(torch.zeros(1, history_frames, views, width))
            if history_frames > 1 else None
        )
        self.view_position = (
            nn.Parameter(torch.randn(1, 1, views, width) * 0.02) if views > 1 else None
        )
        self.history_type = nn.Parameter(torch.zeros(1, 1, width))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, width))
        self.roles = nn.Embedding(3, width)
        roles = [1] + [2] * future_nodes if solver else [1] * future_nodes
        self.register_buffer('node_roles', torch.tensor(roles).repeat_interleave(views),
                             persistent=False)

        frequencies = torch.exp(-math.log(10000) * torch.arange(64) / 63)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.flow_time = nn.Sequential(nn.Linear(128, width), nn.SiLU(),
                                       nn.Linear(width, width), nn.LayerNorm(width))
        self.blocks = nn.ModuleList(
            FlowBlock(width, heads, ffn_width, solver and i >= depth - 2) for i in range(depth)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, view_dim))
        if solver:
            self.future_encoder = nn.Sequential(nn.Linear(future_nodes * latent_dim, width),
                                                nn.SiLU(), nn.Linear(width, width),
                                                nn.LayerNorm(width))

    def condition(self, history, goal, future):
        batch = goal.shape[0]
        if history.ndim == 2:
            history = history[:, None]
        if history.shape != (batch, self.history_frames, self.latent_dim):
            raise ValueError('Expected observation history [B, T, D].')
        history = self.visual_projection(history.reshape(batch, self.history_frames, self.views, -1))
        goal = self.visual_projection(goal.reshape(batch, 1, self.views, -1))
        view_position = 0 if self.view_position is None else self.view_position
        history_position = 0 if self.history_position is None else self.history_position
        history = (history + history_position + view_position).flatten(1, 2)
        goal = (goal + view_position).flatten(1, 2)
        memory = torch.cat((history + self.history_type, goal + self.goal_type), dim=1)
        future_embedding = None
        if self.solver:
            if future is None or future.shape != (batch, self.future_nodes, self.latent_dim):
                raise ValueError('FGSS requires all clean future nodes [B, K, D].')
            future_embedding = self.future_encoder(future.flatten(1))
        return memory, future_embedding

    def forward(self, value, time, memory, future=None, future_embedding=None):
        batch, output_nodes, _ = value.shape
        path = value if future is None else torch.cat((value, future), dim=1)
        view_position = 0 if self.view_position is None else self.view_position
        tokens = self.node_projection(path.reshape(batch, path.shape[1], self.views, -1))
        tokens = (tokens + self.node_position + view_position).flatten(1, 2)
        tokens = tokens + self.roles(self.node_roles)[None]
        phase = time.float()[:, None] * self.frequencies * 1000
        time_embedding = self.flow_time(torch.cat((phase.cos(), phase.sin()), dim=-1))
        for block in self.blocks:
            tokens = block(tokens, memory, time_embedding, future_embedding, self.views)
        output_tokens = output_nodes * self.views
        if self.blocks[0].attention_backends[0] is not None:
            # Native CUDA kernels project every query before reading the target.
            return self.output(tokens)[:, :output_tokens].reshape(batch, output_nodes, -1)
        return self.output(tokens[:, :output_tokens]).reshape(batch, output_nodes, -1)


def _velocity_loss(field, history, goal, label, future):
    label = label.float()
    noise = torch.randn_like(label)
    time = torch.rand(label.shape[0], device=label.device)
    interpolated = (1 - time[:, None, None]) * noise + time[:, None, None] * label
    memory, context = field.condition(history, goal, future)
    velocity = field(interpolated, time, memory, future, context)
    return (velocity.float() - (label - noise)).square().mean()


def _integrate(field, history, goal, noise, future, steps):
    value = noise.float()
    memory, context = field.condition(history, goal, future)
    for i in range(steps):
        time = value.new_full((value.shape[0],), i / steps)
        value = value + field(value, time, memory, future, context).float() / steps
    return value


class FCP(nn.Module):
    def __init__(self, latent_dim, *, history_frames=1, views=1, future_nodes=3,
                 width=512, depth=6, heads=8, ffn_width=2048, euler_steps=8):
        super().__init__()
        if views < 1 or latent_dim < 1 or latent_dim % views:
            raise ValueError('Each view must have the same positive latent dimension.')
        if min(history_frames, future_nodes, euler_steps) < 1 or depth < 2:
            raise ValueError('Require positive input and integration sizes and at least two blocks.')
        args = (latent_dim, history_frames, views, future_nodes, width, depth, heads, ffn_width)
        self.A = VelocityField(*args, solver=False)
        self.B = VelocityField(*args, solver=True)
        self.future_nodes = future_nodes
        self.euler_steps = euler_steps

    def fcg_loss(self, history, goal, demonstrated_future):
        """Fit A to the K demonstrated later states."""
        return _velocity_loss(self.A, history, goal, demonstrated_future, None)

    def fgss_loss(self, history, goal, demonstrated_subgoal, future_condition):
        """Fit B to the demonstrated subgoal under a clean future condition."""
        return _velocity_loss(self.B, history, goal, demonstrated_subgoal[:, None],
                              future_condition.detach())

    @torch.no_grad()
    def generate_future(self, history, goal, *, generator=None):
        """Generate conditions with frozen A for FCP-Generated training."""
        noise = torch.randn((goal.shape[0], self.future_nodes, goal.shape[-1]),
                            device=goal.device, generator=generator)
        return _integrate(self.A, history, goal, noise, None, self.euler_steps)

    @torch.no_grad()
    def sample_from_noise(self, history, goal, noise):
        """Run one complete draw from externally seeded subgoal and future noise."""
        future = _integrate(self.A, history, goal, noise[:, 1:], None, self.euler_steps)
        return _integrate(self.B, history, goal, noise[:, :1], future, self.euler_steps)[:, 0]

    @torch.no_grad()
    def sample(self, history, goal, *, samples=1, generator=None):
        """Average complete A-to-B draws. The histories and goals are frozen latents."""
        if samples < 1:
            raise ValueError('The number of complete samples must be positive.')
        subgoals = []
        for _ in range(samples):
            future = self.generate_future(history, goal, generator=generator)
            noise = torch.randn((goal.shape[0], 1, goal.shape[-1]),
                                device=goal.device, generator=generator)
            subgoal = _integrate(self.B, history, goal, noise, future, self.euler_steps)
            subgoals.append(subgoal[:, 0])
        return torch.stack(subgoals).mean(dim=0)


def load_checkpoint(path, device, stage):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['stage'] != stage:
        raise ValueError(f'Expected a {stage.upper()} checkpoint.')
    model = FCP(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device), checkpoint


def load_fcp(environment_dir, condition, device='cpu'):
    """Load a published FCG/FGSS pair from the environment's directory."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    root = Path(environment_dir)
    settings = json.loads((root / 'config.json').read_text())
    model = FCP(**settings['architecture'])
    components = settings['variants'][condition]
    model.A.load_state_dict(load_file(str(root / components['fcg'])), strict=True)
    model.B.load_state_dict(load_file(str(root / components['fgss'])), strict=True)
    return model.to(device).eval()
