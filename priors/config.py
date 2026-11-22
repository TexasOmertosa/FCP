"""Action-prior dimensions, normalization conventions, and optimizer settings."""

from release import CONFIGS

LIBERO = {'scene2', 'caddy'}
ACTION_DIM = dict(pusht=2, cube=5, scene2=7, caddy=7, antmaze=8, fourroom=2)
LOWDIM_DIM = dict(pusht=11, cube=28, scene2=0, caddy=0, antmaze=27, fourroom=2)
HISTORY_FRAMES = 3
EPOCHS = 30
ACTION_BLOCK = 5
ACTION_TOKENS = 2
DRAWS = 8


def architecture(environment):
    return dict(
        latent_dim=CONFIGS[environment]['architecture']['latent_dim'],
        lowdim_dim=LOWDIM_DIM[environment],
        action_dim=ACTION_DIM[environment] * ACTION_BLOCK,
        max_plan_horizon=ACTION_TOKENS,
        hidden_dim=512, num_heads=8, depth=3, num_modes=8,
        condition_on_offset=environment not in LIBERO,
    )
