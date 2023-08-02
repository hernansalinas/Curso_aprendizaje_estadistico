from . import envs
from gym.envs.registration import register

__version__ = "0.0.1"

register(
    id='mysnake-v0',
    entry_point='snake_gym.envs:SnakeEnv'
)