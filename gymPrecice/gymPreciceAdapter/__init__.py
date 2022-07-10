import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FluidSolverAdapterEnv-v0',
    entry_point='gymPreciceAdapter.envs:FluidSolverAdapterEnv',

    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='FoamAdapterEnv-v0',
    entry_point='gymPreciceAdapter.envs:FoamAdapterEnv',

    reward_threshold=1.0,
    nondeterministic = True,
)
