import gibson2
from gibson2.envs.parallel_env import ParallelNavEnvironment
from gibson2.envs.locomotor_env import (
    NavigateEnv,
    NavigateRandomEnv,
    InteractiveNavigateEnv,
)
import os 
from hrl4in.utils.utils import *
from hrl4in.utils.args import *
import pickle 
config_file = "/home/vkmittal14/WORKSPACE/TUD/HRL_igibon/iGibson/examples/configs/jr_interactive_nav.yaml"

env = InteractiveNavigateEnv(
    config_file=config_file,
    mode="gui",
    action_timestep=1/10.0,
    physics_timestep=1/40.0,
    automatic_reset=True,
    device_idx=0,
    arena="complex_hl_ll"
)
state_hist = []
env.reset()
for i in range(100):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    state_hist.append(state)


with open(f"jr_interactive_nav_states.pkl", "wb") as f:
    pickle.dump(state_hist, f)
