from gibson2.envs.locomotor_env import (
    NavigateEnv,
    NavigateRandomEnv,
    InteractiveNavigateEnv,
)
import os 
import gibson2

config_file = os.path.join(
    os.path.dirname(gibson2.__file__), "../examples/configs", "jr_interactive_nav.yaml"
)
env = InteractiveNavigateEnv(
    config_file=config_file,
    mode="gui",
    action_timestep=1.0 / 10.0,
    physics_timestep=1.0 / 40.0,
    automatic_reset=True,
    device_idx=0,
    arena="complex_hl_ll"
)

# state_hist = []
# reward_hist = []
# done_hist = []
# info_hist = []
env.reset()
for i in range(1000):
    # with Profiler("Environment action step"):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(state)
    # print(reward)
    # state_hist.append(state)
    # reward_hist.append(reward)
    # done_hist.append(done)
    # info_hist.append(info)
    if done:
        logging.info("Episode finished after {} timesteps".format(i + 1))
        break
