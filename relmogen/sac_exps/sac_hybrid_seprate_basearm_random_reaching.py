from re import M
from statistics import mode
import numpy as np
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sac_hybrid_separate import SAC_hybrid
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from gibson2.envs.mushrl_mp_igibson_env_basearm import iGibsonMPEnv

from tqdm import trange
import wandb

start_time = str(time.strftime("%Y%m%d-%H%M%S"))
wandb.init(project="hrl-mm", entity="vassist", id = start_time)

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(6 + 7, n_features)
        self._h2 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        q = self._h2(features1)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(6, n_features)
        self._h2 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, state):
        features1 = F.relu(self._h1(state.float()))
        a = self._h2(features1)
        return a


def experiment(alg, n_epochs, n_steps, n_steps_test):
    np.random.seed()
    dirname = alg.__name__ + "_basearm_random_reaching_run_1"
    logger = Logger(dirname, results_dir="relmogen_exps/", log_console=True)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)

    # MDP
    horizon = 50
    gamma = 0.99
    mdp = iGibsonMPEnv(
        config_file="../../igibson_usage/new_configs/tiago_basearm_random_reaching_relmogen.yaml",
        horizon=horizon,
        gamma=gamma,
        # mode="gui"
    )

    # Settings
    initial_replay_size = 512
    max_replay_size = 75000
    batch_size = 64
    n_features = 128
    warmup_transitions = 256
    tau = 0.005
    lr_alpha = 3e-4
    actor_lr = 3e-4
    critic_lr = 3e-4

    wandb.config = {
        "initial_replay_size": initial_replay_size,
        "max_replay_size": max_replay_size,
        "batch_size": batch_size,
        "n_features": n_features,
        "warmup_transitions": warmup_transitions,
        "tau": tau,
        "lr_aplha": lr_alpha,
        "horizon": horizon,
        "gamma": gamma,
        "n_steps": n_steps,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "n_epochs": n_epochs,
        "n_steps_test": n_steps_test,
        "dirname": dirname,
    }

    use_cuda = torch.cuda.is_available()
    logger.info(f"horizon: {horizon}")
    logger.info(f"gamma: {gamma}")
    logger.info(f"initial_replay_size: {initial_replay_size}")
    logger.info(f"max_replay_size: {max_replay_size}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"n_features: {n_features}")
    logger.info(f"warmup_transitions: {warmup_transitions}")
    logger.info(f"tau: {tau}")
    logger.info(f"lr_alpha: {lr_alpha}")
    logger.info(f"n_steps: {n_steps}")

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=6,
        use_cuda=use_cuda,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=6,
        use_cuda=use_cuda,
    )
    actor_discrete_params = dict(network=ActorNetwork,
                                n_features=n_features,
                                input_shape=actor_input_shape,
                                output_shape=1,
                                use_cuda=use_cuda)

    actor_optimizer = {"class": optim.Adam, "params": {"lr": actor_lr}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
        use_cuda=use_cuda,
    )

    # Agent
    agent = alg(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_discrete_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
    )

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    s = s[:batch_size, :]
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        s = s[:batch_size, :]
        E = agent.policy.entropy(s)

        logger.epoch_info(n + 1, J=J, R=R, entropy=E)
        wandb.log({"J": J, "R": R, "E": E})
        logger.log_best_agent(agent, J)

    logger.log_agent(agent)
    logger.log_dataset(dataset)
    logger.info("Experiment Terminated")


if __name__ == "__main__":
    algs = [SAC_hybrid]

    for alg in algs:
        experiment(alg=alg, n_epochs=600, n_steps=1024, n_steps_test=512)
