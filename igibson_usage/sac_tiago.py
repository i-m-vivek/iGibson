from os import stat
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gibson2.envs.mushrl_mp_igibson_env import iGibsonMPEnv

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._h1 = nn.Linear(64+97+3, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        print("AAH they used me!")
        import pdb; pdb.set_trace()
        aux_n_taskobs = state[:, :97]
        depth = state[:, 97:].view(-1, 1, 128, 128)
        cnn_out = torch.flatten(self.cnn(depth), 1)
        state_action = torch.cat((aux_n_taskobs.float(), cnn_out.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._h1 = nn.Linear(64+97, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

        # cnn init taken from HRL4IN
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, val=0)
            
    def forward(self, state):
        state = state.float()
        aux_n_taskobs = state[:, :97]
        depth = state[:, 97: ].view(-1, 1, 128, 128)
        cnn_out = torch.flatten(self.cnn(depth), 1)
        out = torch.cat((aux_n_taskobs.float(), cnn_out.float()), dim=1)
        features1 = F.relu(self._h1(out.float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

def experiment(alg, n_epochs, n_steps, n_steps_test):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = iGibsonMPEnv(
        config_file="configs/tiago_motion_planning.yaml", horizon=horizon, gamma=gamma
    )

    # Settings
    initial_replay_size = 256
    max_replay_size = 250
    batch_size = 8
    n_features = 8
    warmup_transitions = 100
    tau = 0.005 # not need to tweak much
    lr_alpha = 3e-4 # learning rate for alpha, don't need to change much

    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )

    actor_optimizer = {"class": optim.Adam, "params": {"lr": 1e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 1e-4}},
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
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        # dataset ha 
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n + 1, J=J, R=R, entropy=E)

if __name__ == "__main__":
    algs = [SAC]

    for alg in algs:
        experiment(alg=alg, n_epochs=40, n_steps=1000, n_steps_test=2000)
