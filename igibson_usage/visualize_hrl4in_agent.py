from time import time
from collections import deque
import random
import numpy as np
import argparse
import gym
import os
import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage, MetaPolicy, AsyncRolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *

import gibson2

# from gibson2.envs.parallel_env import ParallelNavEnvironment
# from gibson2.envs.locomotor_env import (
#     NavigateEnv,
#     NavigateRandomEnv,
#     InteractiveNavigateEnv,
# )
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.envs.parallel_env import ParallelNavEnv

from IPython import embed
import matplotlib.pyplot as plt

def evaluate(
    args,
    envs,
    meta_actor_critic,
    actor_critic,
    action_mask_choices,
    subgoal_mask_choices,
    subgoal_tolerance,
    device,
    update=0,
    count_steps=0,
    eval_only=False,
):
    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs._num_envs, 1, device=device)
    episode_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    episode_lengths = torch.zeros(envs._num_envs, 1, device=device)
    episode_collision_steps = torch.zeros(envs._num_envs, 1, device=device)
    # episode_total_energy_costs = torch.zeros(envs._num_envs, 1, device=device)
    # episode_avg_energy_costs = torch.zeros(envs._num_envs, 1, device=device)
    # episode_stage_open_door = torch.zeros(envs._num_envs, 1, device=device)
    # episode_stage_to_target = torch.zeros(envs._num_envs, 1, device=device)

    episode_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs._num_envs, 1, device=device)

    subgoal_rewards = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_lengths = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_subgoal_reward = torch.zeros(envs._num_envs, 1, device=device)

    current_meta_recurrent_hidden_states = torch.zeros(
        envs._num_envs, args.hidden_size, device=device
    )
    next_meta_recurrent_hidden_states = torch.zeros(
        envs._num_envs, args.hidden_size, device=device
    )
    recurrent_hidden_states = torch.zeros(
        envs._num_envs, args.hidden_size, device=device
    )
    subgoals_done = torch.zeros(envs._num_envs, 1, device=device)
    masks = torch.zeros(envs._num_envs, 1, device=device)

    current_subgoals = torch.zeros(batch["sensor"].shape, device=device)
    current_subgoals_steps = torch.zeros(envs._num_envs, 1, device=device)
    current_subgoal_masks = torch.zeros(batch["sensor"].shape, device=device)

    action_dim = envs.action_space.shape[0]
    current_action_masks = torch.zeros(envs._num_envs, action_dim, device=device)

    step = 0
    while episode_counts.sum() < args.num_eval_episodes:
        with torch.no_grad():
            (
                _,
                subgoals,
                _,
                action_mask_indices,
                _,
                meta_recurrent_hidden_states,
            ) = meta_actor_critic.act(
                batch,
                current_meta_recurrent_hidden_states,
                masks,
                deterministic=False,
            )

            if meta_actor_critic.use_action_masks:
                action_masks = action_mask_choices.index_select(
                    0, action_mask_indices.squeeze(1)
                )
                subgoal_masks = subgoal_mask_choices.index_select(
                    0, action_mask_indices.squeeze(1)
                )
            else:
                action_masks = torch.ones_like(current_action_masks)
                subgoal_masks = torch.ones_like(current_subgoal_masks)

            should_use_new_subgoals = (current_subgoals_steps == 0.0).float()
            current_subgoals = (
                should_use_new_subgoals * subgoals
                + (1 - should_use_new_subgoals) * current_subgoals
            )
            current_subgoal_masks = (
                should_use_new_subgoals * subgoal_masks.float()
                + (1 - should_use_new_subgoals) * current_subgoal_masks
            )
            current_subgoals *= current_subgoal_masks
            current_action_masks = (
                should_use_new_subgoals * action_masks
                + (1 - should_use_new_subgoals) * current_action_masks
            )
            next_meta_recurrent_hidden_states = (
                should_use_new_subgoals * meta_recurrent_hidden_states
                + (1 - should_use_new_subgoals) * next_meta_recurrent_hidden_states
            )
            ideal_next_state = batch["sensor"] + current_subgoals

            # if eval_only: NOT IMPLEMENTED FOR NEW iGIBSON ENVS
            #     envs.set_subgoal(ideal_next_state.cpu().numpy())
            #     base_only = (current_subgoal_masks[:, 2] == 0).cpu().numpy()
            #     envs.set_subgoal_type(base_only)

            roll = batch["auxiliary_sensor"][:, 3] * np.pi
            pitch = batch["auxiliary_sensor"][:, 4] * np.pi
            yaw = batch["auxiliary_sensor"][:, 84] * np.pi
            current_subgoals_rotated = rotate_torch_vector_base_arm(
                current_subgoals, roll, pitch, yaw
            )
            current_subgoals_observation = current_subgoals_rotated

            batch["subgoal"] = current_subgoals_observation
            batch["subgoal_mask"] = current_subgoal_masks
            batch["action_mask"] = current_action_masks

            (
                values,
                actions,
                action_log_probs,
                recurrent_hidden_states,
            ) = actor_critic.act(
                batch,
                recurrent_hidden_states,
                1 - subgoals_done,
                deterministic=False,
                update=0,
            )
            actions_masked = actions * current_action_masks

        actions_np = actions_masked.cpu().numpy()
        outputs = envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        next_obs = [
            info["last_observation"] if done else obs
            for obs, done, info in zip(observations, dones, infos)
        ]

        prev_batch = batch

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        next_obs_batch = batch_obs(next_obs)
        for sensor in next_obs_batch:
            next_obs_batch[sensor] = next_obs_batch[sensor].to(device)

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )
        success_masks = torch.tensor(
            [
                [float(info["success"])] if done and "success" in info else [0.0]
                for done, info in zip(dones, infos)
            ],
            dtype=torch.float,
            device=device,
        )
        lengths = torch.tensor(
            [
                [float(info["episode_length"])]
                if done and "episode_length" in info
                else [0.0]
                for done, info in zip(dones, infos)
            ],
            dtype=torch.float,
            device=device,
        )
        collision_steps = torch.tensor(
            [
                [float(info["collision_step"])]
                if done and "collision_step" in info
                else [0.0]
                for done, info in zip(dones, infos)
            ],
            dtype=torch.float,
            device=device,
        )
        # total_energy_cost = torch.tensor(
        #     [[float(info["energy_cost"])] if done and "energy_cost" in info else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        # avg_energy_cost = torch.tensor(
        #     [[float(info["energy_cost"]) / float(info["episode_length"])]
        #      if done and "energy_cost" in info and "episode_length" in info
        #      else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        # stage_open_door = torch.tensor(
        #     [[float(info["stage"] >= 1)] if done and "stage" in info else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        # stage_to_target = torch.tensor(
        #     [[float(info["stage"] >= 2)] if done and "stage" in info else [0.0]
        #      for done, info in zip(dones, infos)],
        #     dtype=torch.float,
        #     device=device
        # )
        collision_rewards = torch.tensor(
            [
                [float(info["collision_reward"])]
                if "collision_reward" in info
                else [0.0]
                for info in infos
            ],
            dtype=torch.float,
            device=device,
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_success_rates += success_masks
        episode_lengths += lengths
        episode_collision_steps += collision_steps
        # episode_total_energy_costs += total_energy_cost
        # episode_avg_energy_costs += avg_energy_cost
        # episode_stage_open_door += stage_open_door
        # episode_stage_to_target += stage_to_target
        episode_counts += 1 - masks
        current_episode_reward *= masks

        current_subgoals_steps += 1

        subgoals_diff = (
            ideal_next_state - next_obs_batch["sensor"]
        ) * current_subgoal_masks
        subgoals_distance = torch.abs(subgoals_diff)

        subgoals_achieved = torch.all(
            subgoals_distance < subgoal_tolerance, dim=1, keepdim=True
        )

        subgoals_done = (
            subgoals_achieved  # subgoals achieved
            | (current_subgoals_steps == args.time_scale)  # subgoals time up
            | (1.0 - masks).bool()  # episode is done
        )
        subgoals_done = subgoals_done.float()
        subgoals_achieved = subgoals_achieved.float()

        prev_potential = ideal_next_state - prev_batch["sensor"]
        prev_potential = torch.norm(
            prev_potential * current_subgoal_masks, dim=1, keepdim=True
        )

        current_potential = ideal_next_state - next_obs_batch["sensor"]
        current_potential = torch.norm(
            current_potential * current_subgoal_masks, dim=1, keepdim=True
        )

        intrinsic_reward = 0.0
        intrinsic_reward += (
            prev_potential - current_potential
        ) * args.intrinsic_reward_scaling
        intrinsic_reward += subgoals_achieved.float() * args.subgoal_achieved_reward
        intrinsic_reward += collision_rewards * args.extrinsic_collision_reward_weight
        intrinsic_reward += rewards * args.extrinsic_reward_weight

        current_subgoal_reward += intrinsic_reward
        subgoal_rewards += subgoals_done * current_subgoal_reward
        subgoal_success_rates += subgoals_achieved
        subgoal_lengths += subgoals_done * current_subgoals_steps
        subgoal_counts += subgoals_done
        current_subgoal_reward *= 1 - subgoals_done

        current_subgoals = (
            ideal_next_state - next_obs_batch["sensor"]
        ) * current_subgoal_masks
        current_subgoals_steps = (1 - subgoals_done) * current_subgoals_steps
        current_meta_recurrent_hidden_states = (
            subgoals_done * next_meta_recurrent_hidden_states
            + (1 - subgoals_done) * current_meta_recurrent_hidden_states
        )
        step += 1



def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    add_hrl_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    config_file = os.path.join(
        os.path.dirname(gibson2.__file__), "../igibson_usage/configs", args.config_file
    )
    env_config = parse_config(config_file)

    def load_env(env_mode, device_idx):
        return iGibsonEnv(
            config_file=config_file,
            mode=env_mode,
            action_timestep=args.action_timestep,
            physics_timestep=args.physics_timestep,
            automatic_reset=True,
            device_idx=device_idx,
        )

    sim_gpu_id = [int(gpu_id) for gpu_id in args.sim_gpu_id.split(",")]
    env_id_to_which_gpu = np.linspace(
        0,
        len(sim_gpu_id),
        num=args.num_train_processes + args.num_eval_processes,
        dtype=np.int,
        endpoint=False,
    )
    eval_envs = [
        lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env(
            "gui", device_idx
        )
        for env_id in range(
            args.num_train_processes,
            args.num_train_processes + args.num_eval_processes - 1,
        )
    ]
    eval_envs += [lambda: load_env(args.env_mode, sim_gpu_id[env_id_to_which_gpu[-1]])]
    eval_envs = ParallelNavEnv(eval_envs, blocking=False)
    action_dim = eval_envs.action_space.shape[0]

    action_mask_choices = torch.zeros(2, action_dim, device=device)
    action_mask_choices[0, 0:2] = 1.0
    action_mask_choices[1, :] = 1.0

    cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]

    meta_observation_space = eval_envs.observation_space
    sensor_space = eval_envs.observation_space.spaces[
        "sensor"
    ]  # this is a 3 dimension Box
    subgoal_space = gym.spaces.Box(
        low=-2.0, high=2.0, shape=sensor_space.shape, dtype=np.float32
    )

    subgoal_mask_choices = torch.zeros(2, sensor_space.shape[0], device=device)
    subgoal_mask_choices[0, 0:2] = 1.0
    subgoal_mask_choices[1, :] = 1.0

    rollout_observation_space = eval_envs.observation_space.spaces.copy()
    rollout_observation_space["subgoal"] = subgoal_space
    rollout_observation_space["subgoal_mask"] = gym.spaces.Box(
        low=0, high=1, shape=subgoal_space.shape, dtype=np.float32
    )
    rollout_observation_space["action_mask"] = gym.spaces.Box(
        low=0, high=1, shape=(action_dim,), dtype=np.float32
    )
    rollout_observation_space = gym.spaces.Dict(rollout_observation_space)
    observation_space = rollout_observation_space

    subgoal_tolerance = torch.tensor(1.0 / 3.0, dtype=torch.float32, device=device)

    meta_actor_critic = MetaPolicy(
        observation_space=meta_observation_space,
        subgoal_space=subgoal_space,
        use_action_masks=args.use_action_masks,
        action_masks_dim=action_mask_choices.shape[0],
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        # initial_stddev=initial_stddev,  # these are optional params
        # min_stddev=min_stddev,  # these are optional params
        stddev_transform=torch.nn.functional.softplus,
    )
    meta_actor_critic.to(device)

    meta_agent = PPO(
        meta_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.meta_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=True,
        normalize_advantage=args.meta_agent_normalize_advantage,
    )

    actor_critic = Policy(
        observation_space=observation_space,
        action_space=eval_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=False,
        normalize_advantage=True,
    )

    # TODO: Add Path HERE
    ckpt_path = "/home/vkmittal14/WORKSPACE/TUD/vk_igibson/trained_model_wts/hrl4in_tiago_tabletop_6dim_subgoal_hrl4in_ss/ckpt.23300.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["state_dict"])

    meta_ckpt_path = ckpt_path.replace("ckpt", "meta_ckpt")
    ckpt = torch.load(meta_ckpt_path, map_location=device)
    meta_agent.load_state_dict(ckpt["state_dict"])
    print("Evaluating the Agent")
    evaluate(
        args,
        eval_envs,
        meta_actor_critic,
        actor_critic,
        action_mask_choices,
        subgoal_mask_choices,
        subgoal_tolerance,
        device,
        update=0,
        count_steps=0,
        eval_only=True,
    )
    return

main()