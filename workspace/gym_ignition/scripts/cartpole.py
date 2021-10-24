#!/usr/bin/env python3

# required imports
import os
import sys
import math
from ruamel.yaml import YAML
import rclpy
from gym_ignition.GymIgnitionVecEnv import GymIgnitionVecEnv as Environment

from gym_ignition.helper.helpers import ConfigurationSaver

import torch
from torch import nn as nn
from gym_ignition.algo.ppo import module as ppo_module
from gym_ignition.algo.ppo import PPO

from gym_ignition.algo.sac.sac import SAC
from gym_ignition.algo.sac.agent import Agent

def main():
    __CONFIG_PATH__ = sys.argv[2]
    __ENV_PATH__ = sys.argv[4]

    # directories
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    log_path = os.path.join(root, 'data', 'cartpole')

    cfg = YAML().load(open(__CONFIG_PATH__, 'r'))

    # create an environment
    env = Environment(cfg['environment'])

    # save the current training configuration
    saver = ConfigurationSaver(
        log_dir=log_path,
        save_items=[__CONFIG_PATH__, __ENV_PATH__])

    print('[cartpole.py] Saving log files for current training run in: {}'.format(saver.data_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_node = cfg['environment']
    algo_node = env_node['ppo']['algorithm']
    ppo_node = env_node['ppo']
    sac_node = env_node['sac']
    n_steps = math.floor(env_node['max_time'] / env_node['control_dt'])

    # policy = Agent(env.num_obs, 
    #                env.num_acts, 
    #                env.num_envs, 
    #                123,
    #                auto_entropy=False,
    #                alpha=sac_node['alpha'],
    #                batch_size=sac_node['batch_size'],
    #                k_epochs=sac_node['k_epochs'],
    #                gamma=sac_node['gamma'],
    #                tau=sac_node['tau'],
    #                target_entropy=sac_node['target_entropy'],
    #                lr_actor=sac_node['lr_actor'],
    #                lr_qnet=sac_node['lr_qnet'],
    #                lr_alpha=sac_node['lr_alpha'],
    #                update_every_n=sac_node['update_every_n'],
    #                layer_dim=sac_node['layer_dim'],
    #                reward_scale=sac_node['reward_scale'],
    #                explore_steps=sac_node['explore_steps'])
    # sac = SAC(policy, n_steps)
    # sac.learn(env, saver)

    # # feedforward policy
    actor = ppo_module.Actor(
        ppo_module.MLP(ppo_node['architecture']['policy'],  # number of layers and neurons in each layer
        getattr(nn, ppo_node['architecture']['activation']),  # activation function at each layer
        env.num_obs,  # number of states (input dimension)
        env.num_acts,  # number of actions (output)
        ppo_node['architecture']['init_scale']),
        ppo_module.MultivariateGaussianDiagonalCovariance(env.num_acts, 1.0),
        device)

    critic = ppo_module.Critic(
        ppo_module.MLP(ppo_node['architecture']['value_net'],
        getattr(nn, ppo_node['architecture']['activation']),
        env.num_obs,
        1,
        ppo_node['architecture']['init_scale']), device)

    ppo = PPO(actor=actor,
            critic=critic,
            num_envs=env_node['num_envs'],
            num_transitions_per_env=n_steps,
            num_learning_epochs=algo_node['epoch'],
            clip_param=algo_node['clip_param'],
            gamma=algo_node['gamma'],
            lam=algo_node['lambda'],
            entropy_coef=algo_node['entropy_coeff'],
            learning_rate=algo_node['learning_rate'],
            lr_factor=algo_node['lr_factor'],
            num_mini_batches=algo_node['minibatch'],
            device=device,
            nets_dir=['actor_architecture_cartpole.pth', 'actor_distribution_cartpole.pth', 'critic_cartpole.pth'])

    ppo.learn(env, saver)

    env.close()

if __name__ == '__main__':
    rclpy.init()
    main()