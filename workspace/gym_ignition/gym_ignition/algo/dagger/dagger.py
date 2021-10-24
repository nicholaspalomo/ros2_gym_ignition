import os
import torch

import torch.optim as optim
import torch.nn as nn
import numpy as np

# For Python typing
from typing import Callable, Optional, List, Tuple, TypedDict, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from torch import FloatTensor, FloatTensor, BoolTensor

from ...helper.helpers import ConfigurationSaver
from ..ppo.ppo import TensorboardLogger
from ..ppo.storage import RolloutStorage
from ..ppo.module import Actor, Critic, MLP, MultivariateGaussianDiagonalCovariance, RecurrentNet

# Typing helper class with the fields that should be logged periodically during the training
class TensorboardLogData(TypedDict):
    loss_log_std: float
    loss_imitation: float
    loss_value: float
    mean_std: float
    mean_return: float

class DAgger:
    def __init__(self,
                 saver: ConfigurationSaver,
                 actor: Actor,
                 critic: Critic,
                 num_envs: int, # number of epochs that are collected before updating the policy parameters. The length of each epoch is num_transitions_per_env
                 num_transitions_per_env: int,
                 num_learning_epochs: int,
                 num_mini_batches: int,
                 clip_param: Optional[float] = 0.2,
                 use_clipped_value_loss: bool = True,
                 value_loss_coeff: Optional[float] = 5e-4,
                 imitiation_loss_coeff: Optional[float] = 50.,
                 log_std_loss_coeff: Optional[float] = 100.,
                 max_grad_norm: Optional[float] = 0.5,
                 gamma: Optional[float] = 0.998,
                 lam: Optional[float] = 0.95,
                 learning_rate: Optional[float] = 5e-4,
                 lr_factor: Optional[float] = 0.98,
                 device: Optional[str] = 'cpu',
                 mini_batch_sampling: Optional[str] = 'in_order', # or 'shuffle'
                 log_intervals: int = 10,
                 policy_weights_init_path: Optional[str] = os.getcwd(),
                 nets_dir: Optional[List[str]] = ['actor_architecture_0.pth', 'actor_distribution_0.pth', 'critic_0.pth']):

        # For Tensorboard plots
        self.tensorboard_logger = TensorboardLogger(log_dir=saver.data_dir)
        self.tensorboard_logger.launchTensorboard()
        self.update: int = 0

        self.actor: Actor = actor
        self.critic: Critic = critic

        self.num_learning_epochs: float = num_learning_epochs
        self.num_mini_batches: float = num_mini_batches
        self.clip_param: float = clip_param
        self.use_clipped_value_loss: float = use_clipped_value_loss
        self.imitation_loss_coeff: float = imitiation_loss_coeff
        self.value_loss_coeff: float = value_loss_coeff
        self.log_std_loss_coeff: float = log_std_loss_coeff
        self.max_grad_norm: float = max_grad_norm

        self.POLICY_INIT_PATH: str = os.path.join(policy_weights_init_path, 'trained_params')
        self.nets_dir: List[str] = nets_dir

        # Load the policy parameters, if they exist in the provided directory
        if os.path.isdir(self.POLICY_INIT_PATH):
            nets: List[Union[RecurrentNet, MLP, MultivariateGaussianDiagonalCovariance]] = [self.actor.architecture, self.actor.distribution, self.critic.architecture]

            for net, net_dir in zip(nets, self.nets_dir):
                if os.path.isfile(os.path.join(self.POLICY_INIT_PATH, net_dir)):
                    print("[dagger.py] Found pre-trained parameters {0} in directory {1}!".format(net_dir, self.POLICY_INIT_PATH))

                    net.load_state_dict(torch.load(os.path.join(self.POLICY_INIT_PATH, net_dir), map_location=device))

                else:
                    print("[dagger.py] Could not find pre-trained parameters {0} in directory {1}!".format(net_dir, self.POLICY_INIT_PATH))

        self.is_actor_recurrent: bool = actor.architecture.is_recurrent
        self.is_critic_recurrent: bool = critic.architecture.is_recurrent

        # Create an object to store the state transitions for later performing gradient updates
        self.storage: RolloutStorage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor.obs_shape,
            critic.obs_shape,
            actor.action_shape,
            device,
            is_actor_recurrent = self.is_actor_recurrent,
            is_critic_recurrent = self.is_critic_recurrent,
            hidden_shape_actor = [actor.hidden_shape if self.is_actor_recurrent else None],
            hidden_shape_critic = [critic.hidden_shape if self.is_critic_recurrent else None])

        if mini_batch_sampling == 'shuffle':
            self.batch_sampler: Callable[[int], Tuple[np.ndarray]] = self.storage.mini_batch_generator_shuffle
        elif mini_batch_sampling == 'in_order':
            self.batch_sampler: Callable[[int], Tuple[np.ndarray]] = self.storage.mini_batch_generator_inorder
        else:
            raise NameError(mini_batch_sampling + ' is not a valid sampling method. Use one of the following: `shuffle`, `in_order`')

        self.optimizer: Optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)

        # TODO: Make this configurable from the config file or from the function parameters
        self.scheduler: Union[ReduceLROnPlateau, StepLR, ExponentialLR] = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_factor, patience=20, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=2e-4, eps=1e-8, verbose=True)

        # Device - 'cpu' or 'cuda'
        self.device: str = device

        # Logging
        self.tot_timesteps: int = 0
        self.tot_time: int = 0
        self.log_intervals: int = log_intervals
        self.log_data: TensorboardLogData = dict()

        # For calculating the discounted sum of rewards
        self.gamma: float = gamma
        self.lam: float = lam

    def add_transitions(self, 
        actor_obs: np.ndarray, 
        value_obs: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        dones: np.ndarray):

        with torch.no_grad():
            actions = torch.from_numpy(actions).type(torch.FloatTensor).detach().to(self.device)
            actor_obs_t = torch.from_numpy(actor_obs).to(self.device)
            value_obs_t = torch.from_numpy(value_obs).to(self.device)
            dones_t = torch.from_numpy(dones.astype(np.uint8)).type(torch.FloatTensor).to(self.device)

            # Add the state transitions to a buffer
            values: torch.FloatTensor = self.critic.predict(value_obs_t, terminal=dones_t)
            actions_log_prob, _ = self.actor.evaluate(actor_obs_t, actions)

            self.storage.add_transitions(
                actor_obs,
                value_obs,
                actions,
                rewards,
                dones.astype(np.uint8),
                values,
                actions_log_prob,
                hidden_cell_actor=[self.actor.architecture.hidden_cell if self.actor.architecture.is_recurrent else None],
                hidden_cell_critic=[self.critic.architecture.hidden_cell if self.critic.architecture.is_recurrent else None])

    def __train_step(self, save=False):
        mean_value_loss: float = 0
        mean_imitiation_loss: float = 0
        mean_return: float = 0
        mean_log_std_loss: float = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, actor_hidden_state_batch, critic_hidden_state_batch, actor_hidden_cell_batch, critic_hidden_cell_batch \
                    in self.batch_sampler(self.num_mini_batches):

                if self.storage.is_actor_recurrent:
                    self.actor.architecture.hidden_cell = (actor_hidden_state_batch, actor_hidden_cell_batch)

                if self.storage.is_critic_recurrent:
                    self.critic.architecture.hidden_cell = (critic_hidden_state_batch, critic_hidden_cell_batch)

                # Let the action computed by the scripted policy (the joint angle targets) serve as the target label. Use the Mahalanobis distance between the RL policy's output distribution and the deterministic actions of the scripted policy in the loss formulation.
                # policy_stdev = torch.repeat_interleave(self.actor.distribution.log_std.exp().view(-1, actions_batch.shape[1]), repeats=actions_batch.shape[0], dim=0)

                # policy_mean = self.actor.noiseless_action(actor_obs_batch)

                # imitation_loss = self.imitation_loss_coeff * ((actions_batch - policy_mean).square() / (policy_stdev + 1e-5).square()).sum(dim=0).sqrt().mean()

                # Imitation loss
                policy_action = self.actor.noisy_action(actor_obs_batch)
                imitation_loss = self.imitation_loss_coeff * (actions_batch - policy_action).square().mean()

                # Value loss the same as usual
                # Value function loss
                value_batch = self.critic.evaluate(critic_obs_batch)

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + \
                                    (value_batch - target_values_batch).clamp(
                                        -self.clip_param,
                                        self.clip_param)
                    value_losses = (value_batch - returns_batch).square()
                    value_losses_clipped = (value_clipped - returns_batch).square()
                    value_loss = self.value_loss_coeff * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = self.value_loss_coeff * (returns_batch - value_batch).square().mean()

                log_std_loss = self.log_std_loss_coeff * self.actor.distribution.log_std.exp().mean()

                loss = imitation_loss + value_loss + log_std_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_return += returns_batch.mean().item()
                mean_value_loss += value_loss.item()
                mean_imitiation_loss += imitation_loss.item()
                mean_log_std_loss += log_std_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.scheduler.step(mean_return / num_updates)
        mean_value_loss /= num_updates
        mean_imitiation_loss /= num_updates
        mean_log_std_loss /= num_updates

        # save updated network parameters
        if save:
            print("[dagger.py] saving parameters at: ", self.POLICY_INIT_PATH)

            nets = [self.actor.architecture, self.actor.distribution, self.critic.architecture]
            for net, net_dir in zip(nets, self.nets_dir):
                torch.save(net.state_dict(), self.POLICY_INIT_PATH + '/' + net_dir)

        return mean_value_loss, mean_imitiation_loss, mean_return / num_updates, mean_log_std_loss

    def train(self, value_obs: np.ndarray, dones: np.ndarray = None, save=False):

        with torch.no_grad():
            value_obs_t = torch.from_numpy(value_obs).to(self.device)
            dones_t = torch.from_numpy(dones.astype(np.uint8)).to(self.device)
            last_values: FloatTensor = self.critic.predict(value_obs_t, terminal=dones_t if dones is not None else None)

            print('[dagger.py] Updated with {} samples'.format(self.storage.step * self.storage.num_envs))

            self.storage.compute_returns(last_values, self.gamma, self.lam)
        
        # Call the training step
        mean_value_loss, mean_imitiation_loss, mean_return, mean_log_std_loss = self.__train_step(save=save)

        ## Logging ##
        mean_std = 0
        if hasattr(self.actor.distribution, 'log_std'):
            mean_std = self.actor.distribution.log_std.exp().mean().item()

        self.log_data['mean_std'] = mean_std
        self.log_data['loss_value'] = mean_value_loss
        self.log_data['loss_imitation'] = mean_imitiation_loss
        self.log_data['mean_return'] = mean_return
        self.log_data['loss_log_std'] = mean_log_std_loss

        self.tensorboard_logger("imitiation", self.log_data, self.update)

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(self.update))
        print('----------------------------------------------------\n')

        self.update += 1

    def clear_storage(self):

        self.storage.clear()