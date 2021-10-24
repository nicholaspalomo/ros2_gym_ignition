from datetime import datetime
import os
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from .storage import RolloutStorage

class TensorboardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def launchTensorboard(self):
        from tensorboard import program
        import webbrowser
        # learning visualizer
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--host', 'localhost'])
        url = tb.launch()
        print("[GYM_IGNITION] Tensorboard session created: " + url)
        webbrowser.open_new(url)

    def __call__(self, scope, data, idx):
        for key in data:
            self.writer.add_scalar(scope + "/" + key, np.mean(data[key]), idx)

    def add_scalar(self, scope, key, data, idx):
        self.writer.add_scalar(scope + "/" + key, data, idx)

class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 lr_factor=0.98,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 device='cpu', # or 'cuda'
                 mini_batch_sampling='in_order', # or 'shuffle'
                 log_intervals=10,
                 policy_weights_init_path=os.getcwd(),
                 nets_dir=['actor_architecture_0.pth', 'actor_distribution_0.pth', 'critic_0.pth']):

        # PPO components
        self.actor = actor
        self.critic = critic

        # load the policy parameters, if found in policy_weights_init_path
        # NOTE: The following lines expect that all the policy parameters are found in the same directory
        self.POLICY_INIT_PATH = policy_weights_init_path + '/trained_params'

        self.nets_dir = nets_dir
        if os.path.isdir(self.POLICY_INIT_PATH):
            nets = [self.actor.architecture, self.actor.distribution, self.critic.architecture]
            for net, net_dir in zip(nets, self.nets_dir):
                if os.path.isfile(self.POLICY_INIT_PATH + '/' + net_dir):
                    print("[ppo.py] Found pre-trained parameters {0} in directory {1}!".format(net_dir, self.POLICY_INIT_PATH))
                    net.load_state_dict(torch.load(self.POLICY_INIT_PATH + '/' + net_dir, map_location=device))
                else:
                    print("[ppo.py] Could not find pre-trained parameters {0} in directory {1}!".format(net_dir, self.POLICY_INIT_PATH))

        self.is_actor_recurrent = actor.architecture.is_recurrent
        self.is_critic_recurrent = critic.architecture.is_recurrent

        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            actor.obs_shape, 
            critic.obs_shape,
            actor.action_shape,
            device,
            is_actor_recurrent=self.is_actor_recurrent,
            is_critic_recurrent=self.is_critic_recurrent,
            hidden_shape_actor=[actor.hidden_shape if self.is_actor_recurrent else None],
            hidden_shape_critic=[critic.hidden_shape if self.is_critic_recurrent else None])

        if mini_batch_sampling == 'shuffle':
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        elif mini_batch_sampling == 'in_order':
            self.batch_sampler = self.storage.mini_batch_generator_inorder
        else:
            raise NameError(
                mini_batch_sampling + ' is not a valid sampling method. Use one of the following: `shuffle`, `in_order`')

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_factor, patience=20, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=2e-4, eps=1e-8, verbose=True)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.tot_timesteps = 0
        self.tot_time = 0
        self.log_intervals = log_intervals
        self.log_data = dict()

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def observe(self, actor_obs, dones=None):
        self.actor_obs = actor_obs
        self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device), terminal=torch.from_numpy(dones).type(torch.FloatTensor).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.detach().cpu().numpy()

    def step(self, value_obs, rews, dones):

        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device), terminal=torch.from_numpy(dones).type(torch.FloatTensor).to(self.device))
        self.storage.add_transitions(
            self.actor_obs,
            value_obs,
            self.actions,
            rews,
            dones,
            values,
            self.actions_log_prob,
            hidden_cell_actor=[self.actor.architecture.hidden_cell if self.is_actor_recurrent else None],
            hidden_cell_critic=[self.critic.architecture.hidden_cell if self.is_critic_recurrent else None])


    def update(self, actor_obs, value_obs, log_this_iteration, update, save=False):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        print('[ppo.py] Update with %i samples ' % (self.storage.step * self.storage.num_envs))
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step(save=save)

        ## Logging ##
        if log_this_iteration:
            mean_std = 0
            if hasattr(self.actor.distribution, 'log_std'):
                mean_std = self.actor.distribution.log_std.exp().mean().item()
            total_steps = self.storage.step * self.storage.num_envs

            self.log_data['mean_std'] = mean_std
            self.log_data['value_function_loss'] = mean_value_loss
            self.log_data['surrogate_loss'] = mean_surrogate_loss

        self.storage.clear()

    def _train_step(self, save=False):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_return = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, \
                critic_obs_batch, \
                actions_batch, \
                target_values_batch, \
                advantages_batch, \
                returns_batch, \
                old_actions_log_prob_batch, \
                actor_hidden_state_batch, \
                critic_hidden_state_batch, \
                actor_hidden_cell_batch, \
                critic_hidden_cell_batch, \
                depth_image_batch, \
                target_depth_image_batch, \
                grayscale_image_batch, \
                target_grayscale_image_batch, \
                time_series_batch, \
                demonstrations_batch \
                    in self.batch_sampler(self.num_mini_batches):

                if self.storage.is_actor_recurrent:
                    self.actor.architecture.hidden_cell = (actor_hidden_state_batch, actor_hidden_cell_batch)

                if self.storage.is_critic_recurrent:
                    self.critic.architecture.hidden_cell = (critic_hidden_state_batch, critic_hidden_cell_batch)

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_return += returns_batch.sum().item()
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.scheduler.step(mean_return / num_updates)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        # save updated network parameters
        if save:
            print("[ppo.py] saving parameters at: ",self.POLICY_INIT_PATH)

            nets = [self.actor.architecture, self.actor.distribution, self.critic.architecture]
            for net, net_dir in zip(nets, self.nets_dir):
                torch.save(net.state_dict(), self.POLICY_INIT_PATH + '/' + net_dir)

        return mean_value_loss, mean_surrogate_loss, locals()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def learn(self, env, saver):
        tensorboard_logger = TensorboardLogger(log_dir=saver.data_dir)
        tensorboard_logger.launchTensorboard()

        n_steps = self.num_transitions_per_env
        total_steps = n_steps * env.num_envs
        avg_rewards = []

        env.reset()
        dones = np.ones((self.num_envs, 1))
        reward_sum = 0
        done_sum = 0
        start = time.time()
        for update in range(1000000):

            if (update+1) % n_steps == 0:
                reward_sum = 0
                done_sum = 0
                start = time.time()

                if self.is_actor_recurrent:
                    self.actor.architecture.get_init_state(self.num_envs)
                
                if self.is_critic_recurrent:
                    self.critic.architecture.get_init_state(self.num_envs)

            # actual training
            obs = env.observe()
            action = self.observe(obs, dones=dones)
            next_obs, reward, dones, _ = env.step(action, False)
            self.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + sum(dones)
            reward_sum = reward_sum + sum(reward)

            save = False
            if (update+1) % n_steps == 0:
                save = True

                self.update(actor_obs=next_obs,
                    value_obs=next_obs,
                    log_this_iteration=True,
                    update=update,
                        save=save)

                end = time.time()

                average_ll_performance = reward_sum / total_steps
                average_dones = done_sum / total_steps
                avg_rewards.append(average_ll_performance)

                tensorboard_logger("ppo", self.log_data, update)
                tensorboard_logger.add_scalar("ppo", "dones", average_dones, update)
                tensorboard_logger.add_scalar("ppo", "mean_reward", average_ll_performance, update)
                tensorboard_logger.add_scalar("ppo", "learning_rate", self.get_lr(), update)

                # Add the extra info and rewards to the Tensorboard plots
                extra_info, rewards, _, _, _ = env.extras()
                tensorboard_logger("simulation_signals", extra_info, update)
                tensorboard_logger("rewards", rewards, update)

                env.print('----------------------------------------------------')
                env.print('{:>6}th iteration'.format(update))
                env.print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
                env.print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
                env.print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
                env.print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
                env.print('----------------------------------------------------\n')