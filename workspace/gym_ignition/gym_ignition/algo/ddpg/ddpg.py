"""
DDPG (Actor-Critic)
"""

import os.path
import torch
import random
import numpy as np
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent

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
class DDPG():
    def __init__(self,
                 policy: Agent,
                 num_transitions_per_episode):

        self._policy = policy

        # Load trained  Actor and Critic network weights for agent
        an_filename = "ddpgActor_Model.pth"
        if os.path.isfile(an_filename):
            self._policy.actor_local.load_state_dict(torch.load(an_filename))
        cn_filename = "ddpgCritic_Model.pth"
        if os.path.isfile(cn_filename):
            self._policy.critic_local.load_state_dict(torch.load(cn_filename))

        self._num_transitions_per_episode = num_transitions_per_episode
        self._log_data = dict()
        self._policy._log_data = self._log_data

    def learn(self, env, saver):
        tensorboard_logger = TensorboardLogger(log_dir=saver.data_dir)
        tensorboard_logger.launchTensorboard()
        self._policy._logger = env.print

        episode = 0
        while(True):
            reward_sum = 0
            done_sum = 0
            start = time.time()

            env.reset()
            for step in range(self._num_transitions_per_episode):
                obs = env.observe()
                action = self._policy.act(obs, add_noise=True)
                obs_next, reward, done, info = env.step(action)

                # Add (S, A, R, S') to the training buffer
                self._policy.step(obs, action, reward, obs_next, done)

                reward_sum += np.mean(reward)
                done_sum += np.mean(done)
            reward_sum /= self._num_transitions_per_episode
            done_sum /= self._num_transitions_per_episode
            end = time.time()

            env.print('----------------------------------------------------')
            env.print('{:>6}th iteration'.format(episode))
            env.print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_sum)))
            env.print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(done_sum)))
            env.print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
            env.print('----------------------------------------------------\n')
            
            self._log_data['reward'] = reward_sum
            self._log_data['dones'] = done
            info, rewards, _, _, _ = env.extras()
            tensorboard_logger('training', self._log_data, episode)
            tensorboard_logger('simulation_signals', info, episode)
            tensorboard_logger('rewards', rewards, episode)

            episode += 1

            # Save trained  Actor and Critic network weights for agent
            an_filename = "ddpgActor_Model.pth"
            torch.save(self._policy.actor_local.state_dict(), an_filename)
            cn_filename = "ddpgCritic_Model.pth"
            torch.save(self._policy.critic_local.state_dict(), cn_filename)