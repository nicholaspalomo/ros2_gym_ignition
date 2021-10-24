'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

from tabnanny import check
from telnetlib import WONT
import numpy as np
import random
import copy
from collections import namedtuple, deque

from .module import Actor, Critic, SoftQNetwork
from .storage import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
K_EPOCHS = 1            # number of epochs for mini batch update
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
TARGET_ENTROPY = -1     # target entropy, = -|A|
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_QNET = 3e-4          # learning rate of q-network
LR_ALPHA = 3e-4         # learning rate for temperature
ALPHA = 1.              # entropy weight
UPDATE_EVERY_N = 1      # update network every n time steps
LAYER_DIM = 256         # number of neurons in hidden layer
REWARD_SCALE = 5.       # scaling for the rewards
EXPLORE_STEPS = 1000    # number of steps during which action applied is sampled uniformly at random, helps with exploration
DETERMINISTIC = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self,
        state_size,
        action_size,
        num_agents,
        random_seed,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        k_epochs=K_EPOCHS,
        gamma=GAMMA,
        tau=TAU,
        target_entropy=TARGET_ENTROPY,
        lr_actor=LR_ACTOR,
        lr_qnet=LR_QNET,
        lr_alpha=LR_ALPHA,
        alpha=ALPHA,
        update_every_n=UPDATE_EVERY_N,
        layer_dim=LAYER_DIM,
        reward_scale=REWARD_SCALE,
        explore_steps=EXPLORE_STEPS,
        deterministic=DETERMINISTIC,
        auto_entropy=True,
        device=DEVICE,
        load=True,
        logger=print):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self._log_data = dict()

        action_range = 1.
        self.actor_local = Actor(state_size, action_size, layer_dim, action_range).to(device)

        self.soft_q_net1 = SoftQNetwork(state_size, action_size, layer_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_size, action_size, layer_dim).to(device)

        self.value_local = Critic(state_size, layer_dim).to(device)
        self.value_target = Critic(state_size, layer_dim).to(device)

        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = torch.ones(1, dtype=torch.float32, requires_grad=True, device=device)

        self.soft_q_criterion1 = torch.nn.MSELoss()
        self.soft_q_criterion2 = torch.nn.MSELoss()

        self.soft_q_optimizer = optim.Adam(list(self.soft_q_net1.parameters()) + list(self.soft_q_net2.parameters()), lr=lr_qnet)
        self.policy_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.value_local.parameters(), lr=lr_actor)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

        self._step = 0
        self._logger = logger

        if load:
            self.load_model()

        self._batch_size = batch_size
        self._k_epochs = k_epochs
        self._update_every_n = update_every_n
        self._explore_steps = explore_steps
        self._deterministic = deterministic
        self._reward_scale = reward_scale
        self._auto_entropy = auto_entropy
        self._target_entropy = target_entropy
        self._gamma = gamma
        self._tau = tau
        self._alpha = alpha
        self._device = device

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > (self._batch_size*self._k_epochs + self._explore_steps):
            if(self._step % self._update_every_n == 0):
                self._step = 0
                self._logger("Updating with {} samples.".format(self._batch_size*self._k_epochs*self.num_agents))
                for _ in range(self._k_epochs):
                    experiences = self.memory.sample()
                    self.learn(experiences)
            self._step += 1

    def act(self, state, step=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self._device)
        acts = np.zeros((self.num_agents, self.action_size))
        if step >= self._explore_steps:
            acts = self.actor_local.get_action(state, deterministic=self._deterministic)
        else:
            acts = self.actor_local.sample_action(self.num_agents)

        return acts

    def learn(self, experiences):
        states, actions, reward, next_states, dones = experiences

        with torch.no_grad():
            vf_next_target = self.value_target(next_states)
            next_q_value = reward + (1 - dones) * self._gamma * vf_next_target

        qf1 = self.soft_q_net1(states, actions)
        qf2 = self.soft_q_net2(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.soft_q_optimizer.zero_grad()
        qf_loss.backward()
        self.soft_q_optimizer.step()

        pi, log_pi, _, mean, log_std = self.actor_local.evaluate(states)

        qf1_pi = self.soft_q_net1(states, pi)
        qf2_pi = self.soft_q_net2(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # Regularization loss
        reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        policy_loss = policy_loss + reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        vf = self.value_local(states)
    
        with torch.no_grad():
            vf_target = min_qf_pi - (self.alpha * log_pi)

        vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        self.critic_optimizer.step()

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if self._auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_pi + self._target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            self._log_data['alpha'] = self.alpha.item()
            self._log_data['alpha_loss'] = alpha_loss.item()
        else:
            self.alpha = self._alpha
            alpha_loss = 0
            self._log_data['alpha'] = self.alpha
            self._log_data['alpha_loss'] = alpha_loss

        self.soft_update(self.value_local, self.value_target, self._tau)

        # Append to the log for plotting in Tensorboard
        self._log_data['q_value_loss1'] = qf1_loss.item()
        self._log_data['q_value_loss2'] = qf2_loss.item()
        self._log_data['policy_loss'] = policy_loss.item()
        self._log_data['value_loss'] = vf_loss.item()
        self._log_data['policy_mean'] = mean.mean().item()
        self._log_data['policy_std'] = log_std.exp().mean().item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            target_param.requires_grad = False

    def save_model(self):
        torch.save({
            'actor_state_dict'      : self.actor_local.state_dict(),
            'q1_state_dict'         : self.soft_q_net1.state_dict(),
            'q2_state_dict'         : self.soft_q_net2.state_dict(),
            'value_state_dict'      : self.value_local.state_dict(),
            'alpha_state_dict'      : self.alpha,
            'actor_optimizer_state_dict'    : self.policy_optimizer.state_dict(),
            'q_optimizer_state_dict'       : self.soft_q_optimizer.state_dict(),
            'value_optimizer_state_dict'    : self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict'    : self.alpha_optimizer.state_dict()
        }, 'trained_params/sac_agent.pth')

    def load_model(self):

        import os.path
        model_param_path = 'trained_params/sac_agent.pth'
        if os.path.isfile(model_param_path):
            self._logger("Found saved parameters in file {}! Attempting to load...".format(model_param_path))
            
            checkpoint = torch.load(model_param_path)
            
            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
            self.value_local.load_state_dict(checkpoint['value_state_dict'])
            self.soft_q_net1.load_state_dict(checkpoint['q1_state_dict'])
            self.soft_q_net2.load_state_dict(checkpoint['q2_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.soft_q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = checkpoint['alpha_state_dict']

            for target_param, param in zip(self.value_target.parameters(), self.value_local.parameters()):
                target_param.data.copy_(param.data)

            self._logger("Successfully loaded parameters from {}!".format(model_param_path))