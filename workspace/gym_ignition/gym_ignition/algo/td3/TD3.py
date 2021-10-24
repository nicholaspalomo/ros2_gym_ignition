import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self._state_dim = state_dim
		self._action_dim = action_dim

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state).to(device)
		return self.actor(state).cpu().data.numpy()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename, print_msg):
		
		if os.path.isfile(filename + "_critic"):
			self.critic.load_state_dict(torch.load(filename + "_critic"))
			print_msg("Successfully loaded {}".format(filename + "_critic"))

		if os.path.isfile(filename + "_critic_optimizer"):
			self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
			print_msg("Successfully loaded {}".format(filename + "_critic_optimizer"))

		self.critic_target = copy.deepcopy(self.critic)

		if os.path.isfile(filename + "_actor"):
			self.actor.load_state_dict(torch.load(filename + "_actor"))
			print_msg("Successfully loaded {}".format(filename + "_actor"))

		if os.path.isfile(filename + "_actor_optimizer"):
			self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
			print_msg("Successfully loaded {}".format(filename + "_actor_optimizer"))

		self.actor_target = copy.deepcopy(self.actor)

	def run(self, env, file_name="", start_timesteps=100, expl_noise=0.1, max_action=1.0, batch_size=256, max_buffer_size=1e6):
		""" Run the environment and launch the training 

			Inputs:
				env: reference to the gym-ignition environment
				file_name: filename/prefix for network parameter files
		"""

		if not os.path.exists("./results"):
			os.makedirs("./results")

		if file_name != "" and not os.path.exists("./models"):
			os.makedirs("./models")

		if file_name != "":
			self.load(f"./models/{file_name}", env.print)

		replay_buffer = utils.ReplayBuffer(self._state_dim, self._action_dim, max_size=int(max_buffer_size))

		# Start the training...
		state = env.reset()
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0
		for t in range(int(1e6)):
			state = env.observe()

			episode_timesteps += 1

			# Select action randomly or according to policy
			if t < start_timesteps:
				action = env.action_space.sample()
			else:
				action = (
					self.select_action(state)
					+ np.random.normal(0, max_action * expl_noise, size=(env.num_envs, self._action_dim))
				).clip(-max_action, max_action)

			# Perform action
			next_state, reward, done, _ = env.step(action)

			replay_buffer.add(state, action, next_state, reward, done)

			state = next_state
			episode_reward += np.mean(reward)

			# Train agent after collecting sufficient data
			if t >= start_timesteps:
				self.train(replay_buffer, batch_size)

			if (t+1) % env.max_timesteps == 0:
				env.print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1

				state = env.reset()

				self.save(f"./models/{file_name}")