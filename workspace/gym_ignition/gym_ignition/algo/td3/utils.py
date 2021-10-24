import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		left = self.ptr*state.shape[0]
		right = (self.ptr+1)*state.shape[0]
		if right >= self.max_size:
			idx = max(state.shape[0] - (right - self.max_size), 0)
		else:
			idx = state.shape[0]
		right = min(right, self.max_size)

		self.state[left:right, :] = state[:idx, ...]
		self.action[left:right, :] = action[:idx, ...]
		self.next_state[left:right, :] = next_state[:idx, ...]
		self.reward[left:right, :] = reward[:idx, ..., np.newaxis]
		self.not_done[left:right, :] = (np.ones(done[:idx, ...].shape) - done[:idx, ...])[:, np.newaxis]

		self.ptr = (self.ptr + 1) % (self.max_size // state.shape[0])
		self.size = min(self.size + state.shape[0], self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)