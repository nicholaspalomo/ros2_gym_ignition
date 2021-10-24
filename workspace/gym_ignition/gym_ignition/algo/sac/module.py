import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .storage import device

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.action_size = action_size

    def hidden(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))

        return mean, x

    def forward(self, state):
        mean, _ = self.hidden(state)

        return mean

    def noiseless_sample(self, state):
        mean, x = self.hidden(state)

        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.noiseless_sample(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action_0 = torch.tanh(z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        log_prob = normal.log_prob(z.to(device)) - torch.log(self.action_range * (1. - action_0.pow(2)) + 1e-6)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_range

        return action, log_prob, z, mean, log_std
    
    def get_action(self, state, deterministic=False):
        mean, log_std = self.noiseless_sample(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action_0 = torch.tanh(z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        
        action = self.action_range*torch.tanh(mean).detach().cpu().numpy() if deterministic else action.detach().cpu().numpy()
        return action

    def sample_action(self, num_envs):
        a=torch.FloatTensor(num_envs, self.action_size).uniform_(-1, 1)
        return self.action_range*a.numpy()

class Critic(nn.Module):
    def __init__(self, state_size, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_size + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x