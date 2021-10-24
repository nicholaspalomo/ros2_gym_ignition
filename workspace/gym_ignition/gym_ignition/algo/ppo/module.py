import torch.nn as nn
import numpy as np
import torch
from torch.distributions import MultivariateNormal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs, terminal=None):
        logits = self.architecture.architecture(obs, terminal)
        actions, log_prob = self.distribution.sample(logits)
        return actions.detach().cpu(), log_prob.detach().cpu()

    def evaluate(self, obs, actions, terminal=None):
        action_mean = self.architecture.architecture(obs, terminal)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs, terminal=None):
        return self.architecture.architecture(obs, terminal)

    def noisy_action(self, obs, terminal=None):
        actions, _ = self.distribution.sample(self.architecture.architecture(obs, terminal))
        return actions

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

    @property
    def hidden_shape(self):
        return self.architecture.hidden_shape

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs, terminal=None):
        return self.architecture.architecture(obs, terminal).detach()

    def evaluate(self, obs, terminal=None):
        return self.architecture.architecture(obs, terminal)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def hidden_shape(self):
        return self.architecture.hidden_shape

class RecurrentNet(nn.Module):
    def __init__(self, hidden_size, activation_fn, input_size, output_size, num_layers, init_scale, device='cpu'):
        super(RecurrentNet, self).__init__()

        self.is_recurrent = True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.policy_layer_logits = nn.Linear(hidden_size, output_size)
        self.action_dim = output_size
        self.hidden_cell = None
        self.device = device
        self.activation_fn = activation_fn()

        self.input_shape = [input_size]
        self.output_shape = [output_size]
        self.hidden_shape = [num_layers, hidden_size]

        scales = [init_scale] * (4 * num_layers)
        self.init_weights(self.lstm, scales)
        self.init_weights(self.hidden_layer, init_scale)
        self.init_weights(self.policy_layer_logits, init_scale)

    @staticmethod
    def init_weights(sequential, scales, bias_init=1.0):

        if isinstance(sequential, nn.LSTM):
            for idx, (name, param) in enumerate(sequential.named_parameters()):
                if 'bias' in name:
                    nn.init.constant(param, bias_init)
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=scales[idx])
        elif isinstance(sequential, nn.Linear):
            torch.nn.init.orthogonal_(sequential.weight, gain=scales)
        else: # nn.Sequential
            MLP.init_weights(sequential, scales)

    def get_init_state(self, batch_size):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

    def forward(self, obs, terminal=None):
        batch_size = obs.shape[0]
        
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size)
        
        if terminal is not None:
            self.hidden_cell = [cell * (1. - terminal).reshape(1, batch_size, 1) for cell in self.hidden_cell]

        _, self.hidden_cell = self.lstm(obs.unsqueeze(0), self.hidden_cell)
        hidden_output = self.activation_fn(self.hidden_layer(self.hidden_cell[0][-1]))

        return self.policy_layer_logits(hidden_output)

    def sample(self, obs, terminal=None):

        return self.forward(obs, terminal=terminal)

    def architecture(self, obs, terminal=None):

        return self.forward(obs, terminal=terminal)

class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size, init_scale, device='cpu'):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.device = device
        self.is_recurrent = False

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [init_scale]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(init_scale)

        modules.append(nn.Linear(shape[-1], output_size))
        self.mlp = nn.Sequential(*modules)
        scale.append(init_scale)

        self.init_weights(self.mlp, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def sample(self, obs, terminal=None):

        return self.forward(obs)

    def forward(self, obs, terminal=None):

        return self.mlp(obs)

    def architecture(self, obs, terminal=None):

        return self.forward(obs)

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.distribution = MultivariateNormal(logits, covariance)

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(logits, covariance)

        actions_log_prob = distribution.log_prob(outputs)
        entropy = distribution.entropy()

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()