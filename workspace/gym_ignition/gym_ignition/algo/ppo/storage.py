import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device, lfd=False, is_actor_recurrent=False, is_critic_recurrent=False, hidden_shape_actor=None, hidden_shape_critic=None,
    depth_img_dim=None,
    gray_img_dim=None, tcn_in_shape=None):
        self.device = device

        # Core
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape).to(self.device)
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape).to(self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape).to(self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1).byte().to(self.device)

        self.is_actor_recurrent = is_actor_recurrent
        self.is_critic_recurrent = is_critic_recurrent

        if self.is_actor_recurrent:
            self.hidden_shape_actor = [hidden_shape_actor[0][0], num_envs, hidden_shape_actor[0][1]]
            self.hidden_state_actor = torch.zeros(num_transitions_per_env, hidden_shape_actor[0][0], num_envs, hidden_shape_actor[0][1]).to(self.device)
            self.cell_state_actor = torch.zeros(num_transitions_per_env, hidden_shape_actor[0][0], num_envs, hidden_shape_actor[0][1]).to(self.device)

        if self.is_critic_recurrent:
            self.hidden_shape_critic = [hidden_shape_critic[0][0], num_envs, hidden_shape_critic[0][1]]
            self.hidden_state_critic = torch.zeros(num_transitions_per_env, hidden_shape_critic[0][0], num_envs, hidden_shape_critic[0][1]).to(self.device)
            self.cell_state_critic = torch.zeros(num_transitions_per_env, hidden_shape_critic[0][0], num_envs, hidden_shape_critic[0][1]).to(self.device)

        self.has_depth_img = False if depth_img_dim is None else True
        if self.has_depth_img:
            self.depth_image_shape = [depth_img_dim, depth_img_dim]
            self.depth_image = torch.zeros(num_transitions_per_env, num_envs, *self.depth_image_shape).to(self.device) 
            self.depth_image_target = torch.zeros(num_transitions_per_env, num_envs, *self.depth_image_shape).to(self.device) 

        self.has_gray_img = False if gray_img_dim is None else True
        if self.has_gray_img:
            self.gray_image_shape = [gray_img_dim, gray_img_dim]
            self.gray_image = torch.zeros(num_transitions_per_env, num_envs, *self.gray_image_shape).to(self.device) 
            self.gray_image_target = torch.zeros(num_transitions_per_env, num_envs, *self.gray_image_shape).to(self.device) 

        self.has_time_series = False
        if tcn_in_shape is not None:
            self.has_time_series = True
            self.tcn_in_shape = tcn_in_shape
            self.time_series = torch.zeros(num_transitions_per_env, num_envs, *tcn_in_shape).to(self.device)

        self.lfd = lfd
        if lfd:
            self.demonstrations = torch.zeros(num_transitions_per_env, num_envs, *actions_shape).to(self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, actor_obs, critic_obs, actions, rewards, dones, values, actions_log_prob, hidden_cell_actor=None, hidden_cell_critic=None, depth_img=None, depth_img_target=None, gray_img=None, gray_img_target=None, time_series=None, demonstration=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step].copy_(torch.from_numpy(critic_obs).to(self.device))
        self.actor_obs[self.step].copy_(torch.from_numpy(actor_obs).to(self.device))
        self.actions[self.step].copy_(actions.to(self.device))
        self.rewards[self.step].copy_(torch.from_numpy(rewards).view(-1, 1).to(self.device))
        self.dones[self.step].copy_(torch.from_numpy(dones).view(-1, 1).to(self.device))
        self.values[self.step].copy_(values.to(self.device))
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1).to(self.device))

        if hidden_cell_actor[0] is not None:
            self.hidden_state_actor[self.step].copy_(hidden_cell_actor[0][0].detach().to(self.device))
            self.cell_state_actor[self.step].copy_(hidden_cell_actor[0][1].detach().to(self.device))

        if hidden_cell_critic[0] is not None:
            self.hidden_state_critic[self.step].copy_(hidden_cell_critic[0][0].detach().to(self.device))
            self.cell_state_critic[self.step].copy_(hidden_cell_critic[0][1].detach().to(self.device))

        if depth_img is not None:
            self.depth_image[self.step].copy_(torch.from_numpy(depth_img).to(self.device))

        if depth_img_target is not None:
            self.depth_image_target[self.step].copy_(torch.from_numpy(depth_img_target).to(self.device))

        if gray_img is not None:
            self.gray_image[self.step].copy_(torch.from_numpy(gray_img).to(self.device))

        if gray_img_target is not None:
            self.gray_image_target[self.step].copy_(torch.from_numpy(gray_img_target).to(self.device))

        if time_series is not None and self.has_time_series:
            self.time_series[self.step].copy_(torch.from_numpy(time_series).to(self.device))

        if demonstration is not None and self.lfd:
            self.demonstrations[self.step].copy_(torch.from_numpy(demonstration).to(self.device))

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs.view(-1, *self.actor_obs.size()[2:])[indices]
            critic_obs_batch = self.critic_obs.view(-1, *self.critic_obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]

            if self.is_actor_recurrent:
                actor_hidden_state_batch = self.hidden_state_actor.view(self.hidden_shape_actor[0], -1, self.hidden_shape_actor[2])[:, indices, :]
                actor_hidden_cell_batch = self.cell_state_actor.view(self.hidden_shape_actor[0], -1, self.hidden_shape_actor[2])[:, indices, :]
            else:
                actor_hidden_state_batch = None
                actor_hidden_cell_batch = None

            
            if self.is_critic_recurrent:
                critic_hidden_state_batch = self.hidden_state_critic.view(self.hidden_shape_critic[0], -1, self.hidden_shape_critic[2])[:, indices, :]
                critic_hidden_cell_batch = self.cell_state_critic.view(self.hidden_shape_critic[0], -1, self.hidden_shape_critic[2])[:, indices, :]
            else:
                critic_hidden_state_batch = None
                critic_hidden_cell_batch = None

            if self.has_depth_img:
                depth_image_batch = self.depth_image.view(-1, self.depth_image_shape[0], self.depth_image_shape[1])[indices, :, :]
                depth_image_target_batch = self.depth_image_target.view(-1, self.depth_image_shape[0], self.depth_image_shape[1])[indices, :, :]
            else:
                depth_image_batch = None
                depth_image_target_batch = None

            if self.has_gray_img:
                gray_image_batch = self.gray_image.view(-1, self.gray_image_shape[0], self.gray_image_shape[1])[indices, :, :]
                gray_image_target_batch = self.gray_image_target.view(-1, self.gray_image_shape[0], self.gray_image_shape[1])[indices, :, :]
            else:
                gray_image_batch = None
                gray_image_target_batch = None

            if self.has_time_series:
                time_series_batch = self.time_series.view(-1, self.tcn_in_shape[0], self.tcn_in_shape[1])[indices, :, :]
            else:
                time_series_batch = None

            if self.lfd:
                demonstrations_batch = self.demonstrations.view(-1, self.demonstrations.size(-1))[indices]
            else:
                demonstrations_batch = None

            yield actor_obs_batch, \
                critic_obs_batch, \
                actions_batch, \
                values_batch, \
                advantages_batch, \
                returns_batch, \
                old_actions_log_prob_batch, \
                actor_hidden_state_batch, \
                critic_hidden_state_batch, \
                actor_hidden_cell_batch, \
                critic_hidden_cell_batch, \
                depth_image_batch, \
                depth_image_target_batch, \
                gray_image_batch, \
                gray_image_target_batch, \
                time_series_batch, \
                demonstrations_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            indices = list(range(batch_id*mini_batch_size, (batch_id+1)*mini_batch_size))

            if self.is_actor_recurrent:
                actor_hidden_state_batch = self.hidden_state_actor.view(self.hidden_shape_actor[0], -1, self.hidden_shape_actor[2])[:, indices, :]
                actor_hidden_cell_batch = self.cell_state_actor.view(self.hidden_shape_actor[0], -1, self.hidden_shape_actor[2])[:, indices, :]
            else:
                actor_hidden_state_batch = None
                actor_hidden_cell_batch = None

            
            if self.is_critic_recurrent:
                critic_hidden_state_batch = self.hidden_state_critic.view(self.hidden_shape_critic[0], -1, self.hidden_shape_critic[2])[:, indices, :]
                critic_hidden_cell_batch = self.cell_state_critic.view(self.hidden_shape_critic[0], -1, self.hidden_shape_critic[2])[:, indices, :]
            else:
                critic_hidden_state_batch = None
                critic_hidden_cell_batch = None

            if self.has_depth_img:
                depth_image_batch = self.depth_image.view(-1, self.depth_image_shape[0], self.depth_image_shape[1])[indices, :, :]
                depth_image_target_batch = self.depth_image_target.view(-1, self.depth_image_shape[0], self.depth_image_shape[1])[indices, :, :]
            else:
                depth_image_batch = None
                depth_image_target_batch = None

            if self.has_gray_img:
                gray_image_batch = self.gray_image.view(-1, self.gray_image_shape[0], self.gray_image_shape[1])[indices, :, :]
                gray_image_target_batch = self.gray_image_target.view(-1, self.gray_image_shape[0], self.gray_image_shape[1])[indices, :, :]
            else:
                gray_image_batch = None
                gray_image_target_batch = None

            if self.has_time_series:
                time_series_batch = self.time_series.view(-1, self.tcn_in_shape[0], self.tcn_in_shape[1])[indices, :, :]
            else:
                time_series_batch = None

            if self.lfd:
                demonstrations_batch = self.demonstrations.view(-1, self.actions.size(-1))[indices]
            else:
                demonstrations_batch = None

            yield self.actor_obs.view(-1, *self.actor_obs.size()[2:])[indices], \
                  self.critic_obs.view(-1, *self.critic_obs.size()[2:])[indices], \
                  self.actions.view(-1, self.actions.size(-1))[indices], \
                  self.values.view(-1, 1)[indices], \
                  self.advantages.view(-1, 1)[indices], \
                  self.returns.view(-1, 1)[indices], \
                  self.actions_log_prob.view(-1, 1)[indices], \
                  actor_hidden_state_batch, \
                  critic_hidden_state_batch, \
                  actor_hidden_cell_batch, \
                  critic_hidden_cell_batch, \
                  depth_image_batch, \
                  depth_image_target_batch, \
                  gray_image_batch, \
                  gray_image_target_batch, \
                  time_series_batch, \
                  demonstrations_batch
