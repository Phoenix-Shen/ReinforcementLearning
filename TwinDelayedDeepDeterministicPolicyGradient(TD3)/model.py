from numpy import ndarray
import torch as t
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from memory import Replay_buffer
import gym
from tensorboardX import SummaryWriter
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 max_action: float,
                 hidden_size: int,
                 lr: float,
                 ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.max_action = max_action
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(self.n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions),
            nn.Tanh()
        )
        # call init weight function
        self.weight_init()
        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net.forward(obs)*self.max_action

    def weight_init(self):
        """
        Xavier Initialization
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight,  gain=nn.init.calculate_gain('relu'))


class Critic(nn.Module):
    """
    contains 2 critic networks
    """

    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 hidden_size: int,
                 lr: float) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(n_features+n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(n_features+n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # call weight init function
        self.weight_init()
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        input_data = t.cat([obs, action], dim=1)
        return self.q1(input_data), self.q2(input_data)

    def weight_init(self):
        """
        Xavier Initialization
        """
        for layer in self.q1:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight,  gain=nn.init.calculate_gain('relu'))
        for layer in self.q2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight,  gain=nn.init.calculate_gain('relu'))


class Agent():
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 hidden_size: int,
                 max_action: float,
                 lr_a: float,
                 lr_c: float,
                 reward_decay: float,
                 buffer_size: int,
                 batch_size: int,
                 env: gym.Env,
                 max_epoch: int,
                 log_dir: str,
                 tau: float,
                 policy_noise: float,
                 noise_clip: float,
                 update_policy_interval: int,
                 cuda: bool) -> None:
        # save parameters
        self.update_policy_interval = update_policy_interval
        self.max_action = max_action
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.buffer_size = buffer_size
        self.GAMMA = reward_decay
        self.env = env
        self.max_epoch = max_epoch
        self.writer = SummaryWriter(log_dir)
        self.tau = tau
        self.CUDA = cuda
        self.device = t.device("cuda:0" if self.CUDA else "cpu")
        # networks
        self.actor = Actor(n_features, n_actions,
                           max_action, hidden_size, lr_a)
        self.actor_target = Actor(
            n_features, n_actions, max_action, hidden_size, lr_a)

        self.critic = Critic(n_features, n_actions, hidden_size, lr_c)
        self.critic_target = Critic(n_features, n_actions, hidden_size, lr_c)

        # copy parameters
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic_target.state_dict())
        # to cuda if cuda is enabled
        if self.CUDA:
            self.actor.to(self.device)
            self.actor_target.to(self.device)
            self.critic.to(self.device)
            self.critic_target.to(self.device)
        # Memory
        self.mem = Replay_buffer(
            buffer_size, batch_size, n_features, n_actions)

    def choose_action(self, obs: ndarray) -> ndarray:
        obs = t.tensor(obs, dtype=t.float32).unsqueeze(-1)
        return self.actor.forward(obs).cpu().numpy()[0]

    def learn(self):
        for ep in range(self.max_epoch):
            s = self.env.reset()
            ep_reward = 0
            done = False

            while not done:

                action = self.choose_action(s)
                s_, r, done, _ = self.env.step(action)

                self.mem.store_transition(s, action, r, float(done), s_)

                s = s_

            # if buffer is filled with data, start training process
            if self.mem.memory_ptr > self.buffer_size:
                self._update_networks(ep)
                ep += 1

    def _update_networks(self, step):
        s, a, r, done, s_ = self.mem.sample()

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        inverse_done = (1-done).to(self.device)
        s_ = s_.to(self.device)

        with t.no_grad():
            noise = t.clamp((t.rand_like(a)*self.policy_noise),
                            min=-self.noise_clip, max=self.noise_clip)
            action_next = t.clamp(self.actor_target.forward(
                s_)+noise, min=-self.max_action, max=self.max_action)

            tar_q1, tar_q2 = self.critic_target.forward(s_, action_next)
            tar_q = t.min(tar_q1, tar_q2)
            tar_q = r+inverse_done*self.GAMMA*tar_q

        cur_q1, cur_q2 = self.critic(s, a)

        loss_critic = F.mse_loss(cur_q1, tar_q)+F.mse_loss(cur_q2, tar_q)
        self.critic.optimizer.zero_grad()
        loss_critic.backward()
        self.critic.optimizer.step()

        # delayed updates of policy network
        if step % self.update_policy_interval:
            action = self.actor.forward(s)
            q_value = t.min(self.critic.forward(s, action)).mean()
            loss_actor = -q_value
            self.actor.optimizer.zero_grad()
            loss_actor.backward()
            self.actor.optimizer.step()
            # soft update
            self._soft_update_target()
            print(
                f"epoch:[{step}/{self.max_epoch}],l_c: {loss_critic.item()}, l_a={loss_actor.item()}")

    def _soft_update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.copy_(self.tau*param.data +
                               (1-self.tau)*target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.copy_(self.tau*param.data +
                               (1-self.tau)*target_param.data)
