# %%

from math import acos
import multiprocessing
from numpy import ndarray
import torch as t
import gym
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np


class BufferTuple:
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.storage_list = list()
        self.transition = namedtuple(
            "Transition",
            ("reward", "done", "state", "action", "log_prob")
        )

    def push(self, *args):
        self.storage_list.append(self.transition(*args))

    def extend_memory(self, storage_list):
        self.storage_list.extend(storage_list)

    def sample_all(self):
        return self.transition(*zip(*self.storage_list))

    def __len__(self):
        return len(self.storage_list)


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, hidden_size) -> None:
        super().__init__()

        self.feedforwardnn = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, n_actions)
        self.log_std = nn.Linear(hidden_size, n_actions)
        self._init_weights()

    def forward(self, obs: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """
        输出均值和方差的log
        """
        feature = self.feedforwardnn(obs)
        return self.mean(feature), self.log_std(feature)

    def choose_action(self, obs: t.Tensor) -> tuple[Tensor, Tensor]:
        """
        输出动作和动作的log值
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = t.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob

    def compute_logprob(self, obs: Tensor, action: Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = t.distributions.Normal(mean, std)

        return dist.log_prob(action).sum(1)

    def _init_weights(self):
        t.nn.init.orthogonal_(self.mean.weight, 1.)  # Tensor正交初始化
        t.nn.init.constant_(self.mean.bias, 1e-6)  # 偏置常数初始化
        t.nn.init.orthogonal_(self.log_std.weight, 1.)  # Tensor正交初始化
        t.nn.init.constant_(self.log_std.bias, 1e-6)  # 偏置常数初始化


class Critic(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_size) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_weights()

    def forward(self, obs: Tensor) -> Tensor:
        return self.net.forward(obs)

    def _init_weights(self):
        t.nn.init.orthogonal_(self.net[-1].weight, 1.)  # Tensor正交初始化
        t.nn.init.constant_(self.net[-1].bias, 1e-6)  # 偏置常数初始化


class GlobalAgent():
    """
    全局的智能体，用于收集子线程Actor的交互数据并进行梯度更新
    """

    def __init__(self, args: dict) -> None:

        # member parameters
        self.env = gym.make(args["env"])
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.hidden_size = args["hidden_size"]
        self.actor = Actor(self.n_features, self.n_actions, self.hidden_size)
        self.critic = Critic(self.n_features, self.hidden_size)
        self.optim_a = optim.Adam(self.actor.parameters(), args["lr_a"])
        self.optim_c = optim.Adam(self.critic.parameters(), args["lr_c"])

    def learn(buffer_data: list):
        pass


class LocalActor():
    """
    多线程里面的actor，仅仅是用于多线程收集数据，增加效率
    """

    def __init__(self, n_features, n_actions, hidden_size, max_mem_size) -> None:
        self.actor = Actor(n_features, n_actions, hidden_size)
        self.buffer = BufferTuple(max_mem_size)

    def collect_data() -> BufferTuple:
        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter <
