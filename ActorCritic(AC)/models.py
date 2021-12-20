from numpy.random import beta
import torch as t
import numpy as np
import torch.nn as nn
from torch.nn.modules import linear
from torch.nn.modules.activation import Softmax
import torch.optim as optim

# actor的目标是最大化奖励期望


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, lr=0.001) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(),
        )

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.action_prob_buffer = None

    def learn(self, td):
        # td是Critic返回的一个值，它的意义是操作a对不对
        log_probability = t.log(self.action_prob_buffer)
        exp_v = log_probability*td.detach()
        loss = -exp_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return exp_v

    def choose_action(self, s):
        # 根据当前状态选择行为a，并返回选择的行为
        s_unsqueeze = t.FloatTensor(s).unsqueeze(dim=0)
        prob_weights = self.net(s_unsqueeze)
        # 根据概率选择动作，保持一定的随机性
        action = np.random.choice(
            range(prob_weights.detach().numpy().shape[1]), p=prob_weights.squeeze(dim=0).detach().numpy())
        # 存储刚刚选择的动作的概率值
        self.action_prob_buffer = prob_weights[0][action]
        return action


# critic负责生成TD_error，它意味着当前的状态比平常好多少？


class Critic(nn.Module):
    def __init__(self, n_features, lr=0.01, GAMMA=0.9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.GAMMA = GAMMA

    def learn(self, s, r, s_):
        # 学习状态的价值，并不是行为的价值
        # 计算TD_error=(r+v_)-v
        # 用TD_error来评判这一步的行为有没有带来比平时更好的结果
        # 返回TD_error
        s = t.FloatTensor(s)
        s_ = t.FloatTensor(s_)
        with t.no_grad():
            v_ = self.net(s_)
        td_error = t.mean((r+self.GAMMA*v_)-self.net(s))
        loss = td_error.square()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return td_error
