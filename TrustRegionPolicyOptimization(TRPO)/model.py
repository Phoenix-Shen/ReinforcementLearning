# %%
from typing import NewType
import torch as t
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, n_features, n_actions) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, state: t.Tensor) -> t.Tensor:
        return self.net(state)

    def choose_action(self, state: t.Tensor):
        action_prob = self.forward(state)
        action = t.multinomial(action_prob, 1)
        return action

    def get_kl(self, state: t.Tensor):
        action_prob1 = self.forward(state)
        # the pi_old is detached so that no gradient will flow from it
        action_prob0 = action_prob1.detach()

        kl = action_prob0 * (t.log(action_prob0) - t.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_probability(self, state: t.Tensor, actions: t.Tensor):
        action_prob = self.forward(state)
        return t.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, state: t.Tensor):
        action_prob = self.forward(state)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob


class Critic(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)

        )

    def forward(self, state: t.Tensor):
        return self.net(state)
