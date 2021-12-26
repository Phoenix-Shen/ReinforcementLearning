import torch as t
import torch.nn as nn
import numpy as np
from torch.nn.functional import fractional_max_pool2d_with_indices
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self,
                 n_features,
                 n_actions,
                 lr: float,
                 reward_decay: float,
                 epsilon: float,
                 eps_dec: float,
                 eps_min: float) -> None:
        super().__init__()
        # member variables
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = reward_decay
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.epsilon = epsilon

        # neural network
        self.net = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        # optimizer, loss function and device
        self.optimzer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.to(self.device)
        self.lossfunc = nn.MSELoss()

    def forward(self, state: t.Tensor) -> t.Tensor:
        state = state.to(self.device)
        return self.net(state)

    def choose_action(self, state: t.FloatTensor) -> int:
        if np.random.random() > self.epsilon:
            state = state.to(self.device)
            actions = self.forward(state)
            action = t.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self, state, action, reward, state_):
        self.optimzer.zero_grad()
        states = t.FloatTensor(state).to(self.device)
        actions = t.tensor(action).to(self.device)
        rewards = t.tensor(reward, dtype=t.float32).to(self.device)
        states_ = t.FloatTensor(state_).to(self.device)

        q_pred = self.forward(states)[actions]

        q_target = rewards+self.gamma * self.forward(states_).max()

        loss = self.lossfunc(q_pred, q_target)
        loss.backward()
        self.optimzer.step()
        self._decrement_epsilon()
        return loss.item()

    def _decrement_epsilon(self):
        self.epsilon = self.epsilon-self.eps_dec\
            if self.epsilon > self.eps_min else self.eps_min
