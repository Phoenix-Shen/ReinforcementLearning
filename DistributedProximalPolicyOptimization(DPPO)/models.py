import multiprocessing
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, lr, hidden_size) -> None:
        super().__init__()

        self.net =
