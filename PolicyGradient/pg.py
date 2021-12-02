import torch as t
import numpy as np
import matplotlib.pyplot as plt
import visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyGradient(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_net(self):
        pass

    def choose_action(self, observation):
        pass

    def store_transition(self, s, a, r):
        pass

    def learn(self, s, a, r, _s):
        pass

    def _discount_and_norm_rewards(self):
        pass
