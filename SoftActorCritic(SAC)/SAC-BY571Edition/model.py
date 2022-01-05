import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as distributions


def layer_init(layer):
    """
    Get the arguments of uniorm_(from,to)
    Use uniform distribution to init data
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """policy model"""

    def __init__(self, n_features: int, n_actions: int, seed: int, hidden_size=32, init_weights=3e-3, log_std_min=-20, log_std_max=2) -> None:
        super().__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        # save parameters to member variables
        self.seed = t.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.init_weights = init_weights
        # network
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, n_actions)
        self.sigma = nn.Linear(hidden_size, n_actions)
        # cudacuda
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.mu.weight.data.uniform_(-self.init_weights, self.init_weights)
        self.sigma.weight.data.uniform_(-self.init_weights, self.init_weights)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        forward function, the state is transferred to CUDA if you have a gpu
        """
        # use inplace operation to save cuda memory
        state = state.to(self.device)
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = t.clamp(sigma, self.log_std_min, self.log_std_max)
        return mu, sigma

    def evaluate(self, state: Tensor, epsilon=1e-6):
        """
        called in the train procedure not action choosing procedure
        """

        """Means and covariances are not derivable so we 
        need to do some sampleing from another distibution and
        then sampled values are multiplied by the covariance and then
        the mean is added to make the network derivable
        """
        state = state.to(self.device)
        # obtain the original mu and sigma which means mean and log standard deviation
        mu, sigma = self.forward(state=state)
        # get std by exp operation
        std = sigma.exp()
        distribution = distributions.Normal(0, 1)
        e = distribution.sample().to(self.device)
        action = t.tanh(mu+e*std)
        log_prob = distributions.Normal(mu, std).log_prob(mu+e*std)
