import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, theta, mu, sigma, time_scale=1e-1,
                 size=1, initial_value=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.time_scale = time_scale
        self.size = size
        self.initial_value = initial_value if initial_value is not None else np.zeros(size)
        self.previous_value = self.initial_value

    def sample(self):
        value = self.previous_value
        value += self.theta * (self.mu - self.previous_value) * self.time_scale
        value += self.sigma * np.sqrt(self.time_scale) * np.random.normal(size=self.size)
        return value

    def reset(self):
        self.previous_value = self.initial_value

    def sampling_parameters(self):
        mean = self.previous_value + self.theta * (self.mu - self.previous_value) * self.time_scale
        sd = self.sigma * np.sqrt(self.time_scale) * np.ones((self.size,))
        return mean, sd
