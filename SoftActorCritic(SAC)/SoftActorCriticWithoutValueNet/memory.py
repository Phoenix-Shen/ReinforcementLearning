# %%
import numpy as np


class Replay_buffer():
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.storage = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, obs_: np.ndarray, done: float):
        data = (obs, action, reward, obs_, done)
        if self.ptr >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
        # reset the ptr if reached maximum capacity
        self.ptr = (self.ptr+1) % self.buffer_size

    # do not use for loop to sample beacuse it is
    def sample(self):
        indexes = np.random.choice(range(len(self.storage)), self.batch_size)
        samples = np.array(self.storage)[indexes]

        obs, action, reward, obs_, done = \
            samples[:, 0], samples[:, 1], samples[:,
                                                  2], samples[:, 3], samples[:, 4]
        return np.array(obs), np.array(action), np.array(reward), np.array(obs_), np.array(done)


"""
# vscode debug cell for test
# %%
buffer = Replay_buffer(1000, 2)

num = [np.array(n) for n in range(10)]

for n in num:
    buffer.add(n, n, n, n, n)
buffer.sample()
# %%
"""
