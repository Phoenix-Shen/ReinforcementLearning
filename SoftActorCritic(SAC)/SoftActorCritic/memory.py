# %%
import numpy as np
import random


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

    def sample(self):

        idx = [random.randint(0, len(self.storage) - 1)
               for _ in range(self.batch_size)]
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)


"""# vscode debug cell for test
# %%
buffer = Replay_buffer(1000, 2)

num = [np.array(n) for n in range(10)]

for n in num:
    buffer.add(n, n, n, n, n)
buffer.sample()
# %%
"""
