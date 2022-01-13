# %%
import numpy as np
import random

"""
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

"""
"""# vscode debug cell for test
# %%
buffer = Replay_buffer(1000, 2)

num = [np.array(n) for n in range(10)]

for n in num:
    buffer.add(n, n, n, n, n)
buffer.sample()
# %%
"""


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
