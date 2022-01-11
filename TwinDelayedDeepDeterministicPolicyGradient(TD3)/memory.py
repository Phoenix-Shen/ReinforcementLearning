# %%
import gym
import torch as t
import numpy as np


class Replay_buffer():
    def __init__(self, buffer_size, batch_size, n_features, n_actions) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_actions = n_actions
        self.memory = t.zeros((batch_size, n_features*2+2+n_actions))
        self.memory_ptr = 0

    def store_transition(self, s: np.ndarray, a: np.ndarray, r: float, done: float, s_: np.ndarray):
        s = t.tensor(s, dtype=t.float32)
        a = t.tensor(a, dtype=t.float32)
        r = t.tensor(r, dtype=t.float32).unsqueeze(-1)
        done = t.tensor(done, dtype=t.float32).unsqueeze(-1)
        s_ = t.tensor(s_, dtype=t.float32)

        transition = t.hstack(([s, a, r, done, s_]))

        index = self.memory_ptr % self.buffer_size
        self.memory[index, :] = transition
        self.memory_ptr += 1

    def sample(self):
        index = np.random.choice(self.buffer_size, size=self.batch_size)

        states = self.memory[index, :self.n_features]
        actions = self.memory[index,
                              self.n_features:self.n_features+self.n_actions]
        rewards = self.memory[index, self.n_features +
                              self.n_actions:self.n_features+self.n_actions+1]
        dones = self.memory[index, self.n_features +
                            self.n_actions+1:self.n_features+self.n_actions+2]
        states_ = self.memory[index, self.n_features+self.n_actions+1:]

        return states, actions, rewards, dones, states_
