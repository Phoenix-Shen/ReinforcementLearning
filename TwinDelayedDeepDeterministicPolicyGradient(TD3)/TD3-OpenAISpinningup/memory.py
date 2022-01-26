import numpy as np
from numpy import float32, ndarray
import torch as t


class ReplayBuffer():
    def __init__(self, n_features, n_actions, mem_size, batch_size, device: t.device) -> None:
        self.states = np.zeros((mem_size, n_features), dtype=np.float32)
        self.next_states = np.zeros((mem_size, n_features), dtype=np.float32)
        self.actions = np.zeros((mem_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros((mem_size), dtype=np.float32)
        self.dones = np.zeros((mem_size), dtype=np.float32)
        # mem pointer
        self.ptr, self.size, self.max_size = 0, 0, mem_size
        self.batch_size = batch_size
        self.device = device

    def store_transition(self, s: ndarray, a: ndarray, r: float32, s_: ndarray, d: float32):
        self.states[self.ptr] = s
        self.next_states[self.ptr] = s_
        self.rewards[self.ptr] = r
        self.actions[self.ptr] = a
        self.dones[self.ptr] = d
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)

        batch = dict(s=self.states[idxs],
                     s_=self.next_states[idxs],
                     a=self.actions[idxs],
                     r=self.rewards[idxs],
                     d=self.dones[idxs])
        return {k: t.as_tensor(v, dtype=t.float32, device=self.device) for k, v in batch.items()}
