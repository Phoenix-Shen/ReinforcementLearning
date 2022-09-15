import numpy as np
import threading


class MemoryBuffer(object):
    def __init__(
        self,
        capacity: int,
        obs_shapes: list[int],
        action_shapes: list[int],
        n_agents: int,
    ):
        self.capacity = capacity
        self.n_agents = n_agents

        self.current_size = 0
        # init elements
        self.buffer = dict()

        for i in range(self.n_agents):
            self.buffer["o_%d" % i] = np.empty([capacity, obs_shapes[i]])
            self.buffer["a_%d" % i] = np.empty([capacity, action_shapes[i]])
            self.buffer["r_%d" % i] = np.empty([capacity])
            self.buffer["o_next_%d" % i] = np.empty([capacity, obs_shapes[i]])
        # add thread lock
        self.lock = threading.Lock()

    def store_transition(self, o, a, r, o_next):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.n_agents):
            with self.lock:
                self.buffer["o_%d" % (i)][idxs] = o[i]
                self.buffer["a_%d" % (i)][idxs] = a[i]
                self.buffer["r_%d" % (i)][idxs] = r[i]
                self.buffer["o_next_%d" % (i)][idxs] = o_next[i]

    def _get_storage_idx(self, inc=1):

        if self.current_size + inc <= self.capacity:
            idx = np.arange(self.current_size, self.current_size + inc)

        elif self.current_size < self.capacity:
            # if overflow, then we random select some idx for 0 to current_size
            overflow = inc - (self.capacity - self.current_size)

            idx_a = np.arange(self.current_size, self.capacity)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.capacity, inc)
        self.current_size = min(self.capacity, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx

    def sample(self, batch_size: int) -> dict():
        tmp_buffer = dict()
        # random choose indices
        idx = np.random.randint(0, self.current_size, batch_size)
        # for all agent we extract the experience
        for key in self.buffer.keys():
            tmp_buffer[key] = self.buffer[key][idx]
        return tmp_buffer
