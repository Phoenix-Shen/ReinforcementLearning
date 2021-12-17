import numpy as np
"""
class EpisodeReplayMemory():
    def __init__(self, capacity: int, max_episode_length) -> None:
        self.num_episodes = capacity//max_episode_length
        self.memory = collections.deque(maxlen=self.num_episodes)
        self.trajectory = []
        self.Transition = collections.namedtuple(
            "Transition", ("state", "action", "reward", "policy"))

    def sample(self, maxlen=0):
        mem = self.memory[random.randrange(len(self.memory))]
        T = len(mem)
        if maxlen > 0 and T > maxlen+1:
            t = random.randrange(T-maxlen-1)
            return mem[t:t+maxlen+1]
        else:
            return mem

    # save the data s_i,a_i,r_i+1 and mu(·|s_i)
    def append(self, state, action, reward, policy):
        self.trajectory.append(self.Transition(state, action, reward, policy))

        # if the epoch ends then turn to the next epoch
        if action is None:
            self.memory.append(self.trajectory)
            self.trajectory = []

    def sample_batch(self, batch_size, maxlen=0):
        batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
        minimum_szie = min(len(traj) for traj in batch)
        batch = [traj[:minimum_szie] for traj in batch]
        return list(map(list, zip(*batch)))

    @property
    def lenth(self):
        return len(self.memory)

    def __len__(self):
        return sum(len(episode for episode in self.memory))
"""


class memory_buffer(object):
    def __init__(self, buffer_size, input_shape) -> None:
        super().__init__()
        # member variables
        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.mem_index = 0
        # buffers
        self.state_buffer = np.zeros(
            (self.buffer_size, self.input_shape), dtype=np.float32)
        self.log_prob_buffer = np.zeros((self.buffer_size), dtype=np.float32)
        self.new_state_buffer = np.zeros(
            (self.buffer_size, self.input_shape), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_size), dtype=np.float32)
        self.terminal_buffer = np.zeros((self.buffer_size), dtype=np.bool8)

    def store_transitions(self, state, log_prob, reward, state_, done):
        # get the index, if the buffer is full the flush the data already exists
        index = self.mem_index % self.buffer_size
        # write data
        self.state_buffer[index] = state
        self.new_state_buffer[index] = state_
        self.log_prob_buffer[index] = log_prob
        self.reward_buffer[index] = reward
        self.terminal_buffer[index] = done
        # increase the pointer
        self.mem_index += 1

    def sample(self, batch_size: int):
        max_mem = min(self.mem_index, self.buffer_size)
        # random sample
        batch = np.random.choice(max_mem, batch_size, False)
        states = self.state_buffer[batch]
        log_probs = self.log_prob_buffer[batch]
        rewards = self.reward_buffer[batch]
        states_ = self.new_state_buffer[batch]
        dones = self.terminal_buffer[batch]

        return states, log_probs, rewards, states_, dones
