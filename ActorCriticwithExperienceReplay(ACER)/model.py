import torch as t
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


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


class ActorCritic(nn.Module):
    def __init__(self, n_features: int, n_actions: int, lr: float) -> None:
        super().__init__()
        # pass the arguments and store in member variables
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        # define our critic and actor
        self.actor = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # returns the value of the state
        )
        # optimizer definition
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)

        # send the networks to the device if we have a GPU
        self.actor.to(self.device)
        self.critic.to(self.device)

    def forward(self, state):
        pi = self.actor(state)
        value = self.critic(state)
        return (pi, value)


class Agent(object):
    def __init__(self,
                 n_features,
                 n_actions: int,
                 lr: float,
                 reward_decay: float,
                 batch_size: int,
                 mem_size: int) -> None:
        super().__init__()

        # save the member variables
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.mem_size = mem_size

        # init memory and AC network
        self.mem = memory_buffer(self.mem_size, n_features)
        self.ac = ActorCritic(self.n_features, self.n_actions, self.lr)

    def store_transition(self, state, log_prob, reward, state_, done):
        self.mem.store_transitions(state, log_prob, reward, state_, done)

    def choose_action(self, observation):
        state = t.tensor(np.array([observation])).to(self.ac.device)
        prob, _ = self.ac.forward(state)
        prob = F.softmax(prob, dim=1)
        action_probs = t.distributions.Categorical(prob)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        return action.item(), log_probs

    def learn(self):
        # if we dont fill up the mem, then we should keep it collecting experience
        if self.mem.mem_index < self.batch_size:
            return None, None

        state, prob, reward, state_, done = self.mem.sample(
            batch_size=self.batch_size)
        state = t.tensor(state).to(self.ac.device)
        prob = t.tensor(prob).to(self.ac.device)
        reward = t.tensor(reward).to(self.ac.device).unsqueeze(dim=1)
        state_ = t.tensor(state_).to(self.ac.device)
        done = t.tensor(done).to(self.ac.device)

        _, c_value = self.ac.forward(state)
        _, c_value_ = self.ac.forward(state_)

        c_value_[done] = 0.
        # the maximal of the next state(mns)
        delta = reward+self.gamma*c_value_
        # mns minus the current value
        actor_loss = -t.mean(prob*(delta-c_value))
        critic_loss = F.mse_loss(c_value, delta)
        # back propagate and step
        self.ac.optim_actor.zero_grad()
        self.ac.optim_critic.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        self.ac.optim_actor.step()
        self.ac.optim_critic.step()
        return actor_loss.item(), critic_loss.item()
