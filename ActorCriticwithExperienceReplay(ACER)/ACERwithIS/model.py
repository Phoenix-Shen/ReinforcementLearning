
from typing import Tuple
import torch as t
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import gym
import memory


class ActorCritic(object):
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 env: gym.Env,
                 lr=1e-4,
                 max_epoch=1500,
                 replay_buffer_size=100,
                 sample_batch_size=32,
                 reward_decay=0.99,
                 render=False) -> None:
        super().__init__()
        # store arguments in member variables
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.max_epoch = max_epoch
        self.env = env
        self.sample_batch_size = sample_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.reward_decay = reward_decay
        self.render = render
        # actor and critic definition
        self.actor = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            # choose softmax in order to output the probabilities of the action
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)  # output the value
        )

        # transefer the parameters to the target device
        self.critic.to(self.device)
        self.actor.to(self.device)

        # memory
        self.memory = memory.ReplayBuffer(self.replay_buffer_size)

        # optimizer
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr)

    def forward(self, observation: t.Tensor) -> Tuple:
        """
        Params
        ----------
        observation:t.tensor
            the states for action probabilities calculation
        """
        observation.to(self.device)
        # choose action under policy pi_theta
        action_probs = self.actor(observation)
        # get the value under policy pi_theta
        action_value = self.critic(observation)
        return action_probs, action_value

    def explore(self):
        state = self.env.reset()
        # dtype = t.float32 device = self.device
        state = t.FloatTensor(state)
        trajectory = []
        while True:
            action_probabilities, _ = self.forward(state)
            action = t.multinomial(action_probabilities, 1)
            exploration_statistics = action_probabilities.unsqueeze(dim=0)
            next_state, reward, done, _ = self.env.step(action.item())
            if self.render:
                self.env.render()
            transition = memory.Transition(state.unsqueeze(0),
                                           action.unsqueeze(0),
                                           t.FloatTensor(reward).unsqueeze(0),
                                           t.FloatTensor(
                                               next_state).unsqueeze(0),
                                           t.FloatTensor(done).unsqueeze(0),
                                           exploration_statistics)
            self.memory.add(transition)
            trajectory.append(transition)
            if done:
                self.env.reset()
                break
            else:
                state = next_state
        return trajectory

    def learn(self):
        for _ in range(self.max_epoch):
            episode_rewards = 0.
            trajectory = self.explore()
            episode_rewards += sum([transition.rewards[0, 0]
                                    for transition in trajectory])
            for _ in range(np.random.poisson(4)):
                trajectory = self.memory.sample(
                    self.sample_batch_size)
                if trajectory:
                    self._learn_from_experience(trajectory)

    def _learn_from_experience(self, trajextory: list):
        """
        Conduct a single discrete learning iteration. Analogue of Algorithm 2 in the paper.
        """
        # select the last element as the terminal
        _, _, _, next_states, _, _ = trajextory[-1]

        action_probs, action_values = self.forward(next_states)
        retrace_action_value = t.sum(
            action_probs*action_values, dim=1).unsqueeze(0)

        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajextory):
            action_probs, action_values = self.forward(states)
            value = t.sum(action_probs*action_values, dim=1).unsqueeze(0)
            if done == 1.:
                value.zero_()
            # importance sampling
            importance_weights = action_probs/exploration_probabilities
            # compute advantage it presents if the action is better than average situation
            naive_advantage = t.gather(action_values, -1, actions)-value
            retrace_action_value = rewards+self.reward_decay * \
                retrace_action_value*(1.-done)
            retrace_advantage = retrace_action_value-value

            # actor loss
            actor_loss = -t.gather(importance_weights, -1, actions).clamp(
                max=10)*retrace_advantage*t.gather(action_probs, -1, actions)

            bias_correction = (1-10/importance_weights).clamp(min=0.) * \
                naive_advantage*action_probs*action_probs.log()
            actor_loss += bias_correction
            actor_loss = t.mean(actor_loss)
            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)

            critic_loss = (t.gather(action_values, -1, actions) -
                           retrace_action_value).pow(2)
            critic_loss = critic_loss.mean()
            self.optimizer_critic.zero_grad()
            critic_loss.backward()

            entropy_loss = (action_probs*t.log(action_probs)).sum(-1).mean()
            entropy_loss.backward()
            retrace_action_value = importance_weights.gather(-1, actions).clamp(max=1.) * \
                (retrace_action_value -
                 action_values.gather(-1, actions).data) + value
            self.optimizer_actor.step()
            self.optimizer_critic.step()
