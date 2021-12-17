
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
                 buffer_size=10000,
                 sample_batch_size=32,
                 reward_decay=0.99) -> None:
        super().__init__()
        # store arguments in member variables
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.max_epoch = max_epoch
        self.env = env
        self.sample_batch_size = sample_batch_size
        self.reward_decay = reward_decay
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
            nn.Linear(256, 1)  # output the value
        )
        # avg network definition
        self.actor_avg = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            # choose softmax in order to output the probabilities of the action
            nn.Softmax(dim=1)
        )
        self.critic_avg = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # output the value
        )
        # transefer the parameters to the target device
        self.critic.to(self.device)
        self.actor.to(self.device)
        self.actor_avg.to(self.device)
        self.critic_avg.to(self.device)

        # memory
        self.memory = memory.memory_buffer(
            buffer_size=buffer_size, input_shape=self.n_features)

    def forward(self, observation: t.Tensor) -> Tuple:
        observation.to(self.device)
        # choose action under policy pi_theta
        policy = self.actor(observation)
        # get the Q value under policy pi_theta
        q = self.critic(observation)
        # v is the expectation of Q value under policy pi_theta,
        # and it presents the "average situation"
        v = (q*policy).sum(dim=1, keepdim=True)
        return (policy, q, v)

    def forward_avg(self, observation: t.Tensor) -> Tuple:
        observation.to(self.device)
        # choose action under policy pi_theta
        policy = self.actor_avg(observation)
        # get the Q value under policy pi_theta
        q = self.critic_avg(observation)
        # v is the expectation of Q value under policy pi_theta,
        # and it presents the "average situation"
        v = (q*policy).sum(dim=1, keepdim=True)
        return (policy, q, v)

    def learn(self):
        for i in range(self.max_epoch):
            # if done then change the done state and reset the environments
            done = False
            observation = self.env.reset()
            # loop in a game, it will keep playing utill the game has finished
            while not done:
                policy, q, v = self.forward(
                    t.Tensor(observation).unsqueeze(dim=0))
                avg_policy, _, _ = self.forward_avg(
                    t.Tensor(observation).unsqueeze(dim=0))
                # sample action from probabilities
                action = t.multinomial(policy.squeeze(dim=0), 1)
                # detach from the graph and transfer it to an single number
                action = action.item()
                # get the log probability
                log_prob = policy.squeeze(dim=0)[action].log()
                # interact with the envoironment
                observation_, reward, done, _ = self.env.step(action)
                # store transitions in memory (state,log_probs,rewards,state_,done)
                self.memory.store_transitions(
                    observation, log_prob, reward, observation_, done)
                # if accmulated experience is not enough for sampling,
                # then pass the training operation, so we use if branch
                ###################
                # Train Operation #
                ###################
                if self.memory.mem_index >= self.sample_batch_size:
                    state, prob, reward, state_, done = self.memory.sample(
                        batch_size=self.sample_batch_size)
                    state = t.tensor(state).to(self.device)
                    prob = t.tensor(prob).to(self.device)
                    # unsqueeze the array to shape (batchsize,1)
                    reward = t.tensor(reward).to(self.device).unsqueeze(dim=1)
                    state_ = t.tensor(state_).to(self.device)
                    done = t.tensor(done).to(self.device)
                    # calculate policy and the value under pi_theta
                    policy, q, v = self.forward(state)
                    _, _, q_ret, _ = self.forward(state_)
                    # get the log probability under policy pi_theta
                    action = t.multinomial(policy.squeeze(dim=0), 1)
                    action = action.item()
                    log_prob = policy.squeeze(dim=0)[action].log()
                    # if done the V(s_i,theta) is 0
                    q_ret[done] = 0
                    # off-policy algorthm needs importance sampling
                    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i)
                    # because we use logprob instead of prob , we have the code here
                    rho = t.exp(log_prob-prob)
                    advantage = reward+q_ret*self.reward_decay-v
                    policy_loss = -(rho.clamp(max=10)*log_prob *
                                    advantage.detach()).mean()
