from caffe2.python.workspace import GlobalInit
import tensorboardX
import torch as t
from tensorboardX import SummaryWriter
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym


# 前馈神经网络，它将是我们actor和critic的结构，输入输出的参数不一样
class FeedForwardNN(nn.Module):
    def __init__(self, n_features, n_actions) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, observation: np.ndarray):
        if isinstance(observation, np.ndarray):
            observation = t.FloatTensor(observation)
        return self.net(observation)


class PPO(object):
    def __init__(self, env: gym.Env, sw: tensorboardX.SummaryWriter) -> None:
        super().__init__()
        # GET THE INFO OF ENV
        self.env = env
        self.n_features = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        # INIT HYPERPARAMS
        self._init_hyperparameters()
        # INITIALIZE ACTOR AND CRITIC
        # actor输出动作，critic输出这个环境的价值
        # Step 1 ，initial policy parameters theta, initial value function parameters phi
        self.actor = FeedForwardNN(self.n_features, self.n_actions)
        self.critic = FeedForwardNN(self.n_features, 1)

        # create variable for the matrix and covariance matrix
        self.conv_var = t.full((self.n_actions,), 0.5)
        self.conv_mat = t.diag(self.conv_var)
        # optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        # add to graph
        self.sw = sw
        self.sw.add_graph(
            self.critic, input_to_model=t.rand((self.n_features)))
        self.sw.add_graph(self.actor, input_to_model=t.rand((self.n_features)))

    def learn(self, total_timesteps):
        timesteps_so_far = 0
        global_index = 0
        # Step 2, for k=0,1,2... do
        while timesteps_so_far < total_timesteps:
            # Step 3 collect data
            batch_obs, batch_acts, batch_log_probs, batch_r, batch_rtgs, batch_lens = self.rollout()
            # Step 5 Compute advantage estimates A^t based on the current value function Vphi k
            v, _ = self.evaluate(batch_obs, batch_acts)
            a_k = batch_rtgs-v.detach()

            # Advantage Normalization, It is necessary since the raw advantage behaves unstably
            # 1e-10 is to avoid the possibility of dividing by 0
            a_k = (a_k-a_k.mean())/(a_k.std()+1e-10)

            # Step 6, update the params theta of actor network
            for i in range(self.n_updates_per_iter):
                v, current_log_probs = self.evaluate(batch_obs, batch_acts)
                # calculate pi_theta(a_t|s_t)/pi_k_theta(a_t|s_t)
                ratios = t.exp(current_log_probs-batch_log_probs)
                # calculate surrogate losses
                surr1 = ratios*a_k
                surr2 = t.clamp(ratios, 1-self.clip, 1+self.clip)*a_k
                # calculate the actor loss
                actor_loss = (-t.min(surr1, surr2)).mean()  # 要让他取最大，所以要是负的
                # back propagation
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                # Step 7, Fit value function by regression on MSE.
                loss_func_critic = nn.MSELoss()
                critic_loss = loss_func_critic(v, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                # save the log and print in tensorboardX
                self.sw.add_scalar("critic_loss", critic_loss, global_index)
                self.sw.add_scalar("actor_loss", actor_loss, global_index)
                global_index += 1
                print("TimeSteps:{},Iteration:{},L_A:{},L_C:{}".format(
                    timesteps_so_far, i, actor_loss.item(), critic_loss.item()))
        # Step 8 Finally end for
            timesteps_so_far += np.sum(batch_lens)
            self.sw.add_scalar("avg_reward", np.mean(batch_r))
    # the function to collect data

    def rollout(self):
        batch_observations = []
        batch_actions = []
        batch_logprobabilities = []
        batch_rewards = []
        batch_rewardstogo = []
        batch_lens = []
        # step 3 collect data
        t_batch = 0
        while t_batch < self.timesteps_per_batch:
            ep_r = []
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t_batch += 1
                batch_observations.append(obs)
                action, log_probability = self.get_action(obs)
                obs, r, done, _ = self.env.step(action)
                # store transitions
                ep_r.append(r)
                batch_actions.append(action)
                batch_logprobabilities.append(log_probability)

                if done:
                    break

            batch_lens.append(ep_t+1)
            batch_rewards.append(ep_r)

        batch_observations = t.FloatTensor(np.array(batch_observations))
        batch_actions = t.FloatTensor(np.array(batch_actions))
        batch_logprobabilities = t.FloatTensor(
            np.array(batch_logprobabilities))

        # Step 4 compute rewars-to-go R^t
        batch_rewardstogo = self.compute_rewardtogo(batch_rewards)
        # Return the data
        return batch_observations, batch_actions, batch_logprobabilities, batch_rewards, batch_rewardstogo, batch_lens

    # Hyper params
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iter = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = 0.005

    def get_action(self, obs):
        # we need a way to do some exploreations
        mean = self.actor(obs)
        distribution = t.distributions.MultivariateNormal(mean, self.conv_mat)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        return action.detach().numpy(), log_probability.detach()

    def compute_rewardtogo(self, batch_rewards):
        batch_rewards_togo = []
        for ep_rews in reversed(batch_rewards):
            discounted_reward = 0
            for rew in ep_rews:
                discounted_reward = rew+discounted_reward*self.gamma
                batch_rewards_togo.insert(0, discounted_reward)
        batch_rewards_togo = t.FloatTensor(batch_rewards_togo)
        return batch_rewards_togo

    def evaluate(self, obs, acts):
        # value
        v = self.critic(obs).squeeze()
        # log_probs
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.conv_mat)
        log_probs = dist.log_prob(acts)
        return v, log_probs
