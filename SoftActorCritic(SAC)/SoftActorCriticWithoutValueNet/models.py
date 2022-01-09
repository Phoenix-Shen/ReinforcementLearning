import datetime
from memory import Replay_buffer
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
import gym
from tensorboardX import SummaryWriter
import os
import numpy as np


def layer_init(layer: nn.Linear):
    """
    Get the arguments of uniorm_(from,to)
    Use uniform distribution to init data
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    The actor takes a state as an input and output
    the mean and log standard deviation of an countiouns action
    Then the means and stds are transformed to the continuous number
    by tanh gaussian policy
    """

    def __init__(self,
                 n_features: int,
                 n_acitons: int,
                 hidden_size: int,
                 log_std_max: float,
                 log_std_min: float,
                 lr: float,
                 max_action: float,
                 cuda: bool) -> None:
        super().__init__()
        # save the arguments to memeber variables
        self.max_action = max_action
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.device = t.device("cuda:0" if cuda else "cpu")
        # network architecture
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, n_acitons)
        self.log_std = nn.Linear(hidden_size, n_acitons)
        # to CUDA
        self.to(self.device)
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr)
        # call weight init
        self.init_weights()

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log stadard deviation
        log_std = t.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        # return mean and std for the reparameterization trick
        return mean, log_std.exp()

    def sample_normal(self,
                      obs: Tensor,
                      reparameterize=True,
                      epsilon=1e-6) -> tuple[Tensor, Tensor]:
        """
        sample an action from a normal distribution,
        you can choose whether to use reparameterize option
        (the epsilon is a small number that prevent divide by zero)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        if reparameterize:
            # equals
            # e=distributions.Normal(0, 1).sample()
            # action = mean + e * std
            actions = dist.rsample()
        else:
            actions = dist.sample()

        action = t.tanh(actions)*t.tensor(self.max_action).to(self.device)
        log_probs = dist.log_prob(actions)-t.log(1.-action.pow(2)+epsilon)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def choose_action(self, obs: Tensor) -> float:
        # obs shape: [1,n_features]
        #state = t.FloatTensor(obs).unsqueeze(0).to(self.device)
        actions, _ = self.sample_normal(obs, reparameterize=False)
        # action shape [1,n_actions]
        return actions.cpu().detach().numpy()[0]

    def init_weights(self):
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    """
    the critic takes the (action,state) pair as an input
    and output Q(a,s),which we call state-action value
    """

    def __init__(self,
                 n_features: int,
                 hidden_size: int,
                 lr: float,
                 cuda: bool) -> None:
        super().__init__()
        self.device = t.device("cuda:0" if cuda else "cpu")
        # network architecture
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # to CUDA
        self.to(self.device)
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr)
        # call weight init
        self.weight_init()

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        pair = t.cat([obs, action], dim=1)
        return self.net(pair)

    def weight_init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(*layer_init(layer))


class Agent(nn.Module):
    """
    The agent interacts with the env and learn from experience
    """

    def __init__(self,
                 env: gym.Env,
                 n_features: int,
                 n_actions: int,
                 hidden_size: int,
                 reward_decay: float,
                 buffer_size: int,
                 batch_size: int,
                 tau: float,
                 lr_c: float,
                 lr_a: float,
                 reward_scale: float,
                 log_std_max: float,
                 log_std_min: float,
                 max_action: float,
                 alpha: float,
                 init_exporation_steps: int,
                 n_epochs: int,
                 update_cycle: int,
                 update_target_interval: int,
                 eval_episodes: int,
                 eval_interval: int,
                 log_dir: str,
                 save_frequency: int,
                 save_dir: str,
                 cuda: bool) -> None:
        super().__init__()
        # parameters
        self.env = env
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.reward_decay = reward_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.reward_scale = reward_scale
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.max_action = max_action
        self.alpha = alpha
        self.init_exporation_steps = init_exporation_steps
        self.n_epochs = n_epochs
        self.update_cycle = update_cycle
        self.update_target_interval = update_target_interval
        self.eval_episodes = eval_episodes
        self.eval_interval = eval_interval
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        self.save_dir = save_dir
        self.CUDA = cuda

        # replay buffer
        self.memory = Replay_buffer(self.buffer_size, self.batch_size)

        # Actor network
        self.actor = Actor(self.n_features,
                           self.n_actions,
                           self.hidden_size,
                           self.log_std_max,
                           self.log_std_min,
                           self.lr_a,
                           self.max_action,
                           self.CUDA)
        # Critic network
        self.critic1 = Critic(self.n_features+self.n_actions,
                              self.hidden_size,
                              self.lr_c, self.CUDA)
        self.critic2 = Critic(self.n_features+self.n_actions,
                              self.hidden_size,
                              self.lr_c, self.CUDA)
        # Target Critic network
        self.target_critic1 = Critic(self.n_features+self.n_actions,
                                     self.hidden_size,
                                     self.lr_c, self.CUDA)
        self.target_critic2 = Critic(self.n_features+self.n_actions,
                                     self.hidden_size,
                                     self.lr_c, self.CUDA)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Adaptive temperature α
        self.log_alpha = t.zeros(
            1, requires_grad=True, device='cuda' if self.CUDA else 'cpu')
        self.target_entropy = -self.n_features
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.lr_a)

        # global step which will be used for target net update and tensorboardX's global step
        self.global_step = 0

        # tensorboardX writer
        self.writer = SummaryWriter(self.log_dir)

    def learn(self):
        # fill up the buffer
        self._initial_exploration()
        print("->>> TRAINING ...")
        # reset the env and start to train
        obs = self.env.reset()
        done = False
        for epoch in range(self.n_epochs):
            while not done:
                obs_tensor = t.tensor(
                    obs, dtype=t.float32, device=self.actor.device).unsqueeze(0)
                action = self.actor.choose_action(obs_tensor)

                obs_, reward, done, _ = self.env.step(action)
                self.memory.add(obs, action, reward, obs_, float(done))
                obs = obs_

            # after collecting the samples, start to update the network for many times
            for _ in range(self.update_cycle):
                result = self._update_network()

                # write data to TensorboardX
                for key in result.keys():
                    self.writer.add_scalar(key, result[key], self.global_step)

                self.global_step += 1
            # Evaluate the performance of policy network
            if epoch % self.eval_interval == 0:
                avg_rewards = self._eval()
                self.writer.add_scalar("EVAL:avg_rewards", avg_rewards, epoch)
                print(f"epoch:{epoch}, AVG_REWARDS:{avg_rewards}")
            # Save the model
            if (epoch+1) % self.save_frequency == 0:
                self.save_model()

    def _update_network(self):
        """
        Update the parameters of 5 networks

        actor 2*critic and 2*critic_target
        """
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.target_critic1.train()
        self.target_critic2.train()
        ##################
        #  sample data   #
        ##################
        obses, actions, rewards, obses_, dones = self.memory.sample()
        # to_tensor
        obses = t.FloatTensor(obses).to(self.actor.device)
        actions = t.FloatTensor(actions).to(self.actor.device)
        rewards = t.FloatTensor(rewards).unsqueeze(1).to(self.actor.device)
        obses_ = t.FloatTensor(obses_, ).to(self.actor.device)
        inverse_dones = t.FloatTensor(
            dones).unsqueeze(1).to(self.actor.device)

        # conpute the current actions and log_probs under current policy PI
        actions_, log_prob = self.actor.sample_normal(
            obses, reparameterize=True)

        ##############
        #update alpha#
        ##############

        # if we use adaptive alpha, we should update it
        alpha = 0
        if self.alpha is None:
            # we want to only update alpha , so use the detach method
            alpha_loss = -(self.log_alpha*(log_prob +
                           self.target_entropy).detach().cpu()).mean()
            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()
            # if α is not defined , use the automatically tuning
            alpha = self.log_alpha.exp()
        else:
            # else use fixed value
            alpha = t.tensor(self.alpha).unsqueeze(0).to(self.actor.device)

        ##################
        #update the actor#
        ##################

        # get the Q value for the new actions
        q_actions_ = t.min(self.critic1.forward(
            obses, actions_), self.critic2.forward(obses, actions_))
        actor_loss = (alpha*log_prob-q_actions_).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        ###################
        #update the critic#
        ###################

        # get the Q value according to the past action
        q1_val = self.critic1(obses, actions)
        q2_val = self.critic2(obses, actions)
        # get predicted next_state actions and Qvalues
        with t.no_grad():
            actions_next, log_prob_next = self.actor.sample_normal(
                obses_, reparameterize=True)
            target_q1 = self.target_critic1(obses_, actions_next)
            target_q2 = self.target_critic2(obses_, actions_next)
            target_qvalue_next = t.min(
                target_q1, target_q2)-alpha*log_prob_next.sum(1, keepdim=True)
            target_qvalue = self.reward_scale*rewards + \
                inverse_dones*self.reward_decay*target_qvalue_next
        loss_c1 = 0.5*F.mse_loss(q1_val, target_qvalue)
        loss_c2 = 0.5*F.mse_loss(q2_val, target_qvalue)
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        loss_c1.backward()
        loss_c2.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        ###############################
        # soft update of the targetNet#
        ###############################
        if self.global_step % self.update_target_interval:
            self._update_target_networks()

        ##################################
        # save the losses in a dictionary#
        ##################################
        dic = dict()
        if not self.alpha:
            dic["alpha_loss"] = alpha_loss.item()
            dic["alpha"] = alpha.item()
        dic["c1"] = loss_c1.item()
        dic["c2"] = loss_c2.item()
        dic["actor"] = actor_loss.item()

        return dic

    def _initial_exploration(self):
        """
        use current policy to fill the buffer for the next training operation
        """
        print("Start to fill the buffer")
        obs = self.env.reset()

        for _ in range(self.init_exporation_steps):
            with t.no_grad():
                obs_tensor = t.tensor(
                    obs, dtype=t.float32, device=self.actor.device).unsqueeze(0)
                # generate the policy
                action = self.actor.choose_action(obs_tensor)
            # input the action and get rewards
            obs_, reward, done, _ = self.env.step(action)
            # store in the episode
            self.memory.add(obs, action, reward, obs_, float(done))
            obs = obs_
            if done:
                # if done, then reset the env and start another loop
                obs = self.env.reset()

        print("The buffer has been filled with {} samples and the buffer size is {}".format(
            self.init_exporation_steps, self.buffer_size))

    def _update_target_networks(self):
        """
        soft update
        """
        self._update_target_network(self.critic1, self.target_critic1)
        self._update_target_network(self.critic2, self.target_critic2)

    def _update_target_network(self, current_net: nn.Module, target_net: nn.Module):
        for target_param, param in zip(target_net.parameters(), current_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def _eval(self):
        self.actor.eval()

        total_reward = 0

        for _ in range(self.eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                with t.no_grad():
                    obs_tensor = t.FloatTensor(
                        obs).unsqueeze(0).to(self.actor.device)
                    mean, _ = self.actor.forward(obs_tensor)
                    # we dont need std for exploration
                    action = t.tanh(mean).detach().cpu().numpy()[0]
                obs_, reward, done, _ = self.env.step(action*self.max_action)
                episode_reward += reward
                obs = obs_
            total_reward += episode_reward
        # back to train mode
        self.actor.train()
        return total_reward/self.eval_episodes

    def save_model(self):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        t.save(self.actor.state_dict(), os.path.join(
            self.save_dir, f"ACTOR {time}.pth"))
        t.save(self.critic1.state_dict(), os.path.join(
            self.save_dir, f"CRITIC1 {time}.pth"))
        t.save(self.critic2.state_dict(), os.path.join(
            self.save_dir, f"CRITIC2 {time}.pth"))
        print(f'{time},model saved')

    def load_model(self):
        pass
