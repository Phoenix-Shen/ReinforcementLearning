import gym
from numpy import ndarray
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayBuffer
from torch import Tensor
import numpy as np
from tensorboardX import SummaryWriter
import os
import datetime


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, hidden_size) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Tanh(),
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        actions shape : (batch_size , n_actions)
        """
        return self.net.forward(state)


class Critic(nn.Module):
    def __init__(self, n_features, n_actions, hidden_size) -> None:
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Linear(n_features+n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.c2 = nn.Sequential(
            nn.Linear(n_features+n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """
        critic function, double critic architecture
        reutrn shape (batch_size,1) and (batch_size,1)
        """
        input_data = t.concat([state, action], dim=-1)
        return self.c1.forward(input_data),\
            self.c2.forward(input_data)


class Agent():
    def __init__(self, args: dict) -> None:
        # Arguments
        self.env = gym.make(args["env"])
        self.hidden_size = args["hidden_size"]
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.gamma = args["reward_decay"]
        self.tau = args["tau"]
        self.act_noise = args["act_noise"]
        self.target_noise = args["target_noise"]
        self.tar_noise_clip = args["tar_noise_clip"]
        self.policy_delay = args["policy_delay"]
        self.device = t.device("cuda" if args["cuda"] else "cpu")
        self.lr_a = args["lr_a"]
        self.lr_c = args["lr_c"]
        self.epsiodes = args["episodes"]
        self.repeat_times = args["repeat_times"]
        self.summary_writer = SummaryWriter(args["log_dir"])
        self.display_interval = args["display_interval"]
        # save and load model
        self.model_save_dir = args["model_save_dir"]
        self.actor_dir = args["actor_dir"]
        self.critic_dir = args["critic_dir"]
        self.save_frequency = args["save_frequency"]
        # 为了输出数据，所以增加了个全局步数
        self.global_step = 0
        # memory
        self.batch_size = args["batch_size"]
        self.mem_size = int(args["mem_size"])
        self.memory = ReplayBuffer(self.n_features,
                                   self.n_actions,
                                   self.mem_size,
                                   self.batch_size,
                                   self.device)
        # network
        self.actor = Actor(self.n_features, self.n_actions,
                           self.hidden_size).to(self.device)
        # 这个类集成了2个critic,不需要再定义两个，这是twin在算法中的体现
        self.critic = Critic(self.n_features, self.n_actions,
                             self.hidden_size).to(self.device)
        self.actor_target = Actor(
            self.n_features, self.n_actions, self.hidden_size).to(self.device)
        self.critic_target = Critic(
            self.n_features, self.n_actions, self.hidden_size).to(self.device)
        if self.actor_dir is not None and self.critic_dir is not None:
            self.load_model()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.optim_actor = optim.Adam(self.actor.parameters(), self.lr_a)
        self.optim_critic = optim.Adam(self.critic.parameters(), self.lr_c)

    def get_action(self, s: ndarray) -> ndarray:
        s = t.as_tensor(s, dtype=t.float32, device=self.device)
        action = self.actor.forward(s).detach().cpu().numpy()
        action += self.act_noise*np.random.randn(self.n_actions)
        action = np.clip(action, self.env.action_space.low,
                         self.env.action_space.high)
        return action

    def _soft_update_target(self):
        with t.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau*param.data +
                                        (1-self.tau)*target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau*param.data +
                                        (1-self.tau)*target_param.data)

    def _update_networks(self):

        ##################
        # 执行多轮梯度下降 #
        ##################

        for i in range(int(self.repeat_times)):

            ##########
            # 采样数据#
            ##########
            data = self.memory.sample_batch()
            s, a, r, s_, d = data["s"], data["a"], data["r"], data["s_"], data["d"]
            r = r.unsqueeze(-1)
            d = d.unsqueeze(-1)
            ################
            # 首先更新c1，c2 #
            ################

            q1, q2 = self.critic.forward(s, a)
            with t.no_grad():
                target_action = self.actor_target(s_)
                # 添加噪声，使critic平滑
                epsilon = t.randn_like(target_action)*self.target_noise
                epsilon = t.clamp(
                    epsilon, -self.tar_noise_clip, +self.tar_noise_clip)
                target_action = target_action+epsilon
                target_action = t.clamp(
                    target_action, t.as_tensor(self.env.action_space.low, device=self.device), t.as_tensor(
                        self.env.action_space.high, device=self.device))
                # 获得q值，取最小
                q1_tar, q2_tar = self.critic_target.forward(s_, target_action)
                q_tar = t.min(q1_tar, q2_tar)

                backup = r+self.gamma*(1-d)*q_tar
            # MSE loss
            loss_q1 = F.mse_loss(q1, backup)
            loss_q2 = F.mse_loss(q2, backup)
            loss_q = loss_q1+loss_q2
            # 更新critic
            self.optim_critic.zero_grad()
            loss_q.backward()
            self.optim_critic.step()

            ###############
            # 延迟更新actor#
            ###############

            if i % self.policy_delay == 0:
                action = self.actor.forward(s)
                # 这里只取Q1的值
                q1_value, _ = self.critic.forward(s, action)
                loss_a = -q1_value.mean()
                self.optim_actor.zero_grad()
                loss_a.backward()
                self.optim_actor.step()

                ###################
                # 软更新target网络 #
                ###################
                self._soft_update_target()

                # 添加到tensorboard里头
                self.summary_writer.add_scalar(
                    "loss_c1", loss_q1.item(), self.global_step)
                self.summary_writer.add_scalar(
                    "loss_c2", loss_q2.item(), self.global_step)
                self.summary_writer.add_scalar(
                    "loss_a", loss_a.item(), self.global_step)
                self.global_step += 1

    def learn(self):

        for ep in range(self.epsiodes):
            s = self.env.reset()
            ep_reward = 0
            done = False
            while not done:

                # collect actions using random policy when starting the algorithm
                if ep > 20:
                    action = self.get_action(s)
                else:
                    action = self.env.action_space.sample()

                # step
                s_, r, done, _ = self.env.step(action)
                self.memory.store_transition(s, action, r, s_, done)

                s = s_
                ep_reward += r
            self.summary_writer.add_scalar("ep_reward", ep_reward, ep)
            self._update_networks()

            if ep % self.display_interval == 0:
                print(f'epoch[{ep}/{self.epsiodes}],reward->{ep_reward}')
            if (ep+1) % self.save_frequency == 0:
                self.save_model()

    def save_model(self):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        t.save(self.actor.state_dict(), os.path.join(
            self.model_save_dir, f"ACTOR {time}.pth"))
        t.save(self.critic.state_dict(), os.path.join(
            self.model_save_dir, f"CRITIC {time}.pth"))
        print(f'{time},model saved')

    def load_model(self):
        self.actor.load_state_dict(t.load(self.actor_dir))
        self.critic.load_state_dict(t.load(self.critic_dir))
        print(f"{self.actor_dir} and {self.critic_dir} are loaded...")
