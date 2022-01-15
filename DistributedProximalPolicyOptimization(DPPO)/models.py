# %%
from multiprocessing.connection import Connection
import datetime
import os
from typing import OrderedDict
from numpy import ndarray
import torch as t
import gym
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
Transition = namedtuple(
    "Transition",
    ("reward", "mask", "state", "action", "log_prob")
)


class BufferTuple:
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.storage_list = list()

    def push(self, *args):
        self.storage_list.append(Transition(*args))

    def extend_memory(self, storage_list):
        self.storage_list.extend(storage_list)

    def sample_all(self):
        return Transition(*zip(*self.storage_list))

    def __len__(self):
        return len(self.storage_list)


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, hidden_size) -> None:
        super().__init__()

        self.feedforwardnn = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, n_actions)
        self.log_std = nn.Linear(hidden_size, n_actions)
        self._init_weights()

    def forward(self, obs: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """
        输出均值和方差的log
        """
        feature = self.feedforwardnn(obs)
        return self.mean(feature), self.log_std(feature)

    def choose_action(self, obs: ndarray, device: t.device) -> tuple[Tensor, Tensor]:
        """
        输出动作和动作的log值
        """
        obs_tensor = t.tensor(obs, dtype=t.float32, device=device).unsqueeze(0)
        mean, log_std = self.forward(obs_tensor)
        std = log_std.exp()

        dist = t.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy()[0], log_prob.sum(1)

    def compute_logprob(self, obs: Tensor, action: Tensor) -> Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = t.distributions.Normal(mean, std)

        return dist.log_prob(action).sum(1, keepdim=True)

    def _init_weights(self):
        t.nn.init.orthogonal_(self.mean.weight, 1.)  # Tensor正交初始化
        t.nn.init.constant_(self.mean.bias, 1e-6)  # 偏置常数初始化
        t.nn.init.orthogonal_(self.log_std.weight, 1.)  # Tensor正交初始化
        t.nn.init.constant_(self.log_std.bias, 1e-6)  # 偏置常数初始化

    def cpu_state_dict(self):
        state_dict_cpu = dict()

        for key in self.state_dict():
            state_dict_cpu[key] = self.state_dict()[key].cpu()
        return state_dict_cpu


class Critic(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_size) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_weights()

    def forward(self, obs: Tensor) -> Tensor:
        return self.net.forward(obs)

    def _init_weights(self):
        t.nn.init.orthogonal_(self.net[-1].weight, 1.)  # Tensor正交初始化
        t.nn.init.constant_(self.net[-1].bias, 1e-6)  # 偏置常数初始化


class GlobalAgent():
    """
    全局的智能体，用于收集子线程Actor的交互数据并进行梯度更新
    """

    def __init__(self, args: dict) -> None:

        # member parameters
        self.env = gym.make(args["env"])
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high
        self.hidden_size = args["hidden_size"]
        self.ratio_clamp = args["ratio_clamp"]
        self.lambda_adv = args["lambda_adv"]
        self.lambda_entropy = args["lambda_entropy"]
        self.update_steps = args["update_steps"]
        self.n_threads = args["n_threads"]
        self.batch_size = args["batch_size"]
        self.device = t.device("cuda:0" if args["cuda"] else "cpu")
        self.gamma = args["reward_decay"]
        self.model_save_dir = args["model_save_dir"]
        self.actor_dir = args["actor_dir"]
        self.critic_dir = args["critic_dir"]
        self.max_episode = args["max_episode"]
        self.writer = SummaryWriter(args["log_dir"])
        # Actor Critic and the optimizer corresponding to them
        self.actor = Actor(self.n_features, self.n_actions, self.hidden_size)
        self.critic = Critic(self.n_features, self.hidden_size)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.optim_a = optim.Adam(self.actor.parameters(), args["lr_a"])
        self.optim_c = optim.Adam(self.critic.parameters(), args["lr_c"])
        # loss function for updating critic
        self.loss_func = nn.SmoothL1Loss()
        # global step will be used in printing data to tensorboard
        self.global_step = 0
        if self.actor_dir is not None and self.critic_dir is not None:
            self.load_model()

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

    def learn(self, buffer_data: list):
        # switch to train mode
        self.actor.train()
        self.critic.train()

        #########################
        # decode the buffer data#
        #########################

        [r, m, s, a, log] = [tuple() for _ in range(5)]
        for buffer in buffer_data:
            r += buffer[0]
            m += buffer[1]
            s += buffer[2]
            a += buffer[3]
            log += buffer[4]

        batch = Transition(r, m, s, a, log)

        rewards, masks, states, actions, log_probs = [
            t.tensor(ary, dtype=t.float32, device=self.device)
            for ary in (np.array(batch.reward), np.array(batch.mask), np.array(batch.state), np.array(batch.action), np.array(batch.log_prob))
        ]

        ####################################################
        # get the values of the states using critic network#
        ####################################################

        with t.no_grad():
            values = t.cat(
                [self.critic(states[i:i+self.batch_size]) for i in range(0, states.size()[0], self.batch_size)], dim=0
            )

        ###############################################
        # compute old policy value and advantage value#
        ###############################################

        delta = t.empty(states.size()[0],
                        dtype=t.float32, device=self.device)
        old_policy_value = t.empty(states.size()[0],
                                   dtype=t.float32, device=self.device)
        advantage_value = t.empty(states.size()[0],
                                  dtype=t.float32, device=self.device)

        prev_old_v, prev_new_v, prev_adv_v = 0, 0, 0

        for i in range(states.size()[0]-1, -1, -1):
            delta[i] = rewards[i]+masks[i]*self.gamma*prev_new_v-values[i]

            old_policy_value[i] = rewards[i]+masks[i]*self.gamma*prev_old_v
            advantage_value[i] = delta[i]+masks[i] * \
                self.gamma*prev_adv_v*self.lambda_adv

            prev_old_v = old_policy_value[i]
            prev_new_v = values[i]
            prev_adv_v = advantage_value[i]
        # normlize the advantages
        advantage_value = (advantage_value -
                           advantage_value.mean())/(advantage_value.std()+1e-6)

        #############################################
        # sample a batch from the buffer and        #
        # perform multiple rounds of gradient update#
        #############################################

        sample_steps = int(self.update_steps*states.size()[0]/self.batch_size)

        for _ in range(sample_steps):

            indices = np.random.randint(
                0, states.size()[0], size=self.batch_size)

            state = states[indices]
            action = actions[indices]
            advantage = advantage_value[indices].unsqueeze(1)
            old_value = old_policy_value[indices].unsqueeze(1)
            old_log_prob = log_probs[indices]

            ########################
            # update critic nework #
            ########################

            new_log_prob = self.actor.compute_logprob(state, action)
            new_value = self.critic.forward(state)

            critic_loss = (self.loss_func(
                new_value, old_value)/(old_value.std()+1e-6))

            self.optim_c.zero_grad()
            critic_loss.backward()
            self.optim_c.step()

            #######################
            # update actor nework #
            #######################

            # surrogate objective of TRPO
            ratio = t.exp(new_log_prob-old_log_prob)
            surrobj0 = advantage*ratio
            surrobj1 = advantage * \
                ratio.clamp(1-self.ratio_clamp, 1+self.ratio_clamp)
            surrobj = -t.min(surrobj0, surrobj1).mean()
            loss_entropy = (t.exp(new_log_prob)*new_log_prob).mean()
            actor_loss = surrobj + loss_entropy * self.lambda_entropy
            self.optim_a.zero_grad()
            actor_loss.backward()
            self.optim_a.step()

            self.writer.add_scalar(
                "loss_actor", actor_loss.item(), self.global_step)
            self.writer.add_scalar(
                "loss_critic", critic_loss.item(), self.global_step)
            self.global_step += 1

    def eval(self):

        self.actor.eval()

        total_reward = 0

        for _ in range(10):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                with t.no_grad():
                    action, _ = self.actor.choose_action(
                        obs, device=self.device)
                obs_, reward, done, _ = self.env.step(
                    np.tanh(action)*self.max_action)
                episode_reward += reward
                obs = obs_
            total_reward += episode_reward
        # back to train mode
        self.actor.train()
        return total_reward/10


class LocalActor():
    """
    多线程里面的actor，仅仅是用于多线程收集数据，增加效率
    """

    def __init__(self, args: dict, actor_state_dict: OrderedDict) -> None:
        self.actor = Actor(args["n_features"],
                           args["n_actions"], args["hidden_size"])
        self.buffer = BufferTuple(args["max_mem_size"])
        self.actor.load_state_dict(actor_state_dict)
        self.env = gym.make(args["env"])
        self.reward_scale = args["reward_scale"]
        self.max_action = self.env.action_space.high
        # 直接用cpu去收集数据，不需要to cuda
        self.actor.to("cpu")

    def collect_data(self) -> Transition:

        step_counter = 0
        while step_counter < self.buffer.max_size:
            done = False
            obs = self.env.reset()
            while not done:
                action, log_prob = self.actor.choose_action(obs, device="cpu")

                obs_, reward, done, _ = self.env.step(
                    self.max_action*np.tanh(action))

                inverse_done = 0. if done else 1.
                reward_ = reward*self.reward_scale

                self.buffer.push(reward_, inverse_done, obs,
                                 action, log_prob.detach().numpy())

                if done:
                    break

                obs = obs_
                step_counter += 1

        return self.buffer.sample_all()

# %%


def collect_data_async(pipe: Connection, args: dict):
    env = gym.make(args["env"])
    env.reset()

    while True:
        net_state_dict = pipe.recv()

        actor = LocalActor(args, net_state_dict)

        transition = actor.collect_data()
        r = transition.reward

        m = transition.mask
        a = transition.action
        s = transition.state
        log = transition.log_prob
        data = (r, m, s, a, log)
        """pipe不能直接传输buffer回主进程，可能是buffer内有transition，因此将数据取出来打包回传"""
        pipe.send((data))
