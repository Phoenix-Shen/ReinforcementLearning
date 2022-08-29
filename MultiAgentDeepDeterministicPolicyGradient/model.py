from tensorboardX import SummaryWriter
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from memory import ReplayBuffer, Experience


class Critic(nn.Module):
    """critic 学习Q(s,a)状态动作价值函数
    ，他是U_t的期望，在这里我们给多个agent的行为进行一次性的打分"""

    def __init__(self, n_agent: int, action_dim: int, obs_dim: int) -> None:
        super().__init__()
        # 保存参数至成员变量
        self.n_agent = n_agent
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        action_dim_total = self.action_dim * self.n_agent
        obs_dim_total = self.obs_dim * self.n_agent

        self.FC1 = nn.Linear(obs_dim_total, 1024)
        self.FC2 = nn.Linear(1024 + action_dim_total, 512)
        self.FC3 = nn.Linear(512, 256)
        # 最后只要输出一个值(value)所以最后outfeatures= 1
        self.FC4 = nn.Linear(256, 1)

    def forward(self, obs: t.Tensor, acts: t.Tensor) -> t.Tensor:
        out = F.relu(self.FC1.forward(obs))
        # 在第一维拼接，第0维是batch
        out = F.relu(self.FC2.forward(t.cat([out, acts], dim=1)))
        out = F.relu(self.FC3.forward(out))
        out = F.relu(self.FC4.forward(out))
        return out


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: t.Tensor) -> t.Tensor:
        return self.net.forward(obs)


class MADDPG(object):
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        batch_size: int,
        capacity: int,
        n_explore: int,
        n_episodes: int,
        reward_decay: float,
        tau: float,
        lr: float,
        cuda: bool,
        env: gym.Env,
        max_steps=1000,
        reward_scale=1,
        soft_update_interval=100,
        seed=100,
    ) -> None:

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.batch_size = batch_size
        self.capacity = capacity
        self.n_explore = n_explore
        self.reward_decay = reward_decay
        self.tau = tau
        self.lr = lr
        self.cuda = cuda
        self.device = "cuda" if self.cuda else "cpu"
        self.env = env
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.soft_update_interval = soft_update_interval
        self.seed = seed
        self.n_epsiodes = n_episodes
        # actor and critic
        self.actors = [
            Actor(self.obs_dim, self.action_dim) for _ in range(self.n_agents)
        ]
        self.critics = [
            Critic(self.n_agents, self.action_dim, self.obs_dim)
            for _ in range(self.n_agents)
        ]
        self.target_actors = [
            Actor(self.obs_dim, self.action_dim) for _ in range(self.n_agents)
        ]
        self.target_critics = [
            Critic(self.n_agents, self.action_dim, self.obs_dim)
            for _ in range(self.n_agents)
        ]
        # copy parameters and to cuda
        for actor, critic, target_actor, target_critic in zip(
            self.actors, self.critics, self.target_actors, self.target_critics
        ):
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

            if self.cuda:
                actor.cuda()
                critic.cuda()
                target_actor.cuda()
                target_critic.cuda()

        # optimizers
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics
        ]
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors
        ]
        # action noise
        self.var = [1.0 for i in range(self.n_agents)]
        # record the learning procedure
        self.steps_done = 0
        # memory
        self.memory = ReplayBuffer(self.capacity)

    def sample_actions(self, state_batch: t.Tensor) -> t.Tensor:
        actions = []
        # state_batch.shape = [n_agents,state_dim]
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i].forward(sb.unsqueeze(0))

            act += t.FloatTensor(np.random.randn(2) * self.var[i], device=self.device)

            # noise decay
            if self.episodes_done > self.n_expolre and self.var[i] > 0.05:
                self.var[i] *= 0.999998

            act = t.clamp(act, -1.0, 1.0)
            actions.append(act)
        # concat
        return t.cat(actions, dim=0)

    def learn():
        pass

    def collect_experience(self):
        obs = self.env.reset()
        obs = np.stack(obs)
        obs = t.FloatTensor(obs, device=self.device)

        total_reward = 0.0

        rr = np.zeros((self.n_agents,))

        for step in range(self.max_steps):
            action = self.sample_actions(obs)

            obs_, reward, done, _ = self.env.step(action.numpy())
            reward = t.FloatTensor(reward, device=self.device)

            obs_ = np.stack(obs_)
            obs_ = t.FloatTensor(obs_, device=self.device)

            if done or step == self.max_steps - 1:
                obs_ = None
            total_reward += reward.sum().numpy()
            rr += reward.cpu().numpy()
            self.memory.push(obs, action, obs_, reward)
            obs = obs_
        return total_reward, rr

    def _soft_update(self):
        for i in range(self.n_agents):
            for target_param, source_param in zip(
                self.target_actors[i], self.actors[i]
            ):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * source_param.data
                )
            for target_param, source_param in zip(
                self.target_critics[i], self.critics[i]
            ):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * source_param.data
                )

    def update_policy(self):
        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = t.ByteTensor(
                list(map(lambda s: s is not None, batch.next_states))
            )
            # state_batch.shape =[batch_size,n_agents,obs_dim]
            state_batch = t.stack(batch.states)
            action_batch = t.stack(batch.actions)
            reward_batch = t.stack(batch.rewards)
            non_final_next_states = t.stack(
                [s for s in batch.next_states if s is not None]
            )

            if self.cuda():
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                non_final_next_states = non_final_next_states.cuda()

            whole_state = state_batch.reshape((self.batch_size, -1))

            whole_action = action_batch.reshape((self.batch_size, -1))
            # update value network
            self.critic_optimizers[agent].zero_grad()

            current_Q = self.critics[agent].forward(whole_state, whole_action)

            non_final_next_actions = [
                self.target_actors[agent].forward(non_final_next_states[:, i, :])
                for i in range(self.n_agents)
            ]

            non_final_next_actions = (
                t.stack(non_final_next_actions).transpose(0, 1).contiguous()
            )

            target_Q = t.zeros(self.batch_size, device=self.device)

            target_Q[non_final_mask] = self.target_critics[agent].forward(
                non_final_next_states.reshape(-1, self.n_agents * self.obs_dim),
                non_final_next_actions.reshape(
                    -1, self.n_agents * self.action_dim
                ).squeeze(),
            )

            TD_Target = (target_Q.unsqueeze(1) * self.reward_decay) + reward_batch[
                :, agent
            ].unsqueeze(1) * self.reward_scale

            loss_Q = nn.MSELoss()(current_Q, TD_Target.detach())
            loss_Q.backward()
            self.critic_optimizers[agent].step()

            # update policy network
            self.actor_optimizers[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent].forward(state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.reshape(self.batch_size, -1)
            actor_loss = -self.critics[agent].forward(whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizers[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

            # soft update target policy and value network
            if (
                self.steps_done % self.soft_update_interval == 0
                and self.steps_done != 0
            ):
                self._soft_update()

        return a_loss, c_loss

    def learn(self):

        # set seed
        np.random.seed(self.seed)
        t.seed(self.seed)
        self.env.seed(self.seed)
        # writer
        writer = SummaryWriter(
            logdir="./MultiAgentDeepDeterministicPolicyGradient/logs"
        )
        # begin main loop
        for i_episode in range(self.n_epsiodes):
            total_reward, rr = self.collect_experience()

            if i_episode >= self.n_explore:
                for _ in range(self.max_steps):
                    loss_a, loss_c = self.update_policy()
                    self.steps_done += 1
        # record the data to a file
        writer.add_scalar("total_reward", total_reward, i_episode)
        writer.add_scalar("rr", rr, i_episode)
        writer.add_scalar("loss_a", loss_a, i_episode)
        writer.add_scalar("loss_c", loss_c, i_episode)

        print("done")

