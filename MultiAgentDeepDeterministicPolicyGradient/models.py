import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch import Tensor
from tensorboardX import SummaryWriter
import gym
import tqdm
import numpy as np
from memory import MemoryBuffer
import os
import datetime


class Actor(nn.Module):
    def __init__(
        self,
        action_high: float,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()

        self.action_high = action_high
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: Tensor) -> Tensor:

        out = self.net.forward(obs)

        actions = self.action_high * out
        return actions

    def select_action(
        self, obs: Tensor, noise_rate: float, epsilon: float
    ) -> np.ndarray:
        # random exploration
        if np.random.uniform() < epsilon:
            mu = np.random.uniform(-self.action_high,
                                   self.action_high, self.action_dim)

        else:

            inputs = t.tensor(
                obs, dtype=t.float32, device=next(self.parameters()).device
            ).unsqueeze(0)
            pi = self.forward(inputs).squeeze()
            # deterministic policy
            mu = pi.cpu().numpy()
            # add gaussian noise
            noise = noise_rate * self.action_high * np.random.randn(*mu.shape)
            mu += noise
            mu = np.clip(mu, -self.action_high, self.action_high)
        return mu


class Critic(nn.Module):
    def __init__(
        self,
        action_high: float,
        obs_dims: list[int],
        action_dims: list[int],
    ) -> None:
        super().__init__()
        self.action_high = action_high
        # critic should give scores for all agents' actions
        self.net = nn.Sequential(
            nn.Linear(sum(obs_dims) + sum(action_dims), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        obs = t.cat(obs, dim=1)
        # denorm and concatenate  dim = 1 means we concatenate along the feature axis
        for i in range(len(actions)):
            actions[i] = actions[i] / self.action_high
        actions = t.cat(actions, dim=1)

        # to the certain device
        param_device = next(self.parameters()).device
        if obs.device != param_device:
            obs = obs.to(param_device)
            actions = actions.to(param_device)

        q_value = self.net.forward(t.cat([obs, actions], dim=1))
        return q_value


class MADDPG(object):
    def __init__(
        self,
        noise_rate,
        epsilon,
        episode_limit,
        env: gym.Env,
        n_agents,
        n_players,
        save_dir,
        action_dims,
        obs_dims,
        action_high,
        lr_a,
        lr_c,
        log_dir,
        time_steps,
        mem_capacity,
        batch_size,
        gamma,
        tau,
        eval_interval,
        eval_episodes,
        eval_episodes_len,
        cuda: bool,
    ):
        # save parameters to member variables
        self.time_steps = time_steps
        self.episode_limit = episode_limit
        self.env = env
        self.save_dir = save_dir
        self.epsilon = epsilon
        self.noise_rate = noise_rate
        self.n_agents = n_agents
        self.n_players = n_players
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_episodes_len = eval_episodes_len
        # init actors ,critics and optimizers
        self.actors = list[Actor]()
        self.critics = list[Critic]()
        self.target_actors = list[Actor]()
        self.target_critics = list[Critic]()
        self.optimizer_a = list[optim.Optimizer]()
        self.optimizer_c = list[optim.Optimizer]()
        self.device = "cuda" if cuda else "cpu"
        for i in range(n_agents):
            self.actors.append(Actor(action_high, obs_dims[i], action_dims[i]))
            self.critics.append(Critic(action_high, obs_dims, action_dims))
            self.target_actors.append(
                Actor(action_high, obs_dims[i], action_dims[i]))
            self.target_critics.append(
                Critic(action_high, obs_dims, action_dims))
            # load_state_dict
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(
                self.critics[i].state_dict())
            # optimizers
            self.optimizer_a.append(optim.Adam(
                self.actors[i].parameters(), lr=lr_a))
            self.optimizer_c.append(optim.Adam(
                self.critics[i].parameters(), lr=lr_c))

            self.actors[i] = self.actors[i].to(self.device)
            self.critics[i] = self.critics[i].to(self.device)
            self.target_actors[i] = self.target_actors[i].to(self.device)
            self.target_critics[i] = self.target_critics[i].to(self.device)

        self.buffer = MemoryBuffer(
            mem_capacity, obs_dims, action_dims, self.n_agents)
        self.writer = SummaryWriter(log_dir=log_dir)

    def learn(self):

        for time_step in tqdm.tqdm(range(self.time_steps)):
            if time_step % self.episode_limit == 0:
                s = self.env.reset()

            mu = []
            actions = []

            with t.no_grad():
                for index, actor in enumerate(self.actors):
                    action = actor.select_action(
                        s[index], self.noise_rate, self.epsilon
                    )
                    mu.append(action)
                    actions.append(action)

            for _ in range(self.n_agents, self.n_players):
                actions.append(
                    [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
                )

            s_next, r, done, _ = self.env.step(actions)
            self.buffer.store_transition(
                s[: self.n_agents], mu, r[: self.n_agents], s_next[: self.n_agents]
            )
            s = s_next

            if self.buffer.current_size > self.batch_size:
                transitions = self.buffer.sample(self.batch_size)
                # call update_policy function
                actor_losses, critic_losses = self._update_policy(transitions)
                for index, (actor_loss, critic_loss) in enumerate(
                    zip(actor_losses, critic_losses)
                ):
                    self.writer.add_scalar(
                        "actor_loss_{}".format(index), actor_loss, time_step
                    )
                    self.writer.add_scalar(
                        "critic_loss_{}".format(index), critic_loss, time_step
                    )

            # update noise and epsilon
            self.noise_rate = max(0.05, self.noise_rate - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)

            # call eval function
            if time_step != 0 and time_step % self.eval_interval == 0:
                mean_reward = self._evaluate()
                self.writer.add_scalar("eval_reward", mean_reward, time_step)

        self._save_model()
        print("training finished and the model has been saved")

    def _update_policy(self, transitions: dict):
        # to tensor
        for key in transitions.keys():
            transitions[key] = t.tensor(transitions[key], dtype=t.float32)
        # variables to store actor loss and cirtic loss
        actor_losses, critic_losses = [], []
        # train each agent
        for i in range(self.n_agents):

            r = transitions["r_%d" % i].to(self.device)
            o, mu, o_next = [], [], []
            for j in range(self.n_agents):
                o.append(transitions["o_%d" % j].to(self.device))
                mu.append(transitions["a_%d" % j].to(self.device))
                o_next.append(transitions["o_next_%d" % j].to(self.device))
            # calculate Q-Target
            mu_next = []
            with t.no_grad():
                # get next actions
                for j in range(self.n_agents):
                    mu_next.append(self.target_actors[j].forward(o[j]))

                # get Q-Value of next observations
                # no need to detach the tensor because we already used t.no_grad()
                q_next = self.target_critics[i].forward(o_next, mu_next)

                q_target = r.unsqueeze(1) + self.gamma * q_next

            # comput td target and use the square of td residual as the loss
            q_value = self.critics[i].forward(o, mu)
            critic_loss = t.mean((q_target - q_value) *(q_target - q_value))

            # actor loss, Actor's goal is to make Critic's scoring higher
            mu[i] = self.actors[i].forward(o[i])
            actor_loss = -self.critics[i].forward(o, mu).mean()

            # then perform gradient descent
            self.optimizer_a[i].zero_grad()
            self.optimizer_c[i].zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.optimizer_a[i].step()
            self.optimizer_c[i].step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        # then soft update the target network
        self._soft_update_target()

        return actor_losses, critic_losses

    def _soft_update_target(self) -> None:
        for i in range(self.n_agents):
            for target_param, param in zip(
                self.target_actors[i].parameters(), self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

            for target_param, param in zip(
                self.target_critics[i].parameters(
                ), self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

    def _evaluate(self) -> float:
        returns = []
        for _ in range(self.eval_episodes):
            s = self.env.reset()
            rewards = 0
            for _ in range(self.eval_episodes_len):
                actions = []
                with t.no_grad():
                    for index, actor in enumerate(self.actors):
                        action = actor.select_action(s[index], 0, 0)
                        actions.append(action)

                    for _ in range(self.n_agents, self.n_players):
                        actions.append(
                            [
                                0,
                                np.random.rand() * 2 - 1,
                                0,
                                np.random.rand() * 2 - 1,
                                0,
                            ]
                        )
                    s_next, r, _, _ = self.env.step(actions)
                    rewards += r[0]
                    s = s_next
            returns.append(rewards)

        return sum(returns) / self.eval_episodes

    def _save_model(self):

        check_point = {}

        for index, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            check_point["actor_%d" % index] = actor
            check_point["critic_%d" % index] = critic

        t.save(
            check_point,
            os.path.join(
                self.save_dir,
                "ckpt_{}.pth".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                ),
            ),
        )
