import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
import numpy as np
import scipy.signal
import gym
import os
import datetime


class Actor(nn.Module):
    """
    Discrete Actor, it takes the states as an input
    and outputs the probabilities of each action
    """

    def __init__(self, n_states: int, n_actions: int, n_hiddens=256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        forward procedure of the actor
        """
        return self.net.forward(x)

    def act(self, x: Tensor):
        """
        sample an action from the given probability distribution
        -------
        Returns:
            the sampled action and the probability ofthe action
        """
        action_prob = self.forward(x)
        distribution = Categorical(action_prob)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)

        return action.detach(), action_logprob.detach()

    def act_all_probs(self, x: Tensor):
        action_prob = self.forward(x)
        distribution = Categorical(action_prob)
        action = distribution.sample()
        return action.detach().item(), action_prob.cpu().detach().numpy()


class Critic(nn.Module):
    """
    Critic Network, it takes the states as an input,
    and outputs a scalar which indicates the value of the state
    """

    def __init__(self, n_states: int, n_hiddens=256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        forward procedure of the critic
        """
        return self.net.forward(x)


class PPOClip():
    def __init__(self, args: dict):
        # select the device
        self.device = t.device(
            "cuda") if args["device"] == "cuda" and t.cuda.is_available() else "cpu"
        t.cuda.empty_cache()
        # save arguments to member variables
        self.state_dim = args["state_dim"]
        self.action_dim = args["action_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.lr_actor = args["lr_actor"]
        self.lr_critic = args["lr_critic"]
        self.eps_clip = args["eps_clip"]
        self.buffer_size = args["buffer_size"]
        self.entropy_weight = args["entropy_weight"]
        self.epochs = args["epochs"]
        self.batch_mode = args["batch_mode"]
        self.batch_size = args["batch_size"]
        self.gamma = args["gamma"]
        self.lamb = args["lambda"]
        self.training_epochs = args["training_epochs"]
        self.max_episode_length = args["max_episode_length"]
        self.update_interval = args["update_interval"]
        now = datetime.datetime.now()
        folder_name = now.strftime("%Y-%m-%d %H-%M-%S")
        self.log_path = os.path.join(args["log_path"], folder_name, "logs")
        self.save_path = os.path.join(
            args["log_path"], folder_name, "saved_models")
        # make dir
        os.makedirs(self.log_path)
        os.makedirs(self.save_path)
        self.actor_path = args["actor_path"]
        self.critic_path = args["critic_path"]
        # define actor critic networks
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.hidden_dim).to(self.device)
        self.old_actor = Actor(
            self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.old_critic = Critic(
            self.state_dim, self.hidden_dim).to(self.device)
        # load the pretrained models
        if self.actor_path is not None and self.critic_path is not None:
            self.load_parameters()
        # copy parameters
        self.update_old_net()
        # optimizer for the actor and critic networks
        self.optimizer = t.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic}
            ]
        )
        # loss function for critics
        self.mseloss = nn.MSELoss()
        # rollout buffer of PPO
        self.buffer = {
            "rewards": np.zeros(self.buffer_size, dtype=np.float32),
            "dones": np.zeros(self.buffer_size, dtype=np.float32),
            "states": np.zeros((self.buffer_size, self.state_dim), dtype=np.float32),
            "actions": np.zeros(self.buffer_size, dtype=np.float32),
            "log_probs": np.zeros(self.buffer_size, dtype=np.float32),
            "values": np.zeros(self.buffer_size, dtype=np.float32),
        }
        # buffers of the derived values
        # advantage
        self.adv_buffer = np.zeros(
            self.buffer_size, dtype=np.float32)

        # discounted reward aka. return
        self.ret_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # buffer of the state variables
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.buffer_size

    def store_transition(self, reward: float, done: float, state: np.ndarray, action: float, log_prob: float, value: float):
        """
        store a transition in the buffer
        """
        assert self.ptr < self.max_size
        self.buffer["rewards"][self.ptr] = reward
        self.buffer["dones"][self.ptr] = done
        self.buffer["states"][self.ptr] = state
        self.buffer["actions"][self.ptr] = action
        self.buffer["log_probs"][self.ptr] = log_prob
        self.buffer["values"][self.ptr] = value
        # increase the pointer
        self.ptr += 1

    def finish_path(self, last_val: float = 0):
        """
        Calculate GAE(General Advantage Estimates) and discounted rewards when the game is finished.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        vals = np.append(self.buffer["values"][path_slice], last_val)
        rews = np.append(self.buffer["rewards"][path_slice], last_val)
        deltas = rews[:-1] + self.gamma*vals[1:]-vals[:-1]
        self.adv_buffer[path_slice] = self._discount_cumsum(
            deltas, self.gamma*self.lamb)
        self.ret_buffer[path_slice] = self._discount_cumsum(rews, self.gamma)[
            :-1]
        self.path_start_idx = self.ptr

    @staticmethod
    def _discount_cumsum(x, discount: float):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        """

        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def get_data(self):
        """
        sample all data from the buffers with the estimated advantage and cumulative discounted rewards
        -------
        Returns:
            the data from the buffers with GAE and returns
        """
        data = {k: self.buffer[k][:self.ptr] for k in self.buffer}
        data["advantages"] = self.adv_buffer[:self.ptr]
        data["discounted_rewards"] = self.ret_buffer[:self.ptr]
        return data

    def clean_buffer(self):
        """
        clear the buffer of past samples
        """
        # just modify the pointer and start index
        self.ptr = 0
        self.path_start_idx = 0

    def evaluate(self, state: Tensor, action: Tensor):
        """
        evaluate the (state,action) pair and record logprobabilities from actor, state values from critic
        and the entropy of the probability distribution from the actor.
        -------
        Parameters:
            state: Tensor
            action: Tensor
        Returns:
            action_log_prob: Tensor
            state_values: Tensor
            entropy: Tensor

        """
        action_probs = self.actor.forward(state)
        distribution = Categorical(action_probs)
        action_log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        state_values = self.critic.forward(state)

        return action_log_prob, state_values, entropy

    def generate_batches(self, discounted_rewards: np.ndarray, batch_size: int) -> np.ndarray:
        """
        generate many batches of samples from the buffer,
        note that the last batch's length may be smaller than batch_size
        """
        one_left = len(discounted_rewards) % batch_size == 1
        n_states = len(discounted_rewards)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        batches = [indices[i:i+batch_size]
                   for i in batch_start if len(indices[i:i+batch_size]) > 1]
        if one_left:
            batches[-1] = np.concatenate((batches[-1], indices[-1:]))
        return batches

    def update(self):
        """
        update the weights of the actor and critic using PPOClip
        -------
        Parameters:
            epochs: int, how many gradient descent epochs do you want to perform?
            batch_mode: bool, set True if you want to activate batch mode, False otherwise
            batch_size: int
        Returns:
            eploss_a: float, the mean loss of the actor in the update procedure
            eploss_c: float, the mean loss of the critic in the update procedure
        """
        # generate batch data
        data = self.get_data()
        discounted_rewards = data["discounted_rewards"]
        batch_size = self.batch_size if self.batch_mode else len(
            discounted_rewards)
        batches = self.generate_batches(discounted_rewards, batch_size)

        # start training procedure
        ep_lossa, ep_lossc = [], []
        for ep in range(self.epochs):
            for batch in batches:
                # use from numpy instead of torch.tensor() to accelerate
                discounted_rewards = t.from_numpy(
                    data["discounted_rewards"][batch]).to(self.device)
                # normalize the advantages
                advantages = t.from_numpy(
                    data["advantages"][batch]).to(self.device)
                advantages = (advantages-advantages.mean()) / \
                    (advantages.std()+1e-10)

                # convert ndarray to tensor
                old_states = t.from_numpy(
                    data["states"][batch]).to(self.device)
                old_actions = t.from_numpy(
                    data["actions"][batch]).to(self.device)
                old_log_probs = t.from_numpy(
                    data["log_probs"][batch]).to(self.device)

                # get the value from the new networks, we only use log_probs and state_values
                # as part of the computation graph to update the weights of actor and critic
                log_probs, state_values, dist_entropy = self.evaluate(
                    old_states, old_actions)
                state_values = state_values.squeeze()

                # calculate the surrogate loss
                ratios = t.exp(log_probs-old_log_probs.detach())
                surr1 = ratios*advantages
                surr2 = t.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip)*advantages
                # calculate the actor loss
                loss_actor = t.mean(-t.min(surr1, surr2) -
                                    self.entropy_weight*dist_entropy)
                # calculate the cirtic loss
                loss_critic = self.mseloss.forward(
                    state_values, discounted_rewards)
                loss = loss_actor + loss_critic
                # zero_grad and step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # record the loss
                ep_lossa.append(loss_actor.item())
                ep_lossc.append(loss_critic.item())
        # copy parameters
        self.update_old_net()
        # clean the buffer
        self.clean_buffer()
        return sum(ep_lossa)/len(ep_lossa), sum(ep_lossc)/len(ep_lossc)

    def learn(self, env: gym.Env):
        """
        start to train the actor critic
        """
        time_steps = 0
        current_episode_reward = 0
        while time_steps < self.training_epochs:
            print(
                f"{time_steps}/{self.training_epochs},reward:{current_episode_reward}")

            state = env.reset()
            current_episode_reward = 0

            for ts in range(1, self.max_episode_length+1):
                # without gradient
                with t.no_grad():
                    state = t.from_numpy(state).to(self.device)
                    # get action and log probability
                    action, log_prob = self.old_actor.act(state)
                    # get critic evaluation of the state
                    value = self.old_critic.forward(state)
                # take a step in the environment
                next_state, reward, done, _ = env.step(action.item())

                # record loss and add timestep
                current_episode_reward += reward
                time_steps += 1

                timeout = (ts == self.max_episode_length)
                update = time_steps % self.update_interval == 0
                terminal = done or timeout or update
                # store the transition
                self.store_transition(
                    reward, done, state, action, log_prob, value)

                if terminal:
                    # if not done, calculate the value of next state
                    if not done:
                        with t.no_grad():
                            value = self.old_critic(
                                t.from_numpy(next_state).to(self.device))
                    else:
                        value = 0
                    # calculate the advantages and returns
                    self.finish_path(value)
                    # if update interval is reached, update the parameters
                    if update:
                        loss_a, loss_c = self.update()
                        self.save_parameters(time_steps)
                        with open(os.path.join(self.log_path, "log.txt"), "a") as f:
                            f.write(
                                f"{time_steps},{loss_a},{loss_c}\n")
                    if done:
                        break
                state = next_state

    def update_old_net(self):
        """
        copy the parameters of the new network to the old network
        """
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

    def save_parameters(self, ep):
        t.save(self.old_actor.state_dict(),
               os.path.join(self.save_path, f"actor_{ep}.pth"))
        t.save(self.old_critic.state_dict(),
               os.path.join(self.save_path, f"critic_{ep}.pth"))

    def load_parameters(self):
        self.actor.load_state_dict(t.load(self.actor_path))
        self.critic.load_state_dict(t.load(self.critic_path))
        self.update_old_net()

    def test(self, env: gym.Env):
        for ep in range(1,11):
            state = env.reset()
            current_episode_reward = 0
            env.render()
            
            for ts in range(1, self.max_episode_length+1):
                with t.no_grad():
                    state = t.from_numpy(state).to(self.device)
                    action, log_prob = self.old_actor.act(state)
                next_state, reward, done, _ = env.step(action.item())
                state = next_state
                done = True if ts == self.max_episode_length else done
                env.render()
                current_episode_reward += reward
                if done:
                    break
            print(f"ep:{ep} reward:{current_episode_reward}")
