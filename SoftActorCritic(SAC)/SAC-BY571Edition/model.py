from numpy.random import standard_exponential
from replaybuffer import ReplayBuffer
import torch as t
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as distributions


def layer_init(layer):
    """
    Get the arguments of uniorm_(from,to)
    Use uniform distribution to init data
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """policy model"""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        seed: int,
        hidden_size=32,
        init_weights=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ) -> None:
        super().__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        # save parameters to member variables
        self.seed = t.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.init_weights = init_weights
        # network
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, n_actions)
        self.sigma = nn.Linear(hidden_size, n_actions)
        # cudacuda
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.mu.weight.data.uniform_(-self.init_weights, self.init_weights)
        self.sigma.weight.data.uniform_(-self.init_weights, self.init_weights)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        forward function, the state is transferred to CUDA if you have a gpu
        """
        # use inplace operation to save cuda memory
        state = state.to(self.device)
        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = t.clamp(sigma, self.log_std_min, self.log_std_max)
        return mu, sigma

    def evaluate(self, state: Tensor, epsilon=1e-6):
        """
        called in the train procedure not action choosing procedure
        """

        """Means and covariances are not derivable so we 
        need to do some sampleing from another distibution and
        then sampled values are multiplied by the covariance and then
        the mean is added to make the network derivable
        """
        state = state.to(self.device)
        # obtain the original mu and sigma which means mean and log standard deviation
        mu, sigma = self.forward(state=state)
        # get std by exp operation
        std = sigma.exp()
        distribution = distributions.Normal(0, 1)
        e = distribution.sample().to(self.device)
        action = t.tanh(mu + e * std)
        # 但是我们的action是有界的，所以加上t.tanh()进行限制边界
        # 已知x原始的概率分布f(x),而且y=g(x)，为严格单调函数，g'存在，h(·)为g的反函数，求f(y)
        # f_a(at)=f_x(v(at))|v'(at)|即下面的f_x(mu + e * std)/(1-tanh^2(xt))
        log_prob = distributions.Normal(mu, std).log_prob(mu + e * std) - t.log(
            1 - action.pow(2) + epsilon
        )  # epsilon : prevent divide by zero
        return action, log_prob

    def get_action(self, state: Tensor):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)

        PARAMS
        -------
        state:the current state
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = distributions.Normal(0, 1)
        e = dist.sample().to(self.device)
        # use tanh to clamp the value to [-1,1]
        action = t.tanh(mu+e*std).cpu()
        return action[0]


class Critic(nn.Module):
    """
    Value Model. Critic
    """

    def __init__(self, n_features, n_actions, seed, hidden_size=32) -> None:
        super().__init__()
        self.seed = t.manual_seed(seed)
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(n_features+n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.to(self.device)

    def reset_parameters(self):
        self.net[0].weight.data.uniform_(*layer_init(self.net[0]))
        self.net[2].weight.data.uniform_(*layer_init(self.net[2]))
        self.net[4].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state, action = state.to(self.device), action.to(self.device)
        x = t.cat((state, action), dim=1)
        return self.net(x)


class Agent():
    """interacts with env and learns from the environment"""

    def __init__(self,
                 n_features,
                 n_actions,
                 random_seed=0,
                 lr_a=5e-4,
                 lr_c=5e-4,
                 weight_decay_c=0,
                 memory_size=int(1e6),
                 batch_size=256,
                 GAMMA=0.99,
                 fixed_alpha=None,
                 tau=0.01,
                 action_prior="uniform") -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.seed = random.seed(random_seed)
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.mem_size = memory_size
        self.batch_size = batch_size
        self.reward_decay = GAMMA
        self.fixed_alpha = fixed_alpha
        self.tau = tau
        # adaptive alpha
        self.target_entropy = -self.n_features  # -dim(features)
        self.alpha = 1
        self.log_alpha = t.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(
            params=[self.log_alpha], lr=self.lr_a)
        self.action_prior = action_prior
        # Actor
        self.actor = Actor(self.n_features, self.n_actions, random_seed)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.lr_a)
        # Critic( include the target network)
        self.critic1 = Critic(self.n_features, self.n_actions, random_seed)
        self.critic2 = Critic(self.n_features, self.n_actions, random_seed)
        # target network
        self.target_critic1 = Critic(
            self.n_features, self.n_actions, random_seed)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = Critic(
            self.n_features, self.n_actions, random_seed)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # define the optimizer
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.lr_c, weight_decay=weight_decay_c)
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.lr_c, weight_decay=weight_decay_c)
        # memory
        self.memory = ReplayBuffer(
            self.n_features, memory_size, batch_size, random_seed)

        print("Agent class initialized...\r\n using {}".format(self.actor.device))

    def step(self, state, action, reward, next_state, done, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn if enough samples are avaliable
        if len(self.memory) > self.batch_size:
            expericences = self.memory.sample()
            self.learn(step, expericences)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = t.from_numpy(state).float()
        action = self.actor.get_action(state).detach()
        return action

    def learn(self, step, experiences, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = -[Q(s,a)-α * log_pi(a|s)]
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # update critic
        # get predicted next_state actions and Qvalues form targe models
        next_action, log_pis_next = self.actor.evaluate(next_states)

        Q_target1_next = self.target_critic1(
            next_states.to(self.critic1.device), next_action.squeeze(0).to(self.critic1.device))
        Q_target2_next = self.target_critic1(
            next_states.to(self.critic1.device), next_action.squeeze(0).to(self.critic1.device))

        # take the minimum of both critics for updating
        Q_target_next = t.min(Q_target1_next, Q_target2_next)

        if self.fixed_alpha == None:
            # compute Q targets for current state y_i
            Q_targets = rewards.cpu()+(self.reward_decay*(1-dones.cpu()) *
                                       (Q_target_next.cpu()-self.alpha*log_pis_next.sum(1, keepdim=True).cpu()))
        else:
            Q_targets = rewards.cpu()+(self.reward_decay*(1-dones.cpu()) *
                                       (Q_target_next.cpu()-self.fixed_alpha*log_pis_next.sum(1, keepdim=True).cpu()))
        # compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # update the actor
        if step % d == 0:
            if self.fixed_alpha == None:
                alpha = t.exp(self.log_alpha)
                # compute alpha loss
                actions_pred, log_pis = self.actor.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() +
                                self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = alpha

                if self.action_prior == "normal":
                    policy_prior = distributions.MultivariateNormal(loc=t.zeros(
                        self.n_actions), scale_tril=t.ones(self.n_actions).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(
                        actions_pred)
                elif self.action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (alpha*log_pis.squeeze(0).cpu()-self.critic1(states,
                              actions_pred.squeeze(0)).cpu()-policy_prior_log_probs).mean()

            else:
                actions_pred, log_pis = self.actor.evaluate(states)
                if self.action_prior == "normal":
                    policy_prior = distributions.MultivariateNormal(loc=t.zeros(
                        self.n_actions), scale_tril=t.ones(self.n_actions).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(
                        actions_pred)
                elif self.action_prior == "uniform":
                    policy_prior_log_probs = 0.0
                actor_loss = (self.fixed_alpha*log_pis.squeeze(0).cpu()-self.critic1(states,
                              actions_pred.squeeze(0)).cpu()-policy_prior_log_probs).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update target networks
            self.soft_update(self.critic1, self.target_critic1, self.tau)
            self.soft_update(self.critic2, self.target_critic2, self.tau)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
