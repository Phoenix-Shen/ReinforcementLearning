import torch as t
import torch.nn as nn
import replayMemory
import torch.optim as optim
import gym
import numpy as np
import tensorboardX


class ActorCritic(nn.Module):
    def __init__(self,
                 n_features,
                 n_acitons) -> None:
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Linear(256, 256))
        # 学习策略Pinn.Sequential(
        self.action_layer = nn.Sequential(
            nn.Linear(256, n_acitons), nn.Softmax(dim=-1))
        # 学习Q_pi(a,s)动作-状态价值函数
        self.value_layer = nn.Linear(256, n_acitons)

    def forward(self, state: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        features = self.feature_net(state)
        actions = self.action_layer(features)
        values = self.value_layer(features)
        return actions, values

    def copy_parameters_from(self, source: nn.Module, decay=0.):
        """
        Copy the parameters from another network.

        Parameters
        ----------
        source : ActorCritic
            The network from which to copy the parameters.
        decay : float, optional
            How much decay should be applied? Default is 0., which means the parameters
            are completely copied.
        """
        for parameter, source_parameter in zip(self.parameters(), source.parameters()):
            parameter.data.copy_(decay*parameter.data +
                                 (1-decay)*source_parameter.data)

    def copy_gradients_from(self, source: nn.Module):
        """
        Copy the gradients from another network.

        Parameters
        ----------
        source : ActorCritic
            The network from which to copy the gradients.
        """
        for parameter, source_parameter in zip(self.parameters(), source.parameters()):
            parameter._grad = source_parameter.grad


class Agent():
    def __init__(self,
                 n_features,
                 n_actions,
                 buffer_size,
                 sample_batch_size,
                 lr,
                 max_episode,
                 env_name="LunarLander-v2",
                 render=False,
                 truncation_param=10,
                 gamma=0.99,
                 entropy_regularization=1e-3,
                 trust_region_decay=0.99,
                 trust_region_constraint=1.,
                 replay_ratio=4) -> None:
        # save parameters to member variables
        self.n_features = n_features
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.sample_batch_size = sample_batch_size
        self.lr = lr
        self.max_episode = max_episode
        self.env = gym.make(env_name)
        self.render = render
        self.gamma = gamma
        self.truncation_param = truncation_param
        self.entropy_regularization = entropy_regularization
        self.trust_region_decay = trust_region_decay
        self.trust_region_constraint = trust_region_constraint
        self.replay_ratio = replay_ratio
        self.writer = tensorboardX.SummaryWriter(
            "ActorCriticwithExperienceReplay(ACER)/logs")
        # neural network contains an actor critic network and an "average" actor critic network
        self.actor_critic = ActorCritic(self.n_features, self.n_actions)
        self.average_actor_critic = ActorCritic(
            self.n_features, self.n_actions)
        # we need share memory in order to learn asynchronizely
        self.actor_critic.share_memory()
        self.average_actor_critic.share_memory()
        self.average_actor_critic.copy_parameters_from(self.actor_critic)
        # replay buffer is also needed in this algorithm
        self.buffer = replayMemory.ReplayBuffer(replay_size=buffer_size)
        # definition of optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)

    def run(self):
        """
        Run the agent for several episodes
        """

        # step 1 Repeat
        for episode in range(self.max_episode):
            episode_rewards = 0.
            # step 2 Call ACER on-policy ,Algorithm 2
            trajectory = self.explore()
            la, lc = self.learning_iteration(trajectory)
            episode_rewards += sum([transition.rewards[0, 0]
                                   for transition in trajectory])
            self.writer.add_scalar("loss_actor", la, episode)
            self.writer.add_scalar("loss_critic", lc, episode)
            self.writer.add_scalar("ep_rewards", episode_rewards, episode)
            print("episode:{},loss_a:{},loss_c:{}".format(episode, la, lc))
            # step 3  n = Possion(r)
            # step 4  for i∈{1,..n} do
            for _ in range(np.random.poisson(self.replay_ratio)):
                trajectory = self.buffer.sample(self.sample_batch_size)
                if trajectory:
                    # step 5 call ACER off-policy, Algorithm 2
                    self.learning_iteration(trajectory)
            # step 6 end for
        self.writer.close()
        # step 7 uitil Max iteration or time reached

    def explore(self):
        """
        Explore an environment by taking a sequence of actions and saving the results in the memory.

        Parameters
        ----------
        None

        Returns
        ----------
        A list of trajectory
        """

        ########################################
        # Algorithm 2 ACER for discrete actions#
        ########################################
        # if On-Policy then:
        #   get state x0

        # shape = (8,) for lunar lander
        state = t.FloatTensor(self.env.reset())
        trajectory = []
        done = False
        # for i ∈ {0,...,k} do
        # perform ai according to f(·|phi_theta_alpha (xi))
        # receive reward ri and new state xi+1
        # u(·|xi)=f(·|phi_theta_alpha (xi))
        while not done:
            action_probabilities, _ = self.actor_critic.forward(
                state.unsqueeze(0))
            # 根据概率进行随机采样
            action = t.multinomial(action_probabilities, 1)
            # shape= (1,1)
            action = action.detach()
            exploration_statistics = action_probabilities.detach()
            # action必定是一个元素，所以调用item
            next_state, reward, done, _ = self.env.step(action.item())
            # shape = (8,) for lunar lander
            next_state = t.FloatTensor(next_state)
            # shape = (1,1)
            reward = t.tensor(reward, dtype=t.float32).unsqueeze(
                0).unsqueeze(0)
            # shape = (1,1)
            done = t.FloatTensor([[done]])
            if self.render:
                self.env.render()
            transition = replayMemory.Transition(state.unsqueeze(0),
                                                 action,
                                                 reward,
                                                 next_state.unsqueeze(0),
                                                 done,
                                                 exploration_statistics
                                                 )
            self.buffer.add(transition)
            trajectory.append(transition)

            state = next_state
        return trajectory

    def learning_iteration(self, trajectory: list[replayMemory.Transition]):
        """
        Conduct a single discrete learning iteration. Analogue of Algorithm 2 in the paper.
        """
        actor_critic = ActorCritic(self.n_features, self.n_actions)
        actor_critic.copy_parameters_from(self.actor_critic)

        _, _, _, next_states, _, _ = trajectory[-1]
        action_probabilities, action_values = actor_critic(next_states)
        retrace_action_value = (action_probabilities *
                                action_values).detach().sum(-1).unsqueeze(-1)
        # we have done the calculations before:
        # for i ∈  {0,...k} do
        # compute f(.|phi_theta'(xi)),Q_theta'_v(xi,.) and f(.|phi_theta_a(xi))

        # Q_ret = {0 if terminal xk else sigema(Q(xk,a)f(a|phi(xk)))}
        # for i ∈ {k-1,...0} do
        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = actor_critic(next_states)
            average_action_probabilities, _ = self.average_actor_critic(states)
            # vi = sigema (Q(xi,a)f(a|phi(xi)))
            value = (action_probabilities *
                     action_values).detach().sum(-1).unsqueeze(-1)*(1.-done)
            actions_indices = t.LongTensor(actions)
            # calculate p_i=min{1,f(ai|phi(xi))/mu(ai|xi)} we will do clamp optim later
            importrance_weights = action_probabilities.detach()/exploration_probabilities
            # naive_advantage ：how good this action is ,compared with average situation
            naive_advantage = action_values.gather(
                -1, actions_indices).detach()-value
            # Qret = ri + gamma * Qret, where gamma equals 0.99 in our Lunar Lander env
            retrace_action_value = rewards+self.gamma * \
                retrace_action_value*(1.-done)
            retrace_advantage = retrace_action_value-value

            # Actor:computing quatities needed for trust region updating:
            # g= min{c,p_i(a_i)}log(f(ai|phi_theta'(xi)))(Qret-Vi)+ bias_correction
            # bias_correction =sigema((1-c/p_i(a))f(a|phi_theta'(xi))*log f(a|phi_theta'(xi))*naive_advantage)
            actor_loss = - t.gather(importrance_weights, -1, actions_indices.detach()).clamp(
                max=self.truncation_param)*retrace_advantage*t.gather(action_probabilities, -1, actions_indices).log()
            bias_correction = (- (1-self.truncation_param/importrance_weights).clamp(
                min=0.)*naive_advantage*action_probabilities.detach()*action_probabilities.log()).sum(-1).unsqueeze(-1)
            actor_loss = actor_loss+bias_correction
            # we have the loss , then compute the gardients
            # So, Accumulate gradients wrt theta' using kl divergance
            # 使用 KL 散度来进行限制。避免像 TRPO 对 KL 关于 theta 求 Hessian，在这里使用了 average policy network
            actor_gradients = t.autograd.grad(
                actor_loss.mean(), action_probabilities, retain_graph=True)
            actor_gradients = self.dicrete_trust_region_update(
                actor_gradients, action_probabilities, average_action_probabilities)
            action_probabilities.backward(actor_gradients, retain_graph=True)

            # Critic: loss equeals (Qret-Qtheta'v(xi,a))^2
            critic_loss = t.pow(
                (t.gather(action_values, -1, actions_indices)-retrace_action_value), 2)
            critic_loss.mean().backward(retain_graph=True)

            # Entropy loss
            entropy_loss = self.entropy_regularization * \
                t.sum((action_probabilities*action_probabilities.log()), dim=-1)
            entropy_loss.mean().backward(retain_graph=True)

            # update retrace Q value
            retrace_action_value = t.gather(importrance_weights, -1, actions_indices.detach()).clamp(
                max=1.)*(retrace_action_value-t.gather(action_values, -1, actions_indices).detach())+value

            self.actor_critic.copy_gradients_from(actor_critic)
            self.optimizer.step()
            # we need softly update the average parameter
            self.average_actor_critic.copy_parameters_from(
                self.actor_critic, decay=self.trust_region_decay)

            return actor_loss.mean(), critic_loss.mean()

    def dicrete_trust_region_update(self, actor_gradients: t.Tensor, action_probabilities: t.Tensor, average_action_probabilities: t.Tensor):
        """
        Update the actor gradients so that they satisfy a linearized KL constraint with respect
        to the average actor-critic network. See Section 3.3 of the paper for details.

        Parameters
        ----------
        actor_gradients : tuple of torch.Tensor's
            The original gradients.
        action_probabilities
            The action probabilities according to the current actor-critic network.
        average_action_probabilities
            The action probabilities according to the average actor-critic network.

        Returns
        -------
        tuple of torch.Tensor's
            The updated gradients.
        """

        # first, we solve the following optimization problem with a linearized KL divergance constraint:
        # minimize (z)   0.5||gt-z||
        # subject to  grad(pi_theta(xt))D_KL[∫(.|phi_theta_avg(xt))||∫(.|phi_theta(xt))].T z<=delta
        # the solution is z* = gt - max{0, (k.T*gt-delta)/||k||2}k
        negative_kullback_leibler = - ((average_action_probabilities.log(
        )-action_probabilities.log())*average_action_probabilities).sum(-1)
        kullback_leibler_gradients = t.autograd.grad(
            negative_kullback_leibler.mean(), action_probabilities, retain_graph=True)

        updated_actor_gradients = []
        for actor_gradient, kullback_leibler_gradient in zip(actor_gradients, kullback_leibler_gradients):
            scale = actor_gradient.mul(
                kullback_leibler_gradient).sum(-1).unsqueeze(-1)-self.trust_region_constraint
            scale = t.div(scale, actor_gradient.mul(
                actor_gradient).sum(-1).unsqueeze(-1)).clamp(min=0.)
            updated_actor_gradients.append(
                actor_gradient-scale*kullback_leibler_gradient)

        return updated_actor_gradients
