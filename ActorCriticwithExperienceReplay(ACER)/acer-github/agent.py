import torch
import numpy as np
import replay_memory
from torch.autograd import Variable
from brain import DiscreteActorCritic, ContinuousActorCritic
from ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
from core import *


class Agent:
    """
    Agent that learns an optimal policy using ACER.

    Parameters
    ----------
    brain : brain.Brain
        The brain to update.
    render : boolean, optional
        Should the agent render its actions in the on-policy phase?
    verbose : boolean, optional
        Should the agent print progress to the console?
    """

    def __init__(self, brain, render=False, verbose=False):
        self.env = gym.make(ENVIRONMENT_NAME)
        self.env.reset()
        self.render = render
        self.verbose = verbose
        self.buffer = replay_memory.ReplayBuffer()
        self.brain = brain
        self.optimizer = torch.optim.Adam(brain.actor_critic.parameters(),
                                          lr=LEARNING_RATE)


class DiscreteAgent(Agent):
    def __init__(self, brain, render=True, verbose=True):
        super().__init__(brain, render, verbose)

    def run(self):
        """
        Run the agent for several episodes.
        """
        for episode in range(MAX_EPISODES):
            episode_rewards = 0.
            end_of_episode = False
            if self.verbose:
                print("Episode #{}".format(episode), end="")
            while not end_of_episode:
                trajectory = self.explore(self.brain.actor_critic)
                self.learning_iteration(trajectory)
                end_of_episode = trajectory[-1].done[0, 0]
                episode_rewards += sum([transition.rewards[0, 0]
                                       for transition in trajectory])
                for trajectory_count in range(np.random.poisson(REPLAY_RATIO)):
                    trajectory = self.buffer.sample(
                        OFF_POLICY_MINIBATCH_SIZE, MAX_REPLAY_SIZE)
                    if trajectory:
                        self.learning_iteration(trajectory)
            if self.verbose:
                print(", episode rewards {}".format(episode_rewards))

    def learning_iteration(self, trajectory):
        """
        Conduct a single discrete learning iteration. Analogue of Algorithm 2 in the paper.
        """
        actor_critic = DiscreteActorCritic()
        actor_critic.copy_parameters_from(self.brain.actor_critic)

        _, _, _, next_states, _, _ = trajectory[-1]
        action_probabilities, action_values = actor_critic(
            Variable(next_states))
        retrace_action_value = (action_probabilities *
                                action_values).data.sum(-1).unsqueeze(-1)

        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = actor_critic(
                Variable(states))
            average_action_probabilities, _ = self.brain.average_actor_critic(
                Variable(states))
            value = (action_probabilities *
                     action_values).data.sum(-1).unsqueeze(-1) * (1. - done)
            action_indices = Variable(actions.long())

            importance_weights = action_probabilities.data / exploration_probabilities

            naive_advantage = action_values.gather(-1,
                                                   action_indices).data - value
            retrace_action_value = rewards + DISCOUNT_FACTOR * \
                retrace_action_value * (1. - done)
            retrace_advantage = retrace_action_value - value

            # Actor
            actor_loss = - ACTOR_LOSS_WEIGHT * Variable(
                importance_weights.gather(-1, action_indices.data).clamp(max=TRUNCATION_PARAMETER) * retrace_advantage) \
                * action_probabilities.gather(-1, action_indices).log()
            bias_correction = - ACTOR_LOSS_WEIGHT * Variable((1 - TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) *
                                                             naive_advantage * action_probabilities.data) * action_probabilities.log()
            actor_loss += bias_correction.sum(-1).unsqueeze(-1)
            actor_gradients = torch.autograd.grad(
                actor_loss.mean(), action_probabilities, retain_graph=True)
            actor_gradients = self.discrete_trust_region_update(actor_gradients, action_probabilities,
                                                                Variable(average_action_probabilities.data))
            action_probabilities.backward(actor_gradients, retain_graph=True)

            # Critic
            critic_loss = (action_values.gather(-1, action_indices) -
                           Variable(retrace_action_value)).pow(2)
            critic_loss.mean().backward(retain_graph=True)

            # Entropy
            entropy_loss = ENTROPY_REGULARIZATION * \
                (action_probabilities * action_probabilities.log()).sum(-1)
            entropy_loss.mean().backward(retain_graph=True)

            retrace_action_value = importance_weights.gather(-1, action_indices.data).clamp(max=1.) * \
                (retrace_action_value -
                 action_values.gather(-1, action_indices).data) + value
        self.brain.actor_critic.copy_gradients_from(actor_critic)
        self.optimizer.step()
        self.brain.average_actor_critic.copy_parameters_from(
            self.brain.actor_critic, decay=TRUST_REGION_DECAY)

    def explore(self, actor_critic):
        """
        Explore an environment by taking a sequence of actions and saving the results in the memory.

        Parameters
        ----------
        actor_critic : ActorCritic
            The actor-critic model to use to explore.
        """
        state = torch.FloatTensor(self.env.env.state)
        trajectory = []
        for step in range(MAX_STEPS_BEFORE_UPDATE):
            action_probabilities, *_ = actor_critic(Variable(state))
            action = action_probabilities.multinomial(1)
            action = action.data
            exploration_statistics = action_probabilities.data.view(1, -1)
            next_state, reward, done, _ = self.env.step(action.numpy()[0])
            next_state = torch.from_numpy(next_state).float()
            if self.render:
                self.env.render()
            transition = replay_memory.Transition(states=state.view(1, -1),
                                                  actions=action.view(1, -1),
                                                  rewards=torch.FloatTensor(
                                                      [[reward]]),
                                                  next_states=next_state.view(
                                                      1, -1),
                                                  done=torch.FloatTensor(
                                                      [[done]]),
                                                  exploration_statistics=exploration_statistics)
            self.buffer.add(transition)
            trajectory.append(transition)
            if done:
                self.env.reset()
                break
            else:
                state = next_state
        return trajectory

    @staticmethod
    def discrete_trust_region_update(actor_gradients, action_probabilities, average_action_probabilities):
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
        negative_kullback_leibler = - ((average_action_probabilities.log() - action_probabilities.log())
                                       * average_action_probabilities).sum(-1)
        kullback_leibler_gradients = torch.autograd.grad(negative_kullback_leibler.mean(),
                                                         action_probabilities, retain_graph=True)
        updated_actor_gradients = []
        for actor_gradient, kullback_leibler_gradient in zip(actor_gradients, kullback_leibler_gradients):
            scale = actor_gradient.mul(
                kullback_leibler_gradient).sum(-1).unsqueeze(-1) - TRUST_REGION_CONSTRAINT
            scale = torch.div(scale, actor_gradient.mul(
                actor_gradient).sum(-1).unsqueeze(-1)).clamp(min=0.)
            updated_actor_gradients.append(
                actor_gradient - scale * kullback_leibler_gradient)
        return updated_actor_gradients


class ContinuousAgent(Agent):
    def __init__(self, brain, render=True, verbose=True):
        super().__init__(brain, render, verbose)
        self.noise = OrnsteinUhlenbeckProcess(size=ACTION_SPACE_DIM, theta=ORNSTEIN_UHLENBECK_NOISE_SCALE * 0.15,
                                              mu=- 0.1, sigma=ORNSTEIN_UHLENBECK_NOISE_SCALE * 0.2)

    def run(self):
        """
        Run the agent for several episodes.
        """
        for episode in range(MAX_EPISODES):
            noise_ratio = INITIAL_ORNSTEIN_UHLENBECK_NOISE_RATIO - (episode / NUMBER_OF_EXPLORATION_EPISODES) \
                if episode < NUMBER_OF_EXPLORATION_EPISODES * INITIAL_ORNSTEIN_UHLENBECK_NOISE_RATIO else 0.
            episode_rewards = 0.
            end_of_episode = False
            if self.verbose:
                print("Episode #{}, noise ratio {:.2f}".format(
                    episode, noise_ratio), end="")
            while not end_of_episode:
                trajectory = self.explore(self.brain.actor_critic, noise_ratio)
                end_of_episode = trajectory[-1].done[0, 0]
                episode_rewards += sum([transition.rewards[0, 0]
                                       for transition in trajectory])
                for trajectory_count in range(np.random.poisson(REPLAY_RATIO)):
                    trajectory = self.buffer.sample(
                        OFF_POLICY_MINIBATCH_SIZE, MAX_REPLAY_SIZE)
                    if trajectory:
                        self.learning_iteration(trajectory)
            self.noise.reset()
            if self.verbose:
                print(", episode rewards {:.2f}".format(episode_rewards))

    def explore(self, actor_critic, noise_ratio=0.):
        """
        Explore an environment by taking a sequence of actions and saving the results in the memory.

        Parameters
        ----------
        actor_critic : ActorCritic
            The actor-critic model to use to explore.
        noise_ratio : float in [0, 1], optional
            What fraction of the action should be exploration noise?
        """
        state = torch.FloatTensor(self.env.env.state)
        trajectory = []
        for step in range(MAX_STEPS_BEFORE_UPDATE):
            policy_mean, *_ = actor_critic(Variable(state))
            policy_logsd = actor_critic.policy_logsd
            action = torch.normal(
                policy_mean.data, torch.exp(policy_logsd.data))

            noise_mean, noise_sd = self.noise.sampling_parameters()
            noise = torch.from_numpy(self.noise.sample()).float()
            action = noise_ratio * noise + (1. - noise_ratio) * action
            sampling_mean = noise_ratio * \
                torch.from_numpy(noise_mean).float() + \
                (1. - noise_ratio) * policy_mean.data
            sampling_logsd = 0.5 * torch.log(noise_ratio**2 * torch.from_numpy(noise_sd).float().pow(2)
                                             + (1. - noise_ratio)**2 * torch.exp(2 * policy_logsd.data))
            exploration_statistics = torch.cat(
                [sampling_mean.view(1, -1), sampling_logsd.view(1, -1)], dim=1)

            scaled_action = float(self.env.action_space.low) \
                + float(self.env.action_space.high -
                        self.env.action_space.low) * torch.sigmoid(action)
            next_state, reward, done, _ = self.env.step(scaled_action.numpy())
            next_state = torch.from_numpy(next_state).float()
            if self.render:
                self.env.render()
            transition = replay_memory.Transition(states=state.view(1, -1),
                                                  actions=action.view(1, -1),
                                                  rewards=torch.FloatTensor(
                                                      [[reward]]),
                                                  next_states=next_state.view(
                                                      1, -1),
                                                  done=torch.FloatTensor(
                                                      [[done]]),
                                                  exploration_statistics=exploration_statistics)
            self.buffer.add(transition)
            trajectory.append(transition)
            if done:
                self.env.reset()
                break
            else:
                state = next_state
        return trajectory

    def learning_iteration(self, trajectory):
        """
        Conduct a single continuous learning iteration. Analogue of Algorithm 3 in the paper.
        """
        actor_critic = ContinuousActorCritic()
        actor_critic.copy_parameters_from(self.brain.actor_critic)

        _, _, _, next_states, _, _ = trajectory[-1]
        _, final_value, _ = actor_critic(Variable(next_states))
        retrace_action_value = final_value.data
        opc_action_value = final_value.data

        for states, actions, rewards, _, done, exploration_statistics in reversed(trajectory):
            policy_mean, value, action_value = actor_critic(
                Variable(states), Variable(actions))
            policy_logsd = actor_critic.policy_logsd
            average_policy_mean, * \
                _ = self.brain.average_actor_critic(
                    Variable(states), Variable(actions))
            average_policy_logsd = self.brain.average_actor_critic.policy_logsd
            exploration_statistics = torch.split(exploration_statistics,
                                                 split_size_or_sections=exploration_statistics.size(-1) // 2, dim=-1)
            exploration_policy_mean, exploration_policy_logsd = exploration_statistics

            importance_weights = self.normal_density(
                actions, policy_mean.data, policy_logsd.data)
            importance_weights /= self.normal_density(
                actions,  exploration_policy_mean, exploration_policy_logsd)
            alternative_actions = torch.normal(policy_mean.data,
                                               torch.exp(torch.ones(policy_mean.size(0), 1) * policy_logsd.data))
            _, _, alternative_action_value = actor_critic(
                Variable(states), Variable(alternative_actions))
            alternative_importance_weights = self.normal_density(alternative_actions,
                                                                 policy_mean.data, policy_logsd.data)
            alternative_importance_weights /= self.normal_density(alternative_actions,
                                                                  exploration_policy_mean, exploration_policy_logsd)
            truncation_parameter = importance_weights.pow(
                1 / ACTION_SPACE_DIM).clamp(max=1.)[0, 0]

            retrace_action_value = rewards + DISCOUNT_FACTOR * \
                retrace_action_value * (1. - done)
            opc_action_value = rewards + DISCOUNT_FACTOR * \
                opc_action_value * (1. - done)
            naive_alternative_advantage = alternative_action_value.data - value.data
            opc_advantage = opc_action_value - value.data

            # Actor
            actor_loss = - ACTOR_LOSS_WEIGHT * Variable(importance_weights.clamp(max=TRUNCATION_PARAMETER) * opc_advantage) \
                * self.normal_log_density(Variable(actions), policy_mean, policy_logsd)
            bias_correction = - ACTOR_LOSS_WEIGHT * Variable(
                (1 - TRUNCATION_PARAMETER / alternative_importance_weights).clamp(min=0.) * naive_alternative_advantage) \
                * self.normal_log_density(Variable(alternative_actions), policy_mean, policy_logsd)
            actor_loss += bias_correction
            actor_gradients = torch.autograd.grad(
                actor_loss.mean(), (policy_mean, policy_logsd), retain_graph=True)
            actor_gradients = self.continuous_trust_region_update(actor_gradients, policy_mean, policy_logsd,
                                                                  average_policy_mean, average_policy_logsd)
            torch.autograd.backward(
                (policy_mean, policy_logsd), actor_gradients, retain_graph=True)

            # Critic
            critic_loss = - Variable(retrace_action_value - action_value.data) * action_value \
                          - Variable(importance_weights.clamp(max=1.) *
                                     (retrace_action_value - action_value.data)) * value
            critic_loss.mean().backward(retain_graph=True)

            # Entropy
            entropy_loss = - ENTROPY_REGULARIZATION * \
                (policy_logsd + 0.5 * np.log(2 * np.pi * np.e)).sum(-1)
            entropy_loss.mean().backward(retain_graph=True)

            retrace_action_value = truncation_parameter * \
                (retrace_action_value - action_value.data) + value.data
            opc_action_value = (opc_action_value -
                                action_value.data) + value.data
        self.brain.actor_critic.copy_gradients_from(actor_critic)
        self.optimizer.step()
        self.brain.average_actor_critic.copy_parameters_from(
            self.brain.actor_critic, decay=TRUST_REGION_DECAY)

    @staticmethod
    def normal_density(action, mean, logsd):
        logsd = torch.ones(mean.size(0), 1) * logsd
        return torch.exp(-(action - mean).pow(2) / 2 / torch.exp(2 * logsd)) / np.sqrt(2 * np.pi) / torch.exp(logsd)

    @staticmethod
    def normal_log_density(action, mean, logsd):
        try:
            logsd = Variable(torch.ones(mean.size(0), 1)) * logsd
        except TypeError:
            logsd = torch.ones(mean.size(0), 1) * logsd
        return -(action - mean).pow(2) / 2 / torch.exp(2 * logsd) - 0.5 * np.log(2 * np.pi) - logsd

    @staticmethod
    def continuous_trust_region_update(actor_gradients, mean, logsd, average_mean, average_logsd):
        """
        Update the actor gradients so that they satisfy a linearized KL constraint with respect
        to the average actor-critic network. See Section 3.3 of the paper for details.

        Parameters
        ----------
        actor_gradients : tuple of torch.Tensor's
            The original gradients.
        mean, logsd
            The policy parameters according to the current actor-critic network.
        average_mean, average_logsd
            The policy parameters according to the average actor-critic network.

        Returns
        -------
        tuple of torch.Tensor's
            The updated gradients.
        """
        negative_kullback_leibler = 0.5 + average_logsd - logsd \
            - (torch.exp(2 * average_logsd) + (mean - average_mean).pow(2)) \
            / 2 / torch.exp(2 * logsd)
        kullback_leibler_gradients = torch.autograd.grad(negative_kullback_leibler.mean(),
                                                         (mean, logsd), retain_graph=True)
        updated_actor_gradients = []
        for actor_gradient, kullback_leibler_gradient in zip(actor_gradients, kullback_leibler_gradients):
            scale = actor_gradient.mul(
                kullback_leibler_gradient).sum(-1).unsqueeze(-1) - TRUST_REGION_CONSTRAINT
            scale = torch.div(scale, kullback_leibler_gradient.mul(kullback_leibler_gradient).sum(-1).unsqueeze(-1)
                              ).clamp(min=0.)
            updated_actor_gradients.append(
                actor_gradient - scale * kullback_leibler_gradient)
        return updated_actor_gradients
