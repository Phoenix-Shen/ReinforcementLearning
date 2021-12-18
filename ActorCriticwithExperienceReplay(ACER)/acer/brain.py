import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import *


class ActorCritic(torch.nn.Module):
    """
    Actor-critic network used in A3C and ACER.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def copy_parameters_from(self, source, decay=0.):
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
            parameter.data.copy_(decay * parameter.data +
                                 (1 - decay) * source_parameter.data)

    def copy_gradients_from(self, source):
        """
        Copy the gradients from another network.

        Parameters
        ----------
        source : ActorCritic
            The network from which to copy the gradients.
        """
        for parameter, source_parameter in zip(self.parameters(), source.parameters()):
            parameter._grad = source_parameter.grad


class DiscreteActorCritic(ActorCritic):
    """
    Discrete actor-critic network used in A3C and ACER.
    """

    def __init__(self):
        super().__init__()
        self.input_layer = torch.nn.Linear(STATE_SPACE_DIM, 32)
        self.hidden_layer = torch.nn.Linear(32, 32)
        self.action_layer = torch.nn.Linear(32, ACTION_SPACE_DIM)
        self.action_value_layer = torch.nn.Linear(32, ACTION_SPACE_DIM)

    def forward(self, states):
        """
        Compute a forward pass in the network.

        Parameters
        ----------
        states : torch.Tensor
            The states for which the action probabilities and the action-values must be computed.

        Returns
        -------
        action_probabilities : torch.Tensor
            The action probabilities of the policy according to the actor.
        action_probabilities : torch.Tensor
            The action-values of the policy according to the critic.
        """
        hidden = F.relu(self.input_layer(states))
        hidden = F.relu(self.hidden_layer(hidden))
        action_probabilities = F.softmax(self.action_layer(hidden), dim=-1)
        action_values = self.action_value_layer(hidden)
        return action_probabilities, action_values


class ContinuousActorCritic(ActorCritic):
    """
    Discrete actor-critic network used in A3C and ACER.
    """

    def __init__(self):
        super().__init__()
        self.policy_input_layer = torch.nn.Linear(STATE_SPACE_DIM, 32)
        self.policy_hidden_layer = torch.nn.Linear(32, 32)
        self.policy_mean_layer = torch.nn.Linear(32, ACTION_SPACE_DIM)
        self.policy_logsd = torch.nn.Parameter(
            np.log(INITIAL_STANDARD_DEVIATION) * torch.ones((1, ACTION_SPACE_DIM)))
        self.value_layer = torch.nn.Linear(32, 1)

        self.sdn_state_input_layer = torch.nn.Linear(STATE_SPACE_DIM, 32)
        self.sdn_action_input_layer = torch.nn.Linear(ACTION_SPACE_DIM, 32)
        self.sdn_hidden_layer = torch.nn.Linear(32, 32)
        self.sdn_advantage_layer = torch.nn.Linear(32, 1)

    def forward(self, states, actions=None):
        """
        Compute a forward pass in the network.

        Parameters
        ----------
        states : torch.Tensor
            The states for which the action probabilities and the action-values must be computed.
        actions : torch.Tensor, optional
            The actions for which the action-values must be computed.

        Returns
        -------
        action_probabilities : torch.Tensor
            The action probabilities of the policy according to the actor.
        value : torch.Tensor
            The value of the policy according to the critic.
        action_value : torch.Tensor
            The action-value of the policy according to the critic.
        """
        hidden = F.relu(self.policy_input_layer(states))
        hidden = F.relu(self.policy_hidden_layer(hidden))
        policy_mean = self.policy_mean_layer(hidden)
        value = self.value_layer(hidden)

        if actions is not None:
            advantage = self.sdn_forward(states, actions)

            action_samples = [Variable(torch.normal(policy_mean.data,
                                                    torch.exp(torch.ones(policy_mean.size(0), 1) * self.policy_logsd.data)))
                              for _ in range(5)]
            advantage_samples = torch.cat([self.sdn_forward(states, action_sample).unsqueeze(-1)
                                           for action_sample in action_samples], -1)
            action_value = value + advantage - advantage_samples.mean(-1)
            return policy_mean, value, action_value
        else:
            return policy_mean, value, None

    def sdn_forward(self, states, actions):
        hidden = F.relu(self.sdn_state_input_layer(states) +
                        self.sdn_action_input_layer(F.tanh(actions)))
        hidden = F.relu(self.sdn_hidden_layer(hidden))
        advantage = self.sdn_advantage_layer(hidden)
        return advantage


class Brain:
    """
    A centralized brain for the agents.
    """

    def __init__(self):
        self.actor_critic = None
        self.average_actor_critic = None


class DiscreteBrain(Brain):
    def __init__(self):
        super().__init__()
        self.actor_critic = DiscreteActorCritic()
        self.actor_critic.share_memory()
        self.average_actor_critic = DiscreteActorCritic()
        self.average_actor_critic.share_memory()
        self.average_actor_critic.copy_parameters_from(self.actor_critic)


class ContinuousBrain(Brain):
    def __init__(self):
        super().__init__()
        self.actor_critic = ContinuousActorCritic()
        self.actor_critic.share_memory()
        self.average_actor_critic = ContinuousActorCritic()
        self.average_actor_critic.share_memory()
        self.average_actor_critic.copy_parameters_from(self.actor_critic)


if CONTROL is 'discrete':
    brain = DiscreteBrain()
else:
    brain = ContinuousBrain()
