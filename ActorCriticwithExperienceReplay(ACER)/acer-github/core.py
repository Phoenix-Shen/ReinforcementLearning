import gym
from gym.spaces import Discrete as DiscreteSpace

ENVIRONMENT_NAME = 'LunarLander-v2'
# ENVIRONMENT_NAME = 'MountainCarContinuous-v0'

env = gym.make(ENVIRONMENT_NAME)
action_space = env.action_space
state_space = env.observation_space
env.close()
del env

if isinstance(action_space, DiscreteSpace):
    ACTION_SPACE_DIM = action_space.n
    CONTROL = 'discrete'
else:
    ACTION_SPACE_DIM = action_space.shape[0]
    CONTROL = 'continuous'
STATE_SPACE_DIM = state_space.shape[0]

# Parameters that work well for CartPole-v0
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 25
TRUNCATION_PARAMETER = 10
DISCOUNT_FACTOR = 0.99
REPLAY_RATIO = 4
MAX_EPISODES = 200
MAX_STEPS_BEFORE_UPDATE = 2000000
NUMBER_OF_AGENTS = 6
OFF_POLICY_MINIBATCH_SIZE = 16
TRUST_REGION_CONSTRAINT = 1.
TRUST_REGION_DECAY = 0.99
ENTROPY_REGULARIZATION = 1e-3
MAX_REPLAY_SIZE = 200
ACTOR_LOSS_WEIGHT = 0.1
# Not used for discrete agents
ORNSTEIN_UHLENBECK_NOISE_SCALE = None
INITIAL_ORNSTEIN_UHLENBECK_NOISE_RATIO = None
NUMBER_OF_EXPLORATION_EPISODES = None
INITIAL_STANDARD_DEVIATION = None

# Parameters that work well for MountainCarContinuous-v0
# LEARNING_RATE = 5e-4
# REPLAY_BUFFER_SIZE = 25
# TRUNCATION_PARAMETER = 5
# DISCOUNT_FACTOR = 0.99
# REPLAY_RATIO = 4
# MAX_EPISODES = 200
# MAX_STEPS_BEFORE_UPDATE = 200
# NUMBER_OF_AGENTS = 12
# OFF_POLICY_MINIBATCH_SIZE = 4
# TRUST_REGION_CONSTRAINT = 1.
# TRUST_REGION_DECAY = 0.99
# ENTROPY_REGULARIZATION = 0.
# MAX_REPLAY_SIZE = 200
# ACTOR_LOSS_WEIGHT = 0.1
# ORNSTEIN_UHLENBECK_NOISE_SCALE = 5
# INITIAL_ORNSTEIN_UHLENBECK_NOISE_RATIO = 1.
# NUMBER_OF_EXPLORATION_EPISODES = 50
# INITIAL_STANDARD_DEVIATION = 5.
