# ACER

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dchetelat/acer/blob/master/LICENSE.md)


PyTorch implementation of both discrete and continuous ACER algorithm for reinforcement learning.

(Original paper: Deepmind's [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224).)

The implementation is set up to interface with OpenAI Gym, although I tried to keep the code as clean and general as possible.
I took much inspiration from [Kaixhin's version](https://github.com/Kaixhin/ACER): the main differences are that my
implementation includes continuous control agents, as well as a correct implementation of ACER's truncated backpropagation
trust-region updates. Working hyperparameter values are provided for the CartPole-v0 and ContinuousMountainCar-v0 toy environments.

I think my code only deviates in one way from a vanilla ACER implementation: the sparse reward structure in 
Continuous Mountain Car made exploration too difficult through a simple entropy regularizer, so I added as well an annealed
Ornstein-Uhlenbeck noise process to the actions to stimulate inertia-driven exploration (like in [DDPG](https://arxiv.org/abs/1509.02971)). 
Depending on the type of target environment, this might not be necessary.
