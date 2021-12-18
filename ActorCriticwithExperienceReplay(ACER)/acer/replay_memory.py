import random
import torch
from collections import deque, namedtuple
from core import *

Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'done', 'exploration_statistics'))


class ReplayBuffer:
    """
    Replay buffer for the agents.
    """

    def __init__(self):
        self.episodes = deque([[]], maxlen=REPLAY_BUFFER_SIZE)

    def add(self, transition):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        transition : Transition
            The transition to add.
        """
        if self.episodes[-1] and self.episodes[-1][-1].done[0, 0]:
            self.episodes.append([])
        self.episodes[-1].append(transition)

    def sample(self, batch_size, window_length=float('inf')):
        """
        Sample a batch of trajectories from the buffer. If they are of unequal length
        (which is likely), the trajectories will be padded with zero-reward transitions.

        Parameters
        ----------
        batch_size : int
            The batch size of the sample.
        window_length : int, optional
            The window length.

        Returns
        -------
        list of Transition's
            A batched sampled trajectory.
        """
        batched_trajectory = []
        # 获得下标，如果个数不满batchsize则全部取出来
        trajectory_indices = random.choices(
            range(len(self.episodes)-1), k=min(batch_size, len(self.episodes)-1))
        trajectories = []
        # 获得所有的轨迹的下标
        for trajectory in [self.episodes[index] for index in trajectory_indices]:
            # 随机选取开始的值，
            start = random.choices(range(len(trajectory)), k=1)[0]
            # 将它加入到轨迹里头
            trajectories.append(trajectory[start:start + window_length])
        # 取得最短的轨迹值，并裁剪所有的值
        smallest_trajectory_length = min(
            [len(trajectory) for trajectory in trajectories]) if trajectories else 0
        for index in range(len(trajectories)):
            trajectories[index] = trajectories[index][-smallest_trajectory_length:]
        # 转置 - 将所有轨迹的第一步 第二步 第三步 放在一起
        for transitions in zip(*trajectories):
            # 将所有的轨迹的第一步 第二步 第三步 …… 的 observation reward ……放在一起
            batched_transition = Transition(
                *[torch.cat(data, dim=0) for data in zip(*transitions)])
            batched_trajectory.append(batched_transition)
        # 返回的 是这个样子 (min_traj_length,batch,5个信息)
        return batched_trajectory

    @staticmethod
    def extend(transition):
        """
        Generate a new zero-reward transition to extend a trajectory.

        Parameters
        ----------
        transition : Transition
            A terminal transition which will become the new transition's previous
            transition in the trajectory.

        Returns
        -------
        Transition
            The new transition that can be used to extend a trajectory.
        """
        if not transition.done[0, 0]:
            raise ValueError("Can only extend a terminal transition.")
        exploration_statistics = torch.ones(transition.exploration_statistics.size()) \
            / transition.exploration_statistics.size(-1)
        transition = Transition(states=transition.next_states,
                                actions=transition.actions,
                                rewards=torch.FloatTensor([[0.]]),
                                next_states=transition.next_states,
                                done=transition.done,
                                exploration_statistics=exploration_statistics)
        return transition
