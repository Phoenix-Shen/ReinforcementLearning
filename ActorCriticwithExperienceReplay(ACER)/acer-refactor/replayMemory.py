# %%
import numpy as np
import random
from collections import deque, namedtuple
import torch


Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'done', 'exploration_statistics'))


class ReplayBuffer():
    """
    replay buffer that stores the trajectories
    """

    def __init__(self, replay_size: int) -> None:
        self.episodes = deque([[]], maxlen=replay_size)

    def add(self, transition):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        transition : Transition
            The transition to add.
        """

        if self.episodes[-1] and self.episodes[-1][-1].done[0, 0]:
            # if we reach a terminal state , then extend a new empty trajectory

            self.episodes.append([])
        # append the transition to the last elements, do not need to worry about the terminal state
        self.episodes[-1].append(transition)

    def sample(self, batch_size: int):
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
        # 随机取下标，如果个数不满batchsize就全部拿出来
        trajectory_indices = np.random.choice(
            range(len(self.episodes)-1), min(batch_size, len(self.episodes)-1))

        trajectories = []
        # 对于选到的下标，进行如下操作：
        for trajectory in [self.episodes[index] for index in trajectory_indices]:
            # 随机选取轨迹的开始
            start = np.random.choice(range(len(trajectory)))
            # 选取出来 加入到轨迹里面
            trajectories.append(trajectory[start:])
        # 取得最短的轨迹长度，并裁剪其它轨迹到最短长度，就跟剪头发一样，都剪一样长,如果没有采样到数据，那么长度就是0了
        smallest_trajectory_length = min(
            [len(trajectory) for trajectory in trajectories]) if trajectories else 0
        for index in range(len(trajectories)):
            # 截断，但是最终状态要保留，所以只能截断前面的
            trajectories[index] = trajectories[index][-smallest_trajectory_length:]
        # 转置： 将所有轨迹的第一步、第二步、第三步……放在一起
        for transitions in zip(*trajectories):
            # 将所有的轨迹的第一步 第二步 第三步 …… 的 observation reward ……放在一起
            # 顺便使用torch.cat转换成tensor
            batched_transition = Transition(
                *[torch.cat(data, dim=0) for data in zip(*transitions)])
            # 添加到返回的值里面去
            batched_trajectory.append(batched_transition)
        return batched_trajectory


'''
# %%
# 测试两个函数有啥不一样
def npchoiceAndRandomchoices():
    a = np.random.choice(range(500),
                         min(32, 500))
    b = np.random.choice(range(10))
    random.choices(range(10), k=1)
# %%

def memTest():
    mem = ReplayBuffer(32)
    for i in range(32):
        mem.add(Transition(1+i, 2+i, 3+i, 4+i,
                np.array([[False]], dtype=np.bool8), 5+i))

    print(mem.sample(4))


# %%
if __name__ == "__main__":
    memTest()
'''
