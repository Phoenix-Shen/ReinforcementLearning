# Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation

经典的 DQN 在面临环境反馈稀疏和反馈延迟的情况下无能为力。原因在于这类游戏需要高级的策略。例如在 Montezuma’s Revenge 游戏中，无论 DQN 如何去学习均为０。原因在于这类游戏需要高级的策略。比如图中要拿到钥匙，然后去开门。这对我们而言是通过先验知识得到的。但是很难想象计算机如何仅仅通过图像感知这些内容。感知不到，那么这种游戏也就无从解决。

# HDQN 解决上述问题的方法

构造两个层级的算法，顶层用于决策，确定下一步的目标，底层用于具体行动。

复杂任务->分解为多个小任务，然后逐个实现小任务。 这符合我们人类的模式

# 具体实现

1. meta_controller : 负责获取当前状态 s，然后从可能的子任务里面选一个任务，交给下层控制器完成
2. controller: 负责接受上一个层级的子任务以及当前状态，然后选择行动去执行。它的目标是最大化由 critic 给出的 intrinsic reward 之和
