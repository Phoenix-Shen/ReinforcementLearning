# TRPO

信赖域策略优化  
策略梯度的算法缺点在于学习率 lr 或者是更新步长，所以要采取措施使步长不能太大

## TRPO 使用了 3 个技巧

1. 使用旧的策略所对应的状态分布来近似新策略的状态分布
2. 采用 importance sampling 对动作分布进行处理
3. 采用平均 KL 散度代替最大 KL 散度

## 关于 nn.parameters

        '''torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。
        注意到，nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的，这与torth.Tensor对象的默认值相反。
        self.action_log_std = nn.Parameter(t.zeros(1, n_actions))'''

## 拿来主义

`the code in this folder is not functional! the code in github https://github.com/Khrylx/PyTorch-RL is recommended.`

Running_state.py 是做什么东西的？

agent.py 第 124 行代码会出错，是真的搞不来

[拿来主义-github 现成代码](https://github.com/Khrylx/PyTorch-RL)

## Trust Region 算法

- 梯度上升算法回顾

  1. 目标：找到 theta 使 J(theta)最大，其中 J 是目标函数
  2. 梯度上升算法重复这个步骤\[计算在 theta_old 处的梯度，然后执行 theta_new=theta_old+alpha\*gradient\]
  3. 因为我们是要使 J 最大，所以使用梯度上升，反过来，我们如果要最小，就要下降梯度
  4. 前提是我们要知道 theta 关于 J 的梯度
  5. 在某些情况下我们不能求梯度
     - 没有解析解的情况下，不能求梯度，需要随机采样求梯度然后重复梯度上升算法
     - 求出来的梯度是对原来梯度的蒙特卡洛近似

- 置信域

  1. N（theta）代表 theta 的邻域，一个集合，包含 theta 旁边的所有点
  2. L(theta|theta_old)在 N（theta_old）邻域非常接近 J（theta），我们称 N（theta_old）为置信域
  3. 在这个邻域上可以用 L 来代替 J，而 J 很复杂，L 相对简单，可以让优化更容易，注意到只有在 N 这个置信域上才能够代替

- 置信域算法
  - 近似：给定 theta_old，构建函数 L(theta|theta_old)使在 N（theta_old)上近似于 J（theta）
  - 最大化：在置信域中，通过使 L 最大的方法找到 theta_new，而不是使 J 最大
  - 速度：比随机梯度上升要慢，但是表现好，更稳定

## 基于策略的强化学习

- 策略梯度
  1. 定义策略网络 pi（a|s；theta），来控制智能体
  2. 状态-价值函数![](assets\value.png)
  3. 目标函数 J（theta）=E_s\[V_pi(S)\]
  4. 状态-价值函数做变换![](assets\value_transform.png)
  5. 目标函数 J 做变换![](assets\Jtheta.png)

## TRPO 算法

- 为什么使用？

  1. 更稳定，对超参数的变动不敏感
  2. Sample efficiency

- 流程：

  1. 给定 theta_old，构造 L 使 L 在置信域中近似于 J
     - 已经有了`目标函数J做变换`，算法从中衍生。![](assets\Jtheta.png)
     - 其中 S 是从环境抽样得到的，A 是从策略网络 pi（A|theta_old)来取得的
     - 作蒙特卡洛近似：在环境中抽取 s，通过 pi（A|theta_old)获得动作 a……，由此连成一条轨迹（s1,a1,r1,s2,a2,r2……）我们称之为 trajectory（轨迹）,基于 N 个观测值的蒙特卡洛近似求期望![](assets\approximation.png) 这就是我们对于 J 的近似
     - 但是我们不知道最后的 Q_pi（si，ai）怎么算，所以又要对 Q_pi 做近似
     - 我们记录到了奖励，记录奖励 r_1,r_2,r_3,r_4，我们还能够得到折扣奖励（discounted rewards）u_i，Q_pi 是 u_i 的期望，可以蒙特卡洛近似![](assets\discountedRewards.png)
     - 这下所有的变量我们都有了
  2. 在 N（theta_old）里面通过使 L 最大的方法来找到 theta_new，即 ARGMAX_theta（L)
     - 即使我们的 theta_new 不好，由于置信域的存在，我们的 theta_new 不会太糟糕
     - N 具体怎么实现呢？
       1. ||theta-theta_old||<δ
       2. 求 KL_divergence，使 KL<δ

- 具体算法

  1. 收集数据，使用 pi(·|s；theta_old)来玩游戏，玩到终点，收集一个 trajectory（轨迹）：s1,a1,r1,s2,a2,r2……
  2. 走了 n 步，计算 n 个折扣奖励 u_i
  3. 近似：构造目标函数 L![](assets\discountedRewards.png)
  4. 最大化：找到使 L 最大的参数 theta_new，而且 theta_new 满足约束：theta_new 在 N(theta_old)也就是置信域里面
     - 求解优化问题需要多轮循环，使用梯度投影等方法求解，属于数值优化范畴，不属于 RL
     - 两个超参数：learning rate 和 delta

- 对比 PG
  - 目标是一样的
  - PG 使用随机梯度上升来最大化 J
  - TRPO 使用置信域算法来最大化 J
