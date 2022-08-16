# Actor-Critic 算法

结合了 policy gradient 和 function approximation 的方法

---

## **简单来说，就是将 vt 从固定值换成神经网络生成的结果**

```python
action=Actior(observation)
score=Critic(action)

loss=loss_function(score)
loss.backward()
```

## actor critic

Actor-Critic 涉及到了两个神经网络, 而且每次都是在连续状态中更新参数, 每次参数更新前后都存在相关性, 导致神经网络只能片面的看待问题, 甚至导致神经网络学不到东西. Google DeepMind 为了解决这个问题, 修改了 Actor Critic 的算法也就是我们之后的 DDPG

## 算法更新可选形式

1. 使用状态价值 state value
2. 使用动作-状态价值 state-action value
3. 基于 TD error（本代码中的方法） tderror=r*t+1+ gamma\*Vs_t+1* Vs_t
4. 基于优势函数 Advantage = state-action value-state value
5. 基于 TD(λ)误差

## A2C

- 可以采用 discounted rewards - state value 也可以直接参考哦上面算法更新中的第四点来乘以 log_probs 作为损失回传。
