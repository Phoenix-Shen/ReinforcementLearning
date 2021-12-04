Actor-Critic 算法
结合了 policy gradient 和 function approximation 的方法<br>

---

## **简单来说，就是将 vt 从固定值换成神经网络生成的结果**

```
action=Actior(observation)
score=Critic(action)

loss=loss_function(score)
loss.backward()
```

## actor critic

Actor-Critic 涉及到了两个神经网络, 而且每次都是在连续状态中更新参数, 每次参数更新前后都存在相关性, 导致神经网络只能片面的看待问题, 甚至导致神经网络学不到东西. Google DeepMind 为了解决这个问题, 修改了 Actor Critic 的算法
