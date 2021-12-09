# DuelingDQN with PER（PrioritizedExperienceReplay）

- 原神经网络输出的是 Q 的值即 q_target=net(state)<br>
  而 DuelingDQN 的每个动作为 q_target=value(state)+advantage(state,action)

- `其实就是将生成Q的值的层替换成两个层：生成value的层和生成advantage的层`

- `具体改动见model.py`

---

最后基本上很快就可以达到结果，缺点是前期的积累经验比较耗时间。<br>

---

预训练 350 轮的参数在 2021-12-9 16-9-21.pth 中，可以导入使用。
