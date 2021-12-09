# DDPG(Deep Deterministic Policy Gradient)

---

- `他在连续的动作空间上面有较好的表现`
- DDPG 是一种 ActorCritic 结构
- 它继承了 DQN 的 fixed Q target 和 Policy Gradient 的思想

---

其中 Critic 和 Actor 都有两个网络，它们是 DQN 的思想，参数异步更新。<br>
