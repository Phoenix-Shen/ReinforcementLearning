# policy greadient

    在连续区间可用，克服 Qlearning 的缺点，没有 Qlearning 的误差，但是它确实进行了反向传递，反向传递的结果就是让这个动作**更有可能发生**，即增加可能性

---

<br>
    不同于Qlearning，它直接输出的是动作而不是动作对应的value，但是他也要接受环境信息observation
<br>

---

## 它应该是这种形式

---

```
class PolicyGradient(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_net(self):
        pass

    def choose_action(self, observation):
        pass

    def store_transition(self, s, a, r):
        pass

    def learn(self, s, a, r, _s):
        pass

    def _discount_and_norm_rewards(self):
        pass

```

## tensorflow 版本在 tf=2.7 的时候无法运行,提示没有 placeholder 这个函数，垃圾玩意

## pytorch 写出来了一个，但是梯度下降不下去，不知道为什么，遂放弃。main.py 是照搬网上的一个代码

## 花了一天时间，看出来了，有一个步骤使用了 detach，导致梯度消失了
