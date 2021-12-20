# TRPO

信赖域策略优化  
策略梯度的算法缺点在于学习率 lr 或者是更新步长，所以要采取措施使步长不能太大

# TRPO 使用了 3 个技巧

1. 使用旧的策略所对应的状态分布来近似新策略的状态分布
2. 采用 importance sampling 对动作分布进行处理
3. 采用平均 KL 散度代替最大 KL 散度

## 关于 nn.parameters

        '''torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。
        注意到，nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的，这与torth.Tensor对象的默认值相反。
        self.action_log_std = nn.Parameter(t.zeros(1, n_actions))'''

## 搞不来哦

`the code in this folder is not functional! the code in github https://github.com/Khrylx/PyTorch-RL is recommended.`

Running_state.py 是做什么东西的？

agent.py 第 124 行代码会出错，是真的搞不来

[拿来主义-github 现成代码](https://github.com/Khrylx/PyTorch-RL)
