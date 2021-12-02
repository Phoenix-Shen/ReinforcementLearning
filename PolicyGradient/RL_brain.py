import torch as t
import numpy as np
import matplotlib.pyplot as plt
import visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyGradient(nn.Module):
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.95,output_graph=False) -> None:
        super().__init__()
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay
        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]
        
        self.net=nn.Sequential(
            nn.Linear(in_features=n_features,out_features=10),
            nn.Tanh(),
            nn.Linear(in_features=10,out_features=n_actions),
            # 在pytorch当中，softmax的参数为NONE会报警：UserWarning: Implicit dimension choice for softmax has been deprecated. Change 
            # the call to include dim=X as an argument. https://blog.csdn.net/weixin_41391619/article/details/104823086
            nn.Softmax()
        )
        
        self.optimizer=optim.Adam(self.net.parameters(),lr=learning_rate)
        self.loss_function=nn.CrossEntropyLoss()

    def choose_action(self, observation):
        x=t.unsqueeze(t.FloatTensor(observation),0)
        prob_weights=self.net(x).detach()
        #根据概率来选择下一步行动,还是具有随机性。
        #p=prob_weights.squeeze(dim=0).numpy() 不进行转Numpy会报错，值加起来不为1，判定较为严格
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.squeeze(dim=0).numpy())
        return action
    
    #存储我们的一系列操作
    def store_transition(self, s, a, r):
        self.ep_as.append(a)
        self.ep_obs.append(s)
        self.ep_rs(r)

    def learn(self):
        ep_obs_tensor = t.FloatTensor(self.ep_obs).unsqueeze(dim=0)
        ep_as_tensor = t.FloatTensor(self.ep_as).unsqueeze(dim=0)
        ep_rs_tensor = t.FloatTensor(self._discount_and_norm_rewards()).unsqueeze(dim=0)
        
        probabilities=self.net(ep_obs_tensor)
        neg_log_prob=self.loss_function(probabilities,ep_as_tensor)
        loss=t.mean(neg_log_prob*ep_rs_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
