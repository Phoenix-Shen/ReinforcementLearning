import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self,
                 n_features:int,
                 n_actions:int,
                 boundary:list=[-2,2]):
        super().__init__()
        # PARAMETERS
        self.n_features=n_features
        self.n_actions=n_actions
        self.boundary=boundary
        # NET STRUCTURE
        self.net=nn.Sequential(
            nn.Linear(self.n_features,256),
            nn.ReLU(),
            nn.Linear(256,self.n_actions),
            nn.Tanh(),
        )
    
    def forward(self,x:t.Tensor)->t.Tensor:
        x_=self.net(x)
        low=self.boundary[0]
        high=self.boundary[1]
        #归一化到我们想要的区间,从tanh出来的那一层它的范围是[-1,1]
        return low+(high-low)/(1-(-1))*(x_-(-1))
    
class Critic(nn.Module):
    def __init__(self,
                 n_features:int,
                 n_actions:int,
                 ):
        super().__init__()
        # PARAMETERS
        self.n_features=n_features
        self.n_actions=n_actions
        
        # NET STRUCTURE
        self.fcs=nn.Linear(self.n_features,256)
        self.fca=nn.Linear(self.n_actions,256)
        self.out=nn.Linear(256,1)
    
    def forward(self,s:t.Tensor,a:t.Tensor)->t.Tensor:
        s_v=self.fcs(s)
        a_v=self.fca(a)
        actions_value=self.out(F.relu(a_v+s_v))
        return actions_value