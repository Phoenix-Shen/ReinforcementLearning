import torch as t
from torch import tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class DuelingDQN(nn.Module):
    def __init__(self,n_states:int,n_actions:int,hidden_layers:int):
        super().__init__()
        self.input_size=n_states
        self.output_size=n_actions
        self.hidden_layers=hidden_layers
        
        self.feature_layer=nn.Linear(self.input_size,self.hidden_layers)
        self.value_layer=nn.Linear(self.hidden_layers,1)#输出value
        self.advantage_layer=nn.Linear(self.hidden_layers,self.output_size)#输出每个动作的优先值
       
    
    def forward(self,x:t.Tensor)->t.Tensor:
        feature1= F.relu(self.feature_layer(x))
        feature2= F.relu(self.feature_layer(x))
        #value
        value=self.value_layer(feature1)
        #advantage
        advantage=self.advantage_layer(feature2)
        return value+advantage-advantage.mean(dim=1,keepdim=True)