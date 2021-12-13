# %%

# init SummaryWriter

import torchvision
import tensorboardX as tb
import torch as t
import os

# args:logdir(where to store the files)
writer = tb.SummaryWriter("./")

# %%
#scalar and histogram
a = t.tensor([1, 2, 3, 4, 5, 6])
for idx, a_ in enumerate(a):
    writer.add_scalar('tensor', a_, idx)
    writer.add_histogram("tensor_his", a_, idx)
# %%
# feature map and conv kernel
writer.add_graph(torchvision.models.resnet18(False), t.rand([1, 3, 224, 224]))

# %%
