# %%
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as transforms
from torch.functional import Tensor
import torch as t
from pltutils import *
import torch.nn as nn

# DATALOADER


def load_data_fashion_mnist(batch_size, resize=None, n_threads=4):
    """下载fashion-MNIST数据集 将其加载到内存当中去"""
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(size=resize))
    trans = transforms.Compose(transform)
    mnist_train = tv.datasets.FashionMNIST(
        root="./PytorchAndPythonLearning/Pytorch/dataset", train=True, transform=trans, download=True)
    mnist_test = tv.datasets.FashionMNIST(
        root="./PytorchAndPythonLearning/Pytorch/dataset", train=False, transform=trans, download=True)
    train_loader = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=n_threads)
    test_loader = data.DataLoader(
        mnist_test, batch_size, shuffle=True, num_workers=n_threads)
    return train_loader, test_loader


def accuracy(y_hat: Tensor, y: Tensor) -> Tensor:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def train(net, train_iter: data.DataLoader, loss, updater, n_epochs=10):
    if isinstance(net, t.nn.Module):
        net.train()
    for i in range(n_epochs):
        for x, y in train_iter:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_hat = net(x)
            l = loss(y_hat, y)

            if isinstance(updater, t.optim.Optimizer):
                updater.zero_grad()
                l.sum().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(x.shape[0])
            print("ep:{},accuracy:{:.4f},loss:{:.4f}".format(
                i, accuracy(y_hat, y)/y.shape[0], l.mean().item()))


def predict(net, test_iter):
    correct = 0.
    for x, y in test_iter:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_hat = net(x)
        correct += accuracy(y_hat, y)
    correct = correct/10000.  # 10000是测试集的长度
    print("test ACC:{}".format(correct))
    return correct

# HYPER PARAMETERS
BATCH_SIZE = 2048
NUM_INPUTS = 784
NUM_OUTPUTS = 10  # 10 类
#DEVICE = t.device("cpu")
DEVICE = t.device("cuda:0" if t.cuda.is_available() else "cpu")

train_iter, test_iter = load_data_fashion_mnist(BATCH_SIZE,n_threads=0)
lossfunc = nn.CrossEntropyLoss()
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)
net.to(DEVICE)
optimizer = t.optim.SGD(net.parameters(), lr=0.05)
# SOFTMAX 的缺点
# 如果exp(net_output)里头有很大的数字，那么它会溢出掉
# 变成NAN或者inf最后导致诈胡
# 最好的办法是：减去一个最大值
# 但是如果相减，就导致结果太小，求log的时候，就会下溢。
# log的值为-inf
# 尽管要计算指数函数，最终我们可以在计算交叉熵损失的时候可以取对数
# 将Softmax和交叉熵结合在一起。
train(net, train_iter, lossfunc, optimizer, 1)
predict(net, test_iter)
