# %%
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as transforms
from torch.functional import Tensor
import torch as t
from pltutils import *

# HYPER PARAMETERS
BATCH_SIZE = 256
NUM_INPUTS = 784
NUM_OUTPUTS = 10  # 10 类
#DEVICE = t.device("cpu")
DEVICE = t.device("cuda:0" if t.cuda.is_available() else "cpu")

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


# MODEL PARAMS
W = t.normal(0, 0.01, size=(NUM_INPUTS, NUM_OUTPUTS),
             requires_grad=True, device=DEVICE)
b = t.zeros(NUM_OUTPUTS, requires_grad=True, device=DEVICE)

# SOFTMAX
# 相当于nn.Softmax(1)
# 注意，虽然这在数学上看起来是正确的，
# 但我们在代码实现中有点草率。 矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，
# 但我们没有采取措施来防止这点。


def softmax(X: Tensor):
    X_exp = t.exp(X)
    # 对列求和，再除以和
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp/partition

# MODEL


def net(X: Tensor):
    result = t.matmul(X.reshape((-1, W.shape[0])), W)+b
    return softmax(result)
# CROSSENTROPY


def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    return -t.log(y_hat[range(len(y_hat)), y])

# ACCURACY


def accuracy(y_hat: Tensor, y: Tensor) -> Tensor:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# TRAIIN


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
            print("ep:{},accuracy:{},loss:{}".format(
                i, accuracy(y_hat, y)/y.shape[0], l.mean().item()))

# OPTIMIZER


def stochastic_gradient_desent(params: t.Tensor, lr, batch_size):
    with t.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()


def updater(batch_size):
    stochastic_gradient_desent([W, b], 0.1, batch_size)


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


# %%
train_loader, test_loader = load_data_fashion_mnist(BATCH_SIZE, n_threads=0)
train(net, train_loader, cross_entropy, updater, 1)
predict(net, test_loader)

# %%
