import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym


class Net(nn.Module):
    def __init__(self, n_features: int, n_actions: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.pi1 = nn.Linear(self.n_features, 256)
        self.pi2 = nn.Linear(256, self.n_actions)

        self.v1 = nn.Linear(self.n_features, 256)
        self.v2 = nn.Linear(256, 1)

        self.distribution = t.distributions.Categorical

    def _init_weights(self):
        for layer in [self.pi1, self.pi2, self.v1, self.v2]:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        pi1 = F.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.tanh(self.v1(x))
        value = self.v2(v1)

        return logits, value

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).detach()
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t-values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a)*td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss+a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, n_features, n_actions):
        super().__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(n_features, n_actions)
        self.env = gym.make('CartPole-v0').unwrapped
        self.MAX_EP = 3000
        self.UPDATE_GLOBAL_ITER = 5
        self.GAMMA = 0.9

    def run(self):
        total_step = 1
        while self.g_ep.value < self.MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                if self.name == "w00":
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done:
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done,
                                  s_, buffer_s, buffer_a, buffer_r, self.GAMMA)

                buffer_s, buffer_a, buffer_r = [], [], []

                if done:  # done and print information
                    record(self.g_ep, self.g_ep_r, ep_r,
                           self.res_queue, self.name)
                    break
                s = s_
                total_step += 1
        self.res_queue.put(None)
##########UTILS###########


def v_wrap(np_array, dtype=np.float32) -> t.Tensor:
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return t.from_numpy(np_array)


def push_and_pull(opt, lnet: Net, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.
    else:
        v_s_ = lnet(v_wrap(s_[None]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:
        v_s_ = r+gamma*v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(
            np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))
    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )
