import model
import memory
import gym

import torch as t
import numpy as np
import torch.nn as nn
import scipy.optimize
import math


class AGENT():
    def __init__(self,
                 n_features,
                 n_actions,
                 buffer_size,
                 batch_size,
                 max_epoch,
                 env_name,
                 render=False,
                 reward_decay=0.99,
                 tau=0.95) -> None:

        self.n_features = n_features
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.render = render
        self.reward_decay = reward_decay
        self.tau = tau
        self.env = gym.make(env_name)

        self.replay_buffer = memory.Memory()
        self.actor = model.Actor(n_features, n_actions)
        self.critic = model.Critic(n_features)
        # 搞不懂这两个是在干什么，暂时注释掉，看看效果
        # self.running_state=running_state.ZFilter((self.n_features,),clip=5)
        # self.running_reward=running_state.ZFilter((1,),demean=False,clip=10)

    def collect_samples(self, render=False):
        num_steps = 0
        mem = memory.Memory()
        total_rewards = 0
        min_reward = 1e6
        max_reward = -1e6
        log = dict()
        while num_steps < self.batch_size:
            state = self.env.reset()
            reward_episode = 0
            while True:
                state = t.FloatTensor(state).unsqueeze(0)
                action_tensor = self.actor.choose_action(state)
                next_state, reward, done, _ = self.env.step(
                    action_tensor.item())

                reward_episode += reward
                mem.push(state, action_tensor.item(),
                         int(done), next_state, reward)
                if render:
                    self.env.render()
                if done:
                    break
                state = next_state
                num_steps += 1
            total_rewards += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
        log["num_steps"] = num_steps
        log["total_reward"] = total_rewards
        log["min_reward"] = min_reward
        log["max_reward"] = max_reward
        log["avg_reward"] = total_rewards/num_steps
        return mem.sample(), log

    def learn(self):

        for i_episode in range(self.max_epoch):
            batch, log = self.collect_samples(render=False)
            states = t.FloatTensor(np.stack(batch.state))
            actions = t.FloatTensor(
                np.stack(batch.action))
            next_states = t.FloatTensor(
                np.stack(batch.next_state))
            dones = t.FloatTensor(np.stack(batch.done))
            rewards = t.FloatTensor(np.stack(batch.reward))

            with t.no_grad():
                values = self.critic(states)

            # ESTIMATE ADVANTAGES
            deltas = t.zeros((rewards.size(0), 1))
            advantage = t.zeros((rewards.size(0), 1))
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(rewards.size(0))):
                deltas[i] = rewards[i]+self.reward_decay * \
                    prev_value*dones[i]-values[i]
                advantage[i] = deltas[i]+self.reward_decay * \
                    self.tau*prev_advantage*dones[i]
                prev_value = values[i, 0]
                prev_advantage = advantage[i, 0]
            returns = values+advantage
            advantage = (advantage - advantage.mean()) / advantage.std()
            trpo_step(self.actor, self.critic, states, actions,
                      returns, advantage, 1e-2, 1e-2, 1e-3)


def trpo_step(policy_net: model.Actor, value_net: model.Critic, states, actions, returns, advantages, max_kl, damping, l2_reg, use_fim=True):
    """update critic"""
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, t.tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred-returns).pow(2).mean()

        for param in value_net.parameters():
            value_loss += param.pow(2).sum()*l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()
    params = get_flat_params_from(value_net).detach().cpu().numpy()
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss, params,
        maxiter=25)
    set_flat_params_to(value_net, t.tensor(flat_params))

    with t.no_grad():
        fixed_log_probs = policy_net.get_log_probability(states, actions)

    def get_loss(volatile=False):
        with t.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_probability(states, actions)
            action_loss = -advantages*t.exp(log_probs-fixed_log_probs)
            return action_loss.item()

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else set(
            [info['std_id']])

        t_ = t.ones(mu.size(), requires_grad=True, device=mu.device)
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(
        ), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = t.autograd.grad(Jtv, t_)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(
        ), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * \
                v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(states)
        kl = kl.mean()

        grads = t.autograd.grad(
            kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = t.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = t.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = t.cat(
            [grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    Fvp = Fvp_fim if use_fim else Fvp_direct

    loss = get_loss()
    grads = t.autograd.grad(loss, policy_net.parameters())
    loss_grad = t.cat([grad.view(-1) for grad in grads]).detach()
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(
        policy_net, get_loss, prev_params, fullstep, expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = t.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(t.zeros(param.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = t.cat(grads)
    return flat_grad


def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = t.autograd.grad(
        output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(t.zeros(param.view(-1).shape,
                             device=param.device, dtype=param.dtype))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = t.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = t.zeros(b.size(), device=b.device)
    r = b.clone()
    p = b.clone()
    rdotr = t.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / t.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = t.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x
