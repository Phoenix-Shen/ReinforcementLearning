import torch as t
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gym
import arguments
import running_filter
import os
from torch.distributions.normal import Normal
# actor 要输出两个数值sigema和mu，所以在layer上面不能用一个sequential代替
# 还是得写两个类，分别调用forward函数


class Actor(nn.Module):
    def __init__(self, n_features, n_actions) -> None:
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.action_mean = nn.Linear(256, n_actions)
        self.sigma_log = nn.Parameter(t.zeros(1, n_actions))

    def forward(self, state: t.Tensor) -> tuple[Tensor, Tensor]:
        features = self.feature_layer(state)
        mean = self.action_mean(features)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = t.exp(sigma_log)
        pi = (mean, sigma)
        return pi


class Critic(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(n_features),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1))

    def forward(self, state: Tensor) -> Tensor:
        value = self.value_net(state)
        return value


class ActorCritic(nn.Module):
    def __init__(self, n_features, n_actions) -> None:
        super().__init__()
        self.actor = Actor(n_features, n_actions)
        self.critic = Critic(n_features)

    def forward(self, state: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        state_value = self.critic(state)
        pi = self.actor(state)
        return state_value, pi


# 集成算法的类
class trpo_agent(object):
    def __init__(self, env: gym.Env, args: arguments.ARGS) -> None:
        super().__init__()
        self.env = env
        self.args = args
        # define the actor critic network
        self.net = ActorCritic(
            self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.old_net = ActorCritic(
            self.env.observation_space.shape[0], self.env.action_space.shape[0])
        # make sure the old net and the current net have the same params
        self.old_net.load_state_dict(self.net.state_dict())
        # define the optimizer, the updation progress of the actor won't use optimizer
        # in other word we use t.autograd to compute the gradients of the actor
        self.optimzer = optim.Adam(
            self.net.critic.parameters(), lr=self.args.lr)
        # define the running mean filter
        self.running_state = running_filter.ZFilter(
            (self.env.observation_space.shape[0],), clip=5)
        # operations about saving the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir+self.args.env_name+"/"
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def learn(self):
        num_updates = self.args.total_timesteps//self.args.nsteps
        obs = self.running_state(self.env.reset())
        final_reward = 0
        episode_reward = 0
        dones = False

        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            for step in range(self.args.nsteps):
                with t.no_grad():
                    obs_tensor = t.tensor(
                        obs, dtype=t.float32).unsqueeze(0)
                    value, pi = self.net.forward(obs_tensor)
                # select actons
                mean, std = pi
                normal_dist = Normal(mean, std)
                actions = normal_dist.sample().detach().numpy().squeeze()
                # store informations
                mb_obs.append(np.copy(obs))
                mb_actions.append(actions)
                mb_dones.append(dones)
                mb_values.append(value.detach().numpy.squeeze())
                # start to execute actions in the environment
                obs_, reward, done, _ = self.env.step(actions)
                dones = done
                mb_rewards.append(reward)
                if done:
                    obs_ = self.env.reset()
                # subsititute the old observation
                obs = self.running_state(obs_)
                episode_reward += reward
                mask = 0.0 if done else 1.0
                # if reach the final state then empty the final_reward
                final_reward *= mask
                # add episode reward
                final_reward += (1-mask)*episode_reward
                episode_reward *= mask
            # to process the rollouts, convert list to ndarray
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.float32)
            mb_values = np.asarray(mb_values, dtype=np.float32)

            # compute the last state value
            with t.no_grad():
                obs_tensor = t.tensor(mb_obs, dtype=t.float32).unsqueeze(0)
                last_value, _ = self.net.forward(obs_tensor)
                last_value = last_value.detach().numpy().squeeze()

            # compute the advantages, which means this action is better or worse than the AVERAGE
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgarlam = 0

            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps-1:
                    nextnonterminal = 1.0-dones
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0-mb_dones[t+1]
                    nextvalues = mb_values[t+1]
                delta = mb_rewards[t]+self.args.gamma * \
                    nextvalues*nextnonterminal-mb_values[t]
                mb_advs[t] = lastgarlam = delta+self.args.gamma * \
                    self.args.tau*nextnonterminal*lastgarlam
            mb_returns = mb_advs+mb_values
            # normalize the advantages
            mb_advs = (mb_advs-mb_advs.mean())/(mb_advs.std()+1e-5)
            # before the update, make the old network has the parameter of the current network
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            policy_loss, value_loss = self._update_network(
                mb_obs, mb_actions, mb_returns, mb_advs)

    def _update_network(self, mb_obs, mb_actions, mb_returns, mb_advs):
        # convert ndarrays to FloatTensor
        mb_obs_tensor = t.tensor(mb_obs, dtype=t.float32)
        mb_actions_tensor = t.tensor(mb_actions, dtype=t.float32)
        mb_returns_tensor = t.tensor(mb_returns, dtype=t.float32).unsqueeze(1)
        mb_adv_tensor = t.tensor(mb_advs, dtype=t.float32).unsqueeze(1)
        # try to get the old policy and the current policy
        values, _ = self.net.forward(mb_obs_tensor)
        with t.no_grad():
            _, pi_old = self.old_net(mb_obs_tensor)
        # get the surrogate loss
        surr_loss = self._get_surrogate_loss(
            mb_obs_tensor, mb_adv_tensor, mb_actions_tensor, pi_old)
        # compute the surrogate gradient
        surr_grad = t.autograd.grad(surr_loss, self.net.actor.parameters())
        flat_surr_grad = t.cat([grad.view(-1) for grad in surr_grad]).detach()
        # use the conjugated gradient to calculate the scaled direction vector
        nature_grad = self._conjugated_gradient(
            self._fisher_vector_product, -flat_surr_grad, 10, mb_obs_tensor, pi_old)
        # calculate the scaleing ration
        non_scale_kl = 0.5 * (nature_grad*self._fisher_vector_product(
            nature_grad, mb_obs_tensor, pi_old)).sum(0, keepdim=True)
        scale_retio = t.sqrt(non_scale_kl/self.args.max_kl)
        final_nature_grad = nature_grad/scale_retio[0]
        # calculate the cxped imporovement rate/
        expected_improve = (-flat_surr_grad*nature_grad).sum(0,
                                                             keepdim=True)/scale_retio[0]
        # get the flat param ...
        prev_params = t.cat([param.data.view(-1)
                             for param in self.net.actor.parameters()])
        # start to do the line search
        success, new_params = self._line_search(self.net.actor, self._get_surrogate_loss, prev_params, final_nature_grad,
                                                expected_improve, mb_obs_tensor, mb_adv_tensor, mb_actions_tensor, pi_old)
        self._set_flat_params_to(self.net.actor, new_params)
        # then trying to update the critic network
        inds = np.arange(mb_obs.shape[0])
        for _ in range(self.args.vf_itrs):
            np.random.shuffle(inds)
            for start in range(0, mb_obs.shape[0], self.args.batch_size):
                end = start + self.args.batch_size
                mbinds = inds[start:end]
                mini_obs = mb_obs[mbinds]
                mini_returns = mb_returns[mbinds]
                # put things in the tensor
                mini_obs = t.tensor(mini_obs, dtype=t.float32)
                mini_returns = t.tensor(
                    mini_returns, dtype=t.float32).unsqueeze(1)
                values, _ = self.net(mini_obs)
                v_loss = (mini_returns - values).pow(2).mean()
                self.optimizer.zero_grad()
                v_loss.backward()
                self.optimizer.step()
        return surr_loss.item(), v_loss.item()

    def _get_surrogate_loss(self, mb_obs_tensor: Tensor, mb_adv_tensor: Tensor, mb_actions_tensor: Tensor, pi_old: Tensor) -> Tensor:
        _, pi = self.net.forward(mb_obs_tensor)
        log_prob = self._eval_actions(pi, mb_actions_tensor)
        old_log_prob = self._eval_actions(pi_old, mb_actions_tensor).detach()
        surr_loss = -t.exp(log_prob-old_log_prob)*mb_adv_tensor
        return surr_loss.mean()

    def _eval_actions(self, pi: tuple[Tensor, Tensor], actions: Tensor) -> Tensor:
        mean, std = pi
        normal_dist = Normal(mean, std)
        return normal_dist.log_prob(actions).sum(dim=1, keepdim=True)

    def _fisher_vector_product(self, v, obs, pi_old):
        kl = self._get_kl(obs, pi_old)
        kl = kl.mean()
        # start to calculate the second order gradient of the kl
        kl_grads = t.autograd.grad(
            kl, self.net.actor.parameters(), create_graph=True)
        flat_kl_grads = t.cat([grad.view(-1) for grad in kl_grads])
        kl_v = (flat_kl_grads*t.autograd.Variable(v)).sum()
        kl_sceond_grads = t.autograd.grad(kl_v, self.net.actor.parameters())
        flat_kl_secont_grads = t.cat(
            [grad.contiguous().view(-1) for grad in kl_sceond_grads]).detach()
        flat_kl_secont_grads = flat_kl_secont_grads+self.args.damping*v
        return flat_kl_secont_grads
    # compute kl divergence between two distributions

    def _get_kl(self, obs, pi_old) -> Tensor:
        mean_old, std_old = pi_old
        _, pi = self.net.forward(obs)
        mean, std = pi
        # start to calculate kl divergence
        kl = - t.log(std/std_old)+(std.pow(2) +
                                   (mean-mean_old).pow(2))/(2*std_old.pow(2))-0.5
        return kl.sum(1, keepdim=True)

    def _conjugated_gradient(self, fvp, b, update_steps, obs, pi_old, residual_tol=1e-10):
        # the initial solution is zero
        x = t.zeros(b.size(), dtype=t.float32)
        r = b.clone()
        p = b.clone()
        rdotr = t.dot(r, r)
        for i in range(update_steps):
            fv_product = fvp(p, obs, pi_old)
            alpha = rdotr / t.dot(p, fv_product)
            x = x + alpha * p
            r = r - alpha * fv_product
            new_rdotr = t.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            # if less than residual tot.. break
            if rdotr < residual_tol:
                break
        return x
    # line search

    def _line_search(self, model, loss_fn, x, full_step, expected_rate, obs, adv, actions, pi_old, max_backtracks=10, accept_ratio=0.1):
        fval = loss_fn(obs, adv, actions, pi_old).data
        for (_n_backtracks, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * full_step
            self._set_flat_params_to(model, xnew)
            new_fval = loss_fn(obs, adv, actions, pi_old).data
            actual_improve = fval - new_fval
            expected_improve = expected_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew
        return False, x

    def _set_flat_params_to(self, model, flat_params):
        prev_indx = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_indx:prev_indx + flat_size].view(param.size()))
            prev_indx += flat_size
