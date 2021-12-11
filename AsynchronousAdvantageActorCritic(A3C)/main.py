import gym
import model
import torch.multiprocessing as mp
import os
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    env = gym.make('CartPole-v0')
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n

    gnet = model.Net(N_S, N_A)
    # https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-multiprocessing/
    gnet.share_memory()

    opt = model.SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))

    global_ep, global_ep_r, res_queue = mp.Value(
        "i", 0), mp.Value('d', 0), mp.Queue()

    # PARALLEL TRAINING

    workers = [model.Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, N_S, N_A)
               for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
