
from models import GlobalAgent, LocalActor, collect_data_async
import yaml
import torch.multiprocessing as mp

if __name__ == "__main__":
    with open("./DistributedProximalPolicyOptimization(DPPO)/settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("###############YOUR SETTINGS#################")
    for key in args.keys():
        print(f"{key}->{args[key]}")

    agent = GlobalAgent(args)

    process_num = args["n_thrads"]

    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
                     for pipe1, pipe2 in (mp.Pipe(),))

    child_process_list = []

    for i in range(process_num):
        process = mp.Process(target=collect_data_async,
                             args=(pipe_dict[i][1],))
        child_process_list.append(process)

    [pipe_dict[i][0].send(agent.actor.state_dict())
     for i in range(process_num)]
    [p.start() for p in child_process_list]

    for episode in range(args["max_episode"]):
        buffer_list = list()
        for i in range(process_num):
            receive_data = pipe_dict[i][0].recv()
            buffer_list.append(receive_data)
        agent.learn(buffer_list)

    [p.terminate() for p in child_process_list]
