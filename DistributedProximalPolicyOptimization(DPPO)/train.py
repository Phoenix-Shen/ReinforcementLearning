
from models import GlobalAgent,  collect_data_async
import yaml
import torch.multiprocessing as mp

if __name__ == "__main__":
    # print your settings
    with open("./DistributedProximalPolicyOptimization(DPPO)/settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("###############YOUR SETTINGS#################")
    for key in args.keys():
        print(f"{key}->{args[key]}")
    # global actor and critic, it will be used for training
    agent = GlobalAgent(args)

    process_num = args["n_threads"]
    # pipe for data transform
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
                     for pipe1, pipe2 in (mp.Pipe(),))

    child_process_list = []
    # child process for collecting data
    for i in range(process_num):
        process = mp.Process(target=collect_data_async,
                             args=(pipe_dict[i][1], args))
        child_process_list.append(process)

    [pipe_dict[i][0].send(agent.actor.cpu_state_dict())
     for i in range(process_num)]
    [p.start() for p in child_process_list]
    # main loop of the training procedure
    for episode in range(args["max_episode"]):
        buffer_list = list()
        # get data under the current policy
        for i in range(process_num):
            receive_data = pipe_dict[i][0].recv()
            buffer_list.append(receive_data)
        # learn from the batch data
        agent.learn(buffer_list)
        [pipe_dict[i][0].send(agent.actor.cpu_state_dict())
         for i in range(process_num)]
        # call save_model function
        save_fre = args["save_frequency"]
        eval_fre = args["eval_frequency"]
        if (episode+1) % save_fre == 0:
            agent.save_model()
        if (episode+1) % eval_fre == 0:
            rewards = agent.eval()
            max_ep = args["max_episode"]
            print(f"episode[{episode}/{max_ep}],rewards:{rewards}")
            agent.writer.add_scalar("rewards_eval", rewards, episode)
    # finally turn off the threads
    [p.terminate() for p in child_process_list]
