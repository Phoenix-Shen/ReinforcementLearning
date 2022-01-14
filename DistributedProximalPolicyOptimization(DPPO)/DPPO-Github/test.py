import multiprocessing
from multiprocessing import queues
import time

def main():
    pipe1, pipe2 = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=child,args=(pipe2,))
    p1.start()
    data = pipe1.recv()
    print(data)

def child(pipe):
    time.sleep(5)
    pipe.send((1,2))


if __name__ == "__main__":
    main()
