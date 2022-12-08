from multiprocessing import Process
import multiprocessing
import numpy as np
import time
import matplotlib.pyplot as plt

global a

a = []

def func1():
    start_time = time.time()
    print('func1: starting')
    current_time = time.time()
    while (current_time - start_time) < 15:
        a.append(np.random.randint(1,100))
    print('func1: finishing')

def func2():
    start_time = time.time()
    print('func2: starting')
    current_time = time.time()
    while (current_time - start_time) < 15:
        b = a.copy()
        plt.plot(np.asarray(b))
        plt.show()
        time.sleep(2)
    
    print('func2: finishing')

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p1.join()
    p2.join()