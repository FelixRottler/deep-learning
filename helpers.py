import numpy as np

def rand_pos(size_x,size_y):
    x = np.random.choice(size_x//10)
    y = np.random.choice(size_y//10)
    pos_x = [10*i for i in range(size_x//10)]
    pos_y = [10*i for i in range(size_y//10)]
    return pos_x[x], pos_y[y]
