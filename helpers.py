import numpy as np

def rand_pos(size_x,size_y):
    x = np.random.choice(size_x//1)
    y = np.random.choice(size_y//1)
    pos_x = [1*i for i in range(size_x//1)]
    pos_y = [1*i for i in range(size_y//1)]
    return pos_x[x], pos_y[y]
