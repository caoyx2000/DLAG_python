from operator import length_hint
import numpy as np
from scipy import linalg

#a function help to merge the block matrix into block diagonal matrix
def blcoks2mat(x):
    t = len(x)
    for i in range(t):
        if i == 0:
            y =x[i]
        else:
            y = linalg.block_diag(y,x[i])
    return y