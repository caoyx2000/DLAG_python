import numpy as np

#help to change the matrix into several blocks in diagonal
#X: data
#blcok_idxs: list of each blocks' dimension
def mat2blocks (X, block_idxs):
    X_blocks = []
    Ns = 0
    Ne = 0
    for i in range(len(block_idxs)):
        if i > 0:
            Ns = Ns+block_idxs[i-1]
        Ne = Ne+block_idxs[i]
        X_blocks.append(X[Ns:Ne,Ns:Ne])
    
    return X_blocks