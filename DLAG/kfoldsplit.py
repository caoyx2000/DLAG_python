import numpy as np
from sklearn.model_selection import train_test_split

#define a class to help us split the datasets to K fold
class kfoldsplit:
    
    #X should be formed as n_trials*n_variables
    def __init__(self, X, n_fold):
        self.data = X
        self.n_fold = n_fold
        self.datasets = list(range(n_fold))
    
    #split the data into n_fold
    def split(self):
        X = self.data
        for i in range(0,self.n_fold-1):
            X, Y = train_test_split(X, test_size = 1/(self.n_fold-i), random_state = None)
            self.datasets[i] = Y
            self.datasets[i+1] = X
        return self.datasets
    
    #choose one fold as test set, the remains will be train sets
    def fold(self, num):
        train_set = np.empty((0,self.data.shape[1]))
        self.datasets = np.array(self.datasets, dtype=object)
        if (num == 1):
            for i in range(num,self.n_fold):
                train_set = np.vstack((train_set, self.datasets[i]))
        elif (num == self.n_fold):
            for i in range(0, num-1):
                train_set = np.vstack((train_set, self.datasets[i]))
        else:
            for i in range(0, num-1):
                train_set = np.vstack((train_set, self.datasets[i]))
            for i in range(num, self.n_fold):
                train_set = np.vstack((train_set, self.datasets[i]))
        return train_set, self.datasets[num-1]