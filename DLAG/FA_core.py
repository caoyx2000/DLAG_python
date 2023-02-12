import numpy as np
from sklearn.decomposition import FactorAnalysis
from DLAG.kfoldsplit import kfoldsplit

#fit the FA from one component to n_iter components
#each fitting will test with n_fold cross validation
#the logliklihood score will be returned for each n_components

#X:                         n_trials*n_variables matrix data
#n_fold:                    how many split do you want to have in cross validation
#n_iter:                    max num of latent variables used to fit the FA model
#fix_split:                 fix the split or not

def score_FA_cv(X, n_fold, n_iter, fix_split):
    CVL = np.zeros(n_iter)
    print ('confirming latent variables number with FA')
    if fix_split:
        kfv = kfoldsplit(X = X, n_fold = n_fold)
        kfv.split()
        for i in range(0,n_iter):
            score_temp = np.zeros(n_fold)
            for j in range(0,n_fold):
                x_train, x_test = kfv.fold(j+1)
                FA = FactorAnalysis(n_components = i+1)
                FA.fit(x_train)
                score_temp[j] = FA.score(x_test)
            CVL[i] = np.mean(score_temp)
            print ('working n_component =', i+1)
    else:
        for i in range(0,n_iter):
            kfv = kfoldsplit(X = X, n_fold = n_fold)
            kfv.split()
            score_temp = np.zeros(n_fold)
            for j in range(0,n_fold):
                x_train, x_test = kfv.fold(j+1)
                FA = FactorAnalysis(n_components = i+1)
                FA.fit(x_train)
                score_temp[j] = FA.score(x_test)
            CVL[i] = np.mean(score_temp)
            print ('working n_component =', i+1)
        
    return CVL