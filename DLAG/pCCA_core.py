import numpy as np
from scipy import linalg
from DLAG.mat2blocks import mat2blocks

class pCCA:
    params = { 'C': ['f4', 'f4'], 
               'R': ['f4', 'f4'],
               'd': ['f4', 'f4'] }
    data_cov = []
    LL = []
    
    def __init__(self, X1, X2, n_components, n_ll = 1e-8, n_iter = 1e8):
        self.data = [X1, X2]
        self.data_tol = np.vstack((X1, X2))
        self.n_components = n_components
        self.n_ll = n_ll
        self.n_iter = n_iter
    
    def initialization(self):
        self.data_cov = np.cov(self.data_tol.astype('float'), bias = True)
        scale = np.exp(2*np.sum(np.log(np.diagonal(np.linalg.cholesky(self.data_cov))))/self.data_tol.shape[0])
        C_init = np.random.randn(self.data_tol.shape[0], self.n_components)*np.sqrt(scale/self.n_components)
        R_init = [None]*2
        for i in range(2):
            R_init[i] = np.cov(self.data[i].astype('float'), bias = True)
        self.params['C'] = [C_init[range(self.data[0].shape[0]),:],C_init[self.data[0].shape[0]:,:]]
        self.params['R'] = R_init
        self.params['d'] = [np.mean(self.data_tol, axis = 1)[range(self.data[0].shape[0])],np.mean(self.data_tol, axis = 1)[range(self.data[1].shape[0])]]

    def fit(self):
        block_mask = linalg.block_diag(np.ones((self.data[0].shape[0],self.data[0].shape[0])),np.ones((self.data[1].shape[0],self.data[1].shape[0])))
        I = np.eye(self.n_components)
        R = linalg.block_diag(self.params['R'][0],self.params['R'][1])
        C = np.vstack((self.params['C'][0],self.params['C'][1]))
        cX = self.data_cov
        lli = 0
        llbase = lli
        const = -1*self.data_tol.shape[0] / 2*np.log(2*np.pi)
        N = self.data_tol.shape[1]

        for i in range(int(self.n_iter)):
            #E step
            iR = np.linalg.inv(R)
            iR = 0.5 * (iR+iR.T)
            iRC = iR.dot(C)
            Mm = iR - iRC.dot(np.linalg.inv(I + C.T.dot(iRC))).dot(iRC.T)
            beta = C.T.dot(Mm)
            cX_beta = cX.dot(beta.T)
            Exx = I - beta.dot(C) + beta.dot(cX_beta)

            #loglikelihood
            llold = lli
            ldm = np.sum(np.log(np.diagonal(np.linalg.cholesky(Mm))))
            lli = N*const + N*ldm - 0.5*N*np.sum(Mm*cX)
            print ('EM iteration {} of {} ,      ll: {}'.format(i+1, self.n_iter, lli))
        
            #M step
            C = cX_beta.dot(np.linalg.inv(Exx))
            R = cX - cX_beta.dot(C.T)
            R = 0.5 * (R+R.T)
            R = R * block_mask

            #update params
            self.params['C'] = [C[0:self.data[0].shape[0],:],C[self.data[0].shape[0]:,:]]
            self.params['R'] = mat2blocks(R, [self.data[0].shape[0],self.data[1].shape[0]])
            self.LL.append(lli)

            #check stop
            if i <= 2:
                llbase = lli
            elif lli < llold:
                print('overfit')
            elif (lli-llbase) < (1+self.n_ll)*(llold-llbase):
                break
        
        if len(self.LL)<self.n_iter:
            print('LL converged')
        else:
            print('n_iter reached')
