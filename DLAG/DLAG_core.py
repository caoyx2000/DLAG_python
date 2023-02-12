import numpy as np
import numpy.matlib
from scipy import linalg
from DLAG.mat2blocks import mat2blocks
from DLAG.blocks2mat import blcoks2mat
from DLAG.pCCA_core import pCCA
from elephant.gpfa import gpfa_core
from DLAG.mat2blocks import mat2blocks

class DLAG:
    params = {'C':          [None]*2,
              'R':          [None]*2,
              'd':          [None]*2,
              'D':          None,
              'gamma_a':    None,
              'gamma_w':    [None]*2,
              'eps_a':      None,
              'eps_w':      [None]*2
            }
    Rs = []
    Cs = []
    ds = []
    Xsm = []
    Vsm = []
    VsmGP_across = []
    VsmGP_within = []
    seqAcross = {}
    seqWithin = [{},{}]
    LL = []


    def __init__(self, n_components_1, n_components_2, n_across_components, T, binwidth = 20, n_ll = 1e-8, n_iter = 1e8):
        self.n_Xa = n_across_components
        self.n_Xw = [n_components_1-n_across_components,n_components_2-n_across_components]
        self.n_X = self.n_Xa*2 + sum(self.n_Xw)
        self.n_ll = n_ll
        self.n_iter = n_iter
        self.binwidth = binwidth
        self.n_T = int(T/binwidth)
        self.T = T

    def initialization(self, Y1, Y2):
        self.Y = [Y1, Y2]
        self.Y_tol = np.vstack((Y1, Y2))
        self.n_trial = int(self.Y_tol.shape[1]/self.n_T)
        #initialize observation model
        print('initializing DLAG with pCCA ...')
        dinit = pCCA(self.Y[0], self.Y[1], self.n_Xa)
        dinit.initialization()
        dinit.fit()
        Ca = dinit.params['C']
        for i in range(2):
            if self.n_Xw[i] > 0:
                cY = np.cov(self.Y[i].astype('float64'))
                u, s, C_uncorr = np.linalg.svd(Ca[i].T.dot(cY))
                C_uncorr = C_uncorr.T
                C_uncorr = C_uncorr[:,self.n_Xa:(self.n_Xa+self.n_Xw[i])]
                self.params['C'][i] = np.hstack((Ca[i],C_uncorr))
            else:
                self.params['C'] = dinit.params['C']
        self.params['d'] = dinit.params['d']
        self.params['R'] = [np.diag(np.diag(dinit.params['R'][0])),np.diag(np.diag(dinit.params['R'][1]))]
        self.params['D'] = np.zeros((2,self.n_Xa))
        self.Rs = linalg.block_diag(self.params['R'][0],self.params['R'][1])
        self.Cs = linalg.block_diag(self.params['C'][0],self.params['C'][1])
        self.ds = np.vstack((self.params['d'][0].reshape(-1,1), self.params['d'][1].reshape(-1,1)))
        
        #initialize state model
        tao = 2*self.binwidth
        sigma = 1e-3
        self.params['gamma_a'] = (self.binwidth/tao)**2 * np.ones(self.n_Xa)
        self.params['eps_a'] = sigma*np.ones(self.n_Xa)
        for i in range(2):
            if self.n_Xw[i]>0:
                self.params['gamma_w'][i] = (self.binwidth/tao)**2 * np.ones(self.n_Xw[i])
                self.params['eps_w'][i] = sigma*np.ones(self.n_Xw[i])
        print('DLAG initialization finished')

    def transform(self, X1, X2):
        self.Y = [X1, X2]
        self.Y_tol = np.vstack((X1, X2))
        self.n_trial = int(self.Y_tol.shape[1]/self.n_T)
        self.LL = self.Estep()
        return self.Xsm
    
    def load(self, params):
        self.params = params
        self.Rs = linalg.block_diag(self.params['R'][0],self.params['R'][1])
        self.Cs = linalg.block_diag(self.params['C'][0],self.params['C'][1])
        self.ds = np.vstack((self.params['d'][0].reshape(-1,1), self.params['d'][1].reshape(-1,1)))

    def fit(self, saveParams = None):
        lli = 0
        llold = 0
        llbase = 0   

        print('starting to fit DLAG')
        for i in range(int(self.n_iter)):
            llold = lli
            #Estep
            lli = self.Estep()
            self.LL.append(lli)
            print('EM iteration {} of {} ,         ll: {} '.format(i+1, self.n_iter, lli))
            #Mstep
            self.Mstep()

            #check stop
            if i<2:
                llbase = lli
            elif (lli-llbase) < (1+self.n_ll)*(llold-llbase):
                break

            #save Params for each several iters
            if saveParams is not None:
                if i%saveParams == 0:
                    self.params['C'] = [self.Cs[0:self.Y[0].shape[0],0:(self.n_Xa+self.n_Xw[0])],
                                        self.Cs[self.Y[0].shape[0]:,(self.n_Xa+self.n_Xw[0]):]]
                    self.params['R'] = mat2blocks(self.Rs, [self.Y[0].shape[0],self.Y[1].shape[0]])
                    self.params['d'] = [self.ds.flatten()[0:self.Y[0].shape[0]],self.ds.flatten()[self.Y[0].shape[0]:]]
                    np.save('./params_nXa_{}_iter_{}'.format(self.n_Xa, i), self.params)
        
        if len(self.LL)<self.n_iter:
            print('LL converged')
        else:
            print('n_iter reached')

        #update params
        self.params['C'] = [self.Cs[0:self.Y[0].shape[0],0:(self.n_Xa+self.n_Xw[0])],
                            self.Cs[self.Y[0].shape[0]:,(self.n_Xa+self.n_Xw[0]):]]
        self.params['R'] = mat2blocks(self.Rs, [self.Y[0].shape[0],self.Y[1].shape[0]])
        self.params['d'] = [self.ds.flatten()[0:self.Y[0].shape[0]],self.ds.flatten()[self.Y[0].shape[0]:]]
        print('model finished training')
    
    def Mstep(self):
        #Initialize relevant variables
        minvarfrac = 0.01
        xdim_across = self.n_Xa
        xdim_within = self.n_Xw
        ydims = [self.Y[0].shape[0], self.Y[1].shape[0]]
        ydim = np.sum(ydims)
        numgroups = 2
        N = self.n_trial
        T = [self.n_T]*N
        varfloor = minvarfrac*np.diag(np.cov(self.Y_tol))
        block_idxs = [[0,ydims[0]],[ydims[0],ydims[0]+ydims[1]]]
        # Solve for C and d together
        C = np.zeros((ydim, self.n_X))
        d = np.zeros((ydim,1))
        for i in range(numgroups):
            currgroup = block_idxs[i]
            obs_idxs = slice(currgroup[0],currgroup[1])
            lat_idxs = slice(int(i*xdim_across+np.sum(xdim_within[0:i])), int((i+1)*xdim_across+np.sum(xdim_within[0:i+1])))
            sum_pauto = np.zeros((xdim_across+xdim_within[i], xdim_across+xdim_within[i]))
            sum_pauto = np.sum(self.Vsm[lat_idxs,lat_idxs,:], axis = 2)*N+self.Xsm[lat_idxs,:].dot(self.Xsm[lat_idxs,:].T)
            Y = self.Y_tol
            Y = Y[obs_idxs,:]
            Xsm = self.Xsm
            Xsm = Xsm[lat_idxs,:]
            sum_yxtrans = Y.dot(Xsm.T)
            sum_xall = np.sum(Xsm, axis = 1).reshape(-1,1)
            sum_yall = np.sum(Y, axis = 1).reshape(-1,1)
            term = np.vstack((np.hstack((sum_pauto, sum_xall)),np.hstack((sum_xall.T, np.sum(T).reshape(1,1)))))
            Cd = np.hstack((sum_yxtrans,sum_yall)).dot(np.linalg.inv(term))
            C[obs_idxs,lat_idxs] = Cd[:,0:(xdim_across+xdim_within[i])]
            d[obs_idxs,0] = Cd[:,-1]
        # Collect outputs
        self.Cs = C
        self.ds = d
        # solve for R
        Y = self.Y_tol
        Xsm = self.Xsm
        sum_yxtrans = Y.dot(Xsm.T)
        sum_xall = np.sum(Xsm, axis = 1).reshape(-1,1)
        sum_yall = np.sum(Y, axis = 1).reshape(-1,1)
        sum_yytrans = np.sum(Y**2, axis = 1).reshape(-1,1)
        yd = sum_yall*self.ds
        term = np.sum((sum_yxtrans-self.ds.dot(sum_xall.T))*self.Cs, axis = 1).reshape(-1,1)
        r = self.ds**2+(sum_yytrans-2*yd-term)/np.sum(T)
        r = np.maximum(varfloor,r.flatten())
        self.Rs = np.diag(r)
        # Learn GP kernel params and delays
        seqAcross, seqWithin = self.partitionlatents()
        res = self.learnGPparams_plusDelay(seqAcross)
        self.params['gamma_a'] = res['gamma']
        self.params['D']= res['DelayMatrix']
        # Within-group parameters: Learning these parameters is identical to learn GPFA model params
        for i in range(2):
            paramWithin = {}
            paramWithin['gamma'] = self.params['gamma_w'][i]
            paramWithin['eps'] = self.params['eps_w'][i]
            paramWithin['covType'] = 'rbf'
            paramWithin['notes'] = {}
            paramWithin['notes']['learnKernelParams'] = True
            paramWithin['notes']['learnGPNoise'] = False
            paramWithin['notes']['RforceDiagonal'] = True
            res = gpfa_core.learn_gp_params(seqWithin[i], paramWithin)
            self.params['gamma_w'][i] = res['gamma']
    
    def learnGPparams_plusDelay(self, seqAcross):
        MAXITERS = -10
        xDim = self.n_Xa
        numGroups = 2
        oldParams = np.vstack((self.params['gamma_a'], self.params['D']))
        precomp = self.makeprecomp_timescales(seqAcross)
        gamma = np.zeros(xDim)
        DelayMatrix = np.zeros((numGroups,xDim))
        for i in range(xDim):
            init_gam = np.log(oldParams[0,i])
            init_delay = oldParams[2,i]
            init_delay = -1*np.log(self.n_T/(init_delay+0.5*self.n_T)-1)
            init_p = np.array([init_gam, init_delay])
            res_p, fX, res_iters = self.minimize(init_p, MAXITERS, precomp, i)
            gamma[i] = np.exp(res_p[0])
            DelayMatrix[1,i] = self.n_T/(1+np.exp(-res_p[1])) - 0.5*self.n_T
        res = {}
        res['DelayMatrix'] = DelayMatrix
        res['gamma'] = gamma
        res['res_iters'] = res_iters
        res['fX'] = fX
        return res

    def minimize(self, init_p, MAXITERS, precomp, i):
        INT = 0.1
        EXT = 3.0
        MAX = 20
        RATIO = 10
        SIG = 0.1
        RHO = SIG/2
        red = 1
        length = MAXITERS
        ii = 0
        ls_failed = 0
        f0, df0 = self.grad_betam_delay(init_p, precomp, i)
        fX = f0
        ii = ii+(length<0)
        s = -df0
        d0 = -np.sum(s**2)
        x3 = red/(1-d0)
        while ii<np.abs(length):
            ii = ii + (length>0)
            X0 = init_p
            F0 = f0
            dF0 = df0
            if length>0:
                M = MAX
            else:
                M = min(MAX, -length-ii)
            while True:
                x2 = 0
                f2 = f0
                d2 = d0
                f3 = f0
                df3 = df0
                success = 0
                while (not success) and (M>0):
                        M = M-1
                        ii = ii+(length<0)
                        f3, df3 = self.grad_betam_delay(init_p+x3*s, precomp, i)
                        success = 1
                if f3<f0:
                    X0 = init_p+x3*s
                    F0 = f3
                    dF0 = df3
                d3 = np.sum(df3*s)
                if d3 > SIG*d0 or f3>f0+x3*RHO*d0 or M==0:
                    break
                x1 = x2
                f1 = f2
                d1 = d2
                x2 = x3
                f2 = f3
                d2 = d3
                A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)
                B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
                x3 = x1-d1*(x2-x1)**2/(B+np.sqrt(B*B-A*d1*(x2-x1)))
                if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3<0:
                    x3 = x2*EXT
                elif x3>x2*EXT:
                    x3 = x2*EXT
                elif x3<x2+INT*(x2-x1):
                    x3 = x2+INT*(x2-x1)
            while (np.abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:
                if d3 > 0 or f3 > f0+x3*RHO*d0:
                    x4 = x3
                    f4 = f3
                    d4 = d3
                else:
                    x2 = x3
                    f2 = f3
                    d2 = d3
                if f4>f0:
                    x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
                else:
                    A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)
                    B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                    x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
                if np.isnan(x3) or np.isinf(x3):
                    x3 = (x2+x4)/2
                x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))
                f3, df3 = self.grad_betam_delay(init_p+x3*s, precomp, i)
                if f3<F0:
                    X0 = init_p+x3*s
                    F0 = f3
                    dF0 = df3
                M = M-1
                ii = ii + (length<0)
                d3 = np.sum(df3*s)
            if np.abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:
                init_p = init_p+x3*s
                f0 = f3
                fX = [fX, f0]
                s = (np.sum(df3**2)-np.sum(df0*df3))/(np.sum(df0**2))*s - df3
                df0 = df3
                d3 = d0
                d0 = np.sum(df0*s)
                if d0>0:
                    s = -df0
                    d0 = -np.sum(s**2)
                x3 = x3 * min(RATIO, d3/d0)
                ls_failed = 0
            else:
                init_p = X0
                f0 = F0
                df0 = dF0
                if ls_failed or i>np.abs(length):
                    break
                s = -df0
                d0 = -np.sum(s**2)
                x3 = 1/(1-d0)
                ls_failed = 1
        return init_p, fX, ii

    def grad_betam_delay(self, p, precomp, i):
        df = np.array([0,0]).astype('float64')
        f = 0
        gamma = np.exp(p[0])
        betaall = np.array([0,p[1]]).reshape(-1,1)
        delayall = self.n_T/(1+np.exp(-betaall))-0.5*self.n_T
        T = self.n_T
        mT = 2*T
        dK_dBetak = np.zeros((mT,mT))
        delaydif = np.matlib.repmat(delayall, T, 1)
        delaydif = np.matlib.repmat(delaydif.T, mT, 1)-np.matlib.repmat(delaydif, 1, mT)
        deltaT = precomp['Tdif'][i][0:mT,0:mT] - delaydif
        deltaTsq = deltaT**2
        temp = (1-self.params['eps_a'][i])*np.exp(-gamma/2*deltaTsq)
        dtemp = gamma*temp*deltaT
        K = temp+self.params['eps_a'][i]*np.eye(mT)
        KinvPautoSUM = np.linalg.inv(K).dot(precomp['PautoSUM'][i])
        dE_dK = -0.5*(self.n_trial*np.eye(mT) - KinvPautoSUM).dot(np.linalg.inv(K))
        dK_dgamma = -0.5*temp*deltaTsq
        dE_dgamma = np.sum(dE_dK*dK_dgamma)
        df[0] = df[0] + dE_dgamma
        exp_Beta = np.exp(-betaall)
        dDelayall_dBetaall = -self.n_T*exp_Beta/((1+exp_Beta)**2)
        idx = slice(1, 2*T, 2)
        dK_dBetak[:,idx] = -dtemp[:,idx]*dDelayall_dBetaall[1]
        dK_dBetak[idx,:] = dtemp[idx,:]*dDelayall_dBetaall[1]
        dK_dBetak[idx,idx] = 0
        dE_dBetak = np.sum(dE_dK*dK_dBetak)
        df[1] = df[1]+dE_dBetak
        ans, logdet_K = np.linalg.slogdet(K)
        f = f - 0.5*self.n_trial*logdet_K -0.5*np.trace(KinvPautoSUM)
        f = -f
        df[0] = df[0]*gamma
        df = -df
        return f, df

    def makeprecomp_timescales(self, seq):
        xDim = self.n_Xa
        yDims = [self.Y[0].shape[0], self.Y[1].shape[0]]
        numGroups = 2
        Tdif = np.matlib.repmat(np.arange(1,self.n_T+1),2,1)
        Tdif = np.matlib.repmat(np.reshape(Tdif.T, (1,-1)), 2*self.n_T, 1) - np.matlib.repmat(np.reshape(Tdif.T, (1,-1)).T, 1, 2*self.n_T)
        precomp = {}
        precomp['Tdif'] = [Tdif]*xDim
        precomp['T'] = [self.n_T]*xDim
        precomp['PautoSUM'] = [np.zeros((2*self.n_T,2*self.n_T))]*xDim
        for i in range(xDim):
            for j in range(self.n_trial):
                xsm_i = seq['xsm'][j][i:xDim*numGroups:xDim,:]
                xsm_i = np.reshape(xsm_i, (1, numGroups*self.n_T), order = 'F')
                precomp['PautoSUM'][i] = precomp['PautoSUM'][i]+seq['VsmGP'][j][:,:,i]+xsm_i.T.dot(xsm_i)
        return precomp

    def partitionlatents(self):
        numgroups = 2
        N = self.n_trial
        xdim_across = self.n_Xa
        xdim_within = self.n_Xw
        seqAcross = {}
        seqWithin = [np.recarray((N,), dtype = [('latent_variable','object'), ('VsmGP', 'object'), ('T', 'int')]), np.recarray((N,), dtype = [('latent_variable','object'), ('VsmGP', 'object'), ('T', 'int')])]
        seqAcross['xsm'] = [None]*N
        seqAcross['VsmGP'] = [self.VsmGP_across]*N
        seqAcross['T'] = [self.n_T]*N
        for i in range(numgroups):
            seqWithin[i]['latent_variable'] = [None]*N
            seqWithin[i]['VsmGP'] = [self.VsmGP_within[i]]*N
            seqWithin[i]['T'] = [self.n_T]*N
        for i in range(N):
            for groupidx in range(numgroups):
                acrossIdxs_start = int(groupidx*xdim_across+np.sum(xdim_within[0:groupidx]))
                acrossIdxs_end = acrossIdxs_start+xdim_across
                if seqAcross['xsm'][i] is None:
                    seqAcross['xsm'][i] = self.Xsm[acrossIdxs_start:acrossIdxs_end, i*self.n_T:(i+1)*self.n_T]
                else:
                    seqAcross['xsm'][i] = np.vstack((seqAcross['xsm'][i],self.Xsm[acrossIdxs_start:acrossIdxs_end, i*self.n_T:(i+1)*self.n_T]))
                withinIdxs_start = acrossIdxs_end
                withinIdxs_end = withinIdxs_start+xdim_within[groupidx]
                seqWithin[groupidx]['latent_variable'][i] = self.Xsm[withinIdxs_start:withinIdxs_end, i*self.n_T:(i+1)*self.n_T]
        return seqAcross, seqWithin
    
    def Estep(self):
        q = self.Y_tol.shape[0]
        # Total number of within- and across-group latents, for all groups
        mp = self.n_X
        # Precomputations for the DLAG E-step
        Rinv, logdet_R, CRinv, CRinvC, dif, termat = self.precomp_Estep()
        dif = dif.astype('float64')
        termat = termat.astype('float64')
        lli = 0
        # Construct K_big
        k_big = self.make_k_big()
        k_big_inv = np.linalg.inv(k_big)    
        ans, logdet_k_big = np.linalg.slogdet(k_big)
        CRinvC_big = [None]*self.n_T
        for i in range(self.n_T):
            CRinvC_big[i] = CRinvC
        invM = np.linalg.inv(k_big_inv+blcoks2mat(CRinvC_big))
        ans, logdet_M = np.linalg.slogdet(k_big_inv+blcoks2mat(CRinvC_big))
        # (xDim*numGroups) X (xDim*numGroups) Posterior covariance for each timepoint
        Vsm = np.empty((mp,mp,self.n_T))
        for t in range(self.n_T):
            cidx = slice(mp*t, mp*(t+1))
            Vsm[:,:,t] = invM[cidx,cidx]
        # (numGroups*T) x (numGroups*T) Posterior covariance for each GP
        # Posterior covariance for across-group
        VsmGP_across = np.empty((2*self.n_T, 2*self.n_T, self.n_Xa))
        idxs = self.extractacrossindices()
        for i in range(self.n_Xa):
            idx = idxs[i]
            temp = invM[idx,:]
            VsmGP_across[:,:,i] = temp[:,idx]
        # Posterior covariances for within-group latents
        VsmGP_within = [None]*2
        idxs = self.extractwithinindices()
        for i in range(2):
            if self.n_Xw[i]>0:
                VsmGP_within[i] = np.empty((self.n_T,self.n_T,self.n_Xw[i]))
                for j in range(self.n_Xw[i]):
                    idx = idxs[i][j]
                    temp = invM[idx,:]
                    VsmGP_within[i][:,:,j] = temp[:,idx]
        # Compute blkProd = CRinvC_big * invM efficiently
        blkprod = np.zeros((mp*self.n_T,mp*self.n_T))
        for t in range(self.n_T):
            idx = slice(mp*t, mp*(t+1))
            blkprod[idx,:] = CRinvC.dot(invM[idx,:])
        blkprod = k_big.dot(np.eye(mp*self.n_T)-blkprod)
        xsmat = blkprod.dot(termat)
        # update X estimated/variance and GP Variance matrix
        self.Xsm = np.reshape(xsmat, (mp,self.n_trial*self.n_T), order = 'F')
        self.Vsm = Vsm
        self.VsmGP_across = VsmGP_across
        self.VsmGP_within = VsmGP_within
        #compute loglikelihood
        val = -self.n_T*logdet_R-logdet_k_big-logdet_M-q*self.n_T*np.log(2*np.pi)
        lli = self.n_trial*val-np.sum(Rinv.dot(dif)*dif)+np.sum(termat.T.dot(invM)*termat.T)
        lli = lli/2
        return lli

    def extractwithinindices(self):
        idxs = [None]*2
        for i in range(2):
            if self.n_Xw[i]>0:
                idxs[i] = [None]*self.n_Xw[i]
                for j in range(self.n_Xw[i]):
                    idxtemp = []
                    for t in range(self.n_T):
                        bigbaseidx = self.n_X*t
                        for groupidx in range(2):
                            bigidx = bigbaseidx+groupidx*self.n_Xa+sum(self.n_Xw[0:groupidx])
                            if groupidx == i:
                                idxtemp.append(bigidx+self.n_Xa+j)
                    idxs[i][j] = idxtemp
        return idxs

    def extractacrossindices(self):
        idxs = [None]*self.n_Xa
        for i in range(self.n_Xa):
            idxtemp = []
            for t in range(self.n_T):
                bigbaseidx = self.n_X*t
                for groupidx in range(2):
                    bigidx = bigbaseidx+groupidx*self.n_Xa+sum(self.n_Xw[0:groupidx])
                    idxtemp.append(bigidx+i)
            idxs[i]=idxtemp
        return idxs
    
    def precomp_Estep(self):
        mp = self.n_X
        Rinv = np.linalg.inv(self.Rs)
        Rinv = (Rinv+Rinv.T)/2
        ans, logdet_R = np.linalg.slogdet(self.Rs)
        CRinv = self.Cs.T.dot(Rinv)
        CRinvC = CRinv.dot(self.Cs)
        dif = (self.Y_tol.T - np.mean(self.Y_tol.T, axis = 0)).T
        CRinvd = CRinv.dot(dif)
        termat = np.reshape(CRinvd, (self.n_T*mp,self.n_trial), order = 'F')
        return Rinv, logdet_R, CRinv, CRinvC, dif, termat

    def make_k_big(self):
        k_big = np.zeros((self.n_T*self.n_X,self.n_T*self.n_X))
        T = self.n_T
        k_big_across = self.make_k_big_across()
        k_big_within = [None]*2
        for i in range(2):
            if self.n_Xw[i]>0:
                k_big_within[i] = self.make_k_big_within(i)
        for t1 in range(T):
            bigbaseidx1 = self.n_X*t1
            acrossbaseidx1 = (self.n_Xa*2)*t1
            for t2 in range(T):
                bigbaseidx2 = self.n_X*t2
                acrossbaseidx2 = (self.n_Xa*2)*t2
                for i1 in range(2):
                    acrossidx1 = acrossbaseidx1+i1*self.n_Xa
                    bigidx1 = bigbaseidx1+i1*self.n_Xa+sum(self.n_Xw[0:i1])
                    for i2 in range(2):
                        acrossidx2 = acrossbaseidx2+i2*self.n_Xa
                        bigidx2 = bigbaseidx2+i2*self.n_Xa+sum(self.n_Xw[0:i2])
                        k_big[bigidx1:(bigidx1+self.n_Xa), bigidx2:(bigidx2+self.n_Xa)] = k_big_across[acrossidx1:(acrossidx1+self.n_Xa), acrossidx2:(acrossidx2+self.n_Xa)]
                        if i1 == i2:
                            if self.n_Xw[i1]>0:
                                bigwithinidx1 = slice(bigidx1+self.n_Xa, bigidx1+self.n_Xa+self.n_Xw[i1])
                                bigwithinidx2 = slice(bigidx2+self.n_Xa, bigidx2+self.n_Xa+self.n_Xw[i2])
                                k_big[bigwithinidx1,bigwithinidx2] = k_big_within[i1][t1*self.n_Xw[i1]:(t1+1)*self.n_Xw[i1], t2*self.n_Xw[i1]:(t2+1)*self.n_Xw[i1]]
        return k_big

    def make_k_big_within(self, numgroup):
        xdim = self.n_Xw[numgroup]
        T = self.n_T
        k_big = np.zeros((xdim*T, xdim*T))
        tdif = np.matlib.repmat(np.arange(1,T+1).reshape(1,-1).T, 1, T) - np.matlib.repmat(np.arange(1, T+1), T, 1)
        for i in range(xdim):
            idx = slice(i, xdim*T+i, xdim)
            K = (1-self.params['eps_w'][numgroup][i])*np.exp(-self.params['gamma_w'][numgroup][i]/2*(tdif**2))+self.params['eps_w'][numgroup][i]*np.eye(T)
            k_big[idx, idx] = K
        return k_big

    def make_k_big_across(self):
        xdim = self.n_Xa
        ydim = [self.Y[0].shape[0], self.Y[1].shape[0]]
        k_big = np.zeros((xdim*2*self.n_T, xdim*2*self.n_T))
        mT = 2*self.n_T
        tdif = np.matlib.repmat(np.arange(1, self.n_T+1), 2, 1)
        tdif = np.matlib.repmat(np.reshape(tdif.T, (1,-1)), mT, 1) - np.matlib.repmat(np.reshape(tdif.T, (1,-1)).T, 1, mT)
        for i in range(xdim):
            delayall = self.params['D'][:, i:i+1]
            delaydif = np.matlib.repmat(delayall, self.n_T, 1)
            delaydif = np.matlib.repmat(delaydif.T, mT, 1) - np.matlib.repmat(delaydif, 1, mT)
            deltaT = tdif - delaydif
            deltaTsq = deltaT**2
            temp = np.exp(-0.5*self.params['gamma_a'][i]*deltaTsq)
            K_i = ((1-self.params['eps_a'][i])*temp + self.params['eps_a'][i]*np.eye(mT))
            idx = slice(i, xdim*2*self.n_T, xdim)
            k_big[idx,idx] = K_i
        return k_big
