import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture                             # Unsupervised soft clustering (required for density estimation)
from scipy import stats
import pickle
import itertools
import copy
import pandas as pd

class LearnFEL:
    """
    This class approximates the global partition funcion with a Variational Bayesian Gaussian Mixture Model.
    Next the partition function is converted to a FEL with the eq: 
    FE = -kT*log(Global_Partition_func), k = Boltzmann const., T = Temperature in Kelvin
    """
    def __init__(self,X=None,Rng=None,grids=100,periodic=None):
        """
        X (numpy.ndarray) = Time series data, rows: time, columns: features
        Rng (numpy.ndarray) = Range of the features
        grids (int) = number of grids on 1D. For N-dimension, it would be (grids)^N. 
        Depending on dimensionality, one may reduce the grids value to avoid the insufficient memory issue.
        periodic (numpy.ndarray) = Periodicity of degrees of freedom. (1 if periodic)
        """
        self.Xarr = X
        self.maxComponents = 10
        self.LLs = None
        self.MODELs = None
        self.bestMODEL = None
        self.rfc = None
        self.rfc_error = None
        self.logL = None
        self.Posterior = None
        self.kT = 300*8.314/10**3       # kcal/mol
        self.globalRng = Rng
        self.numGrid = grids
        self.periodic = periodic
        if type(periodic) != type(None):
            if np.sum(periodic)>0:
                self.wrap_data()

    def wrap_data(self):
        """
        wrapping the data if periodicity exists in one or more degrees of freedom
        """
        xMIN,xMAX = self.globalRng.T
        en_Xarr = self._coder(X=self.Xarr,mins=xMIN,maxs=xMAX,code='encode')
        enwr_Xarr = self._wrapper(X=en_Xarr,period=self.periodic)
        self.Xarr = self._coder(X=enwr_Xarr,mins=xMIN,maxs=xMAX,code='decode')

    def _coder(self,X,mins,maxs,code):
        if code=='encode':
            sX = (X-mins)/(maxs-mins)
        elif code=='decode':
            sX = mins+X*(maxs-mins)
        return sX

    def _wrapper(self,X,period,part_scale=6):
        if np.sum(period)!=0:
            wrapLevel = 1
            rtol = 1-1/part_scale
            ltol = 0+1/part_scale
            pX = X[:,np.min(np.where(period==True)):]
            partID = np.sum((pX>rtol)+(pX<ltol),axis=1)>0
            partX = X[partID]
            idx=len(partX)*3**(np.sum(period))
            cX=wX=np.empty(shape=(0,partX.shape[1]))
            Wt=np.arange(-wrapLevel,wrapLevel+1)
            for w in Wt:
                cX=np.vstack((cX,partX+w*period))
            cX=list(cX.T.reshape(partX.shape[1],len(Wt),partX.shape[0]))
            for pair in itertools.product(*cX):
                wX=np.vstack((wX,np.array(pair).T))
            wX = wX[:idx,:]
            rtol = 1+1/part_scale
            ltol = 0-1/part_scale
            outID = np.sum(wX<ltol,axis=1)+np.sum(wX>rtol,axis=1)>0
            inID = outID==0
            wX = wX[inID]
            return np.vstack((X,wX))
        else:
            return X

    def get_bestGMM(self,MODELs,LLs):
        """
        get maximum-likelihood model from the mixtures fitted with different types of covariance.
        """
        maxLL_idx = np.argmax(LLs)
        bestMODEL = MODELs[maxLL_idx]
        self.bestMODEL = bestMODEL

    def train_GMMs(self, Xarr):
        """
        train the Variational Bayesian Gaussian Mixture Model
        """
        N_comp = self.maxComponents
        covariance_types = ['spherical', 'tied', 'diag', 'full']
        LLarr = np.zeros([len(covariance_types)])
        MODELarr = np.empty([len(covariance_types)], dtype=object)
        row = -1
        for cv_type in covariance_types:
            row += 1
            #gmm = mixture.BayesianGaussianMixture(n_components=N_comp,covariance_type=cv_type,random_state=50,weight_concentration_prior_type="dirichlet_distribution",max_iter=1500,warm_start=True)
            gmm = mixture.BayesianGaussianMixture(n_components=N_comp,covariance_type=cv_type,random_state=50,weight_concentration_prior_type="dirichlet_process",max_iter=1500,warm_start=True)
            gmm.fit(Xarr)
            LLarr[row] = gmm.score(Xarr)
            MODELarr[row] = gmm
        self.LLs = LLarr
        self.MODELs = MODELarr

    def MH_ellipsoid(self, meshX, means=None, covs=None, wts=None, cfd=0.75):
        """
        Mahalanobis ellipsoid for cutting the Gaussian Mixtures at a certain confidence interval
        """
        if type(means)==type(None):
            means=copy.deepcopy(self.bestMODEL.means_)
            covs=copy.deepcopy(self.bestMODEL.covariances_)
            wts=copy.deepcopy(self.bestMODEL.weights_)
        ellp = np.zeros((meshX.shape[0],0))
        cutoff = stats.chi2.ppf(cfd,meshX.shape[1])
        cutoff = np.sqrt(cutoff)                            # cutoff for Chi-sq distribution is Sq(cutoff for Normal distribution)
        if np.prod(covs[0].shape) == 1:
            SQdiff = (meshX-means[:,np.newaxis])**2         # Squared difference
            covin = 1/covs
            covin = covin.reshape(-1,1)
            mahalD = np.sqrt(SQdiff*covin[:,np.newaxis])    # Mahalanobis distance
            bins = mahalD<cutoff
            ellp = bins[:,:,0]
        else:
            for i in range(means.shape[0]):
                mean = means[i]
                diff = (meshX-mean)
                cov = covs[i]
                covin = np.linalg.inv(cov)
                mahalD = np.diag(np.sqrt(np.abs(np.dot(np.dot(diff,covin),diff.T))))
                bins = mahalD<cutoff
                bins = bins.reshape(-1,1)
                ellp = np.hstack((ellp,bins))
            ellp = ellp.T
        sWT = (wts>1e-2).reshape(-1,1)                  # Scaled weights
        ellp = ellp*sWT
        unionELLP = np.sum(ellp,axis=0)
        unionELLP[unionELLP>0] = 1
        return ellp,unionELLP

    def getPosterior(self, Xarr, MODEL=None):
        """
        get the Posterior distribution from the best fit GMM.
        """
        if type(MODEL)==type(None):
            PX = np.exp(self.bestMODEL.score_samples(Xarr))
            self.logL = self.bestMODEL.score_samples(Xarr)
        else:
            PX = np.exp(MODEL.score_samples(Xarr))
            self.logL = MODEL.score_samples(Xarr)
        _,uELLP = self.MH_ellipsoid(meshX=Xarr,cfd=1)       # Here we take the full GMM i.e. with 100% confidence interval
        PXcut = PX*uELLP
        PXcut[PXcut==0]=np.min(PXcut[PXcut>0])
        PXcut = PXcut/np.min(PXcut)
        self.Posterior = PXcut
        return self.Posterior

    def trigger(self, Xarr=None, saveModel=False):
        """
        runs the training and stores best GMM
        """
        if type(Xarr)==type(None):
            pass
        else:
            self.Xarr = Xarr
        self.train_GMMs(self.Xarr)
        self.get_bestGMM(MODELs=self.MODELs, LLs=self.LLs)
        if saveModel==True:
            MODELf = 'final_GMM.sav'
            pickle.dump(self.bestMODEL, open(MODELf, 'wb'))
