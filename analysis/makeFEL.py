import numpy as np
import matplotlib.pyplot as plt
import GMModel as gmm
from matplotlib import cm
import sys
from scipy.cluster.vq import kmeans2

args = sys.argv
inFile = args[-2]
outFile = args[-1]

MIX = np.load(inFile)

fitM = gmm.LearnFEL(MIX)
fitM.trigger()

XX=np.linspace(0,11,100)
YY=np.linspace(0,11,100)
meshX=np.array(np.meshgrid(XX,YY)).T
meshX = meshX.reshape(meshX.shape[0]**meshX.shape[-1],meshX.shape[-1])

PGX = fitM.bestMODEL.predict_proba(meshX)
PX = fitM.getPosterior(meshX)
PX = PX/np.sum(PX)
PXG = PGX.T*PX
PXG = PXG.T/np.sum(PXG,axis=1)

means = fitM.bestMODEL.means_
centroid, label = kmeans2(means, 3)

print("Centroids are\n",centroid)
BiasIn=input("Enter Biases seperated with Comma: ")
BiasIn=BiasIn.replace(' ','')
Bias=[float(x) for x in BiasIn.split(',')]
wts = fitM.bestMODEL.weights_
for i in range(len(centroid)):
    wts[label==i] = Bias[i]/np.sum(wts[label==i])

wts[wts<1e-2]=0
rwtPX = np.sum(PXG*wts,axis=1)

X,Y = meshX.T
kB = 8.314/(4.2*1000)       # Boltzmann constant in unit kcal/mol/K
T = 300                     # Temperature in Kelvin (K)
kT = kB*T
FE = -kT*np.log(rwtPX)
FE = FE - np.min(FE)

MAT = np.array([X.ravel(),Y.ravel(),FE.ravel()]).T
np.save(outFile,MAT)
