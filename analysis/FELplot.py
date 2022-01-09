import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib import cm
from matplotlib import rc
import sys

N = 100
args=sys.argv
outF=args[-1]
mat = []
for i in range(1,len(args)-1):
    fileN = args[i]
    x,y,z = np.load(fileN).T
    mat.append(z)

mat = np.array(mat)
z1 = np.mean(mat,axis=0)
z2 = np.std(mat,axis=0)
meanFE = np.vstack((x,y,z1)).T
stdvFE = np.vstack((x,y,z2)).T
np.savez(outF,mean_FEL=meanFE,stdv_FEL=stdvFE)
exit()

L = int(np.sqrt(len(x)))

X1=x.reshape(L,L)
X2=y.reshape(L,L)
FE=z.reshape(L,L)

plt.figure(dpi=100)
contour = plt.contour(X1, X2, FE, colors='k', labels=10)
contour_filled = plt.contourf(X1, X2, FE, cmap=cm.inferno)
cbar=plt.colorbar(contour_filled, aspect=20)
plt.xlabel(r'$CV_1$', fontsize=14)
plt.ylabel(r'$CV_2$', fontsize=14)
cbar.set_label('Free Energy (kcal/mol)')
ax = plt.gca()
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
ax.spines["top"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax1=cbar.ax
ax1.tick_params(axis='y', colors='black')
cbar.outline.set_linewidth(2)
plt.clabel(contour, manual=True, rightside_up=True, colors = 'k', fmt = '%.1f', fontsize=11)
plt.show()
