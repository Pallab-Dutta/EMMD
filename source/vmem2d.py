#vmem
"""
vmem
========================

Author  -   Pallab Dutta

Date    -   22 Dec 2019

========================

Provides
    1.  Probability Density Function along the Angular Coordinate you input
    2.  Free Energy Landscape in room Temperature [300K], along the same axis

How to use the documentation
----------------------------
Documentation is available only in docstrings provided with the code

I recommend exploring the docstrings using `IPython <https://ipython.org>`_,
an advanced Python shell with TAB-completion and introspection capabilities.
See below for further instructions.

I assume that `vmem` has been imported as `vm`::
        >>> import vmem as vm

Code snippets are indicated by three greater-than signs::
        >>> a=1
        >>> print "Hello World!"

Use the built-in ``help`` function to view a class's docstring::
        >>> help(vm.VMmix1D)

Brief Overview
--------------
This Module can tackle 1D clustering over Circular Coordinate.
The famous Gaussian/Normal distribution in specially defined for such 
periodic coordinate system -- named as Von Mises Distribution.
This program takes a Single-Column datafile and number of Clusters as Input.
For most of the cases Cluster Number is simply the number of Stable States 
along your Collective Variable [CV]. However features inside each such Stable
State are not considered as different clusters. Rather they are sub clusters.
To maximize fitting and to avoid data overflow error, This program uses 4 Von
Mises distribution per stable state. Therefore for example of you have Cis-Trans
isomerization and hence two stable states, You provide numCluster as 2. The pro-
gram takes 2*4=8 total Von Mises distribution to get a best fit.
Next it calculates the Free Energy [FE] according to the following formula:
    FE = -kB*T*ln(Probability)  |   kB=Boltzmann Const. T=Room Temperature

Examples
--------

For example this folder has a test.dat file. Which is basically a omega-angle data
for the peptide AcProNH2 [N-Acetyl L-Proline Amide]. It has 2 clusters. Let'see how
to get the Free Energy Curve for this...

1.  Start with importing the module:
    >>> import vmem as vm

2.  Create a VM mixture object:
    >>> # Entries are datafile followed by number of Cluster
    >>> fe=vm.VMmix1D("test.dat",2)     # 2 Clusters for test.dat

3.  Expectation Maximization:
    >>> fe.run()

4.  Re-arrange the means, kappas and weights of Von Mises distributions::
    >>> # Left-End can be found from sampling.py
    >>> # leftEnd = Sampling1D.leftBar
    >>> fe.reArrange(leftEnd)
    >>> # Check the Centroid positions of Clusters
    >>> print(fe.Centroids)

5.  According to Centroid Position order, provide the bias list. Biases are relative 
    probability of stable state. For our case relative probabilities of Cis and Trans
    are 0.01 and 0.99 respectively. Centroids are at 0 rad and either at pi or -pi rad.
    Case-1:
    >>> print(fe.Centroids)
    >>> [0,pi]
    >>> # Then biases would be
    >>> bias=[0.01,0.99]    # [for Cis, for Trans]
    
    Case-2:
    >>> print(fe.Centroids)
    >>> [-pi,0]
    >>> # Then biases would be
    >>> bias=[0.99,0.01]    # [for Trans, for Cis]

6.  Re-weighting the Clusters in the basis of the bias list::
    >>> fe.reWeight(bias)

7.  Get the Free Energy array::
    >>> FE=fe.getFE()

8.  For plotting extract the X-axis from `fe`::
    >>> X=fe.Xrange

9.  Plot with matplotlib::
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(X,FE)

More detailed instructions are provided with the docstrings under class and methods.

"""
#import modules
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from scipy.stats import vonmises
from scipy.optimize import fsolve
from scipy.cluster.vq import kmeans2
import os

#Core Class
class VMmix2D:
	"""
	Runs Expectation Maximization [Soft Clustering] Algorithm 
	in the mixture of Von Mises distributions [Gaussian Distributions 
	for Circular Coordinate like an angle]. For each cluster it fits 4 normal
	distributions to get a good fit. So when you put number of Clusters is 2,
	it takes total 2*4=8 normal distributions to start with.
	Finally it returns you the Free Energy Values in Room Temperature.

	INPUTs:	
		1a. datafile with Single column
		1b. number of Clusters [Groups] into which the whole data will be separated
		
		2.  bias array: List of relative probabilities of the Clusters

	HOW TO RUN:
		>>> import vmem as vm
		>>> fe = vm.VMmix1D('data.dat',2)	# for 2 cluster case
		>>> fe.run()				# Expectation Minimization Run
		>>> # leftEnd = sampling.Sampling1D.leftBar
		>>> fe.reArrange(leftEnd)
		>>> # Check the order of Centroids of Clusters 
		>>> print(fe.Centroids)			# returns Centroids' List in ascending order

		Now according to the order of centroid positions, you have to provide the bias list.
		For example if we have PRO residue, then we will get one centroid at Cis [0 rad] and
		another at Trans [pi or -pi rad]. Depending on them their can be two cases:

		Case-1 | The Centroid list is:
		>>> [0, pi]				# [Cis centroid, Trans centroid]
		>>> bias=[0.1, 0.9]			# [relative Probability of Cis, relative Probability of Trans]

		Case-2 | The Centroid list is:
		>>> [-pi, 0]				# [Trans centroid, Cis centroid]
		>>> bias=[0.9, 0.1]			# [relative Probability of Trans, relative Probability of Cis]

		After this step we do a reweighting over the clusters:
		>>> fe.reWeight(bias)
		
		To get the Free-Energy Landscape:
		>>> FE=fe.getFE()

		For plotting purpose following three attributes might be important:
		>>> X=fe.Xrange				# Xrange for FE
		>>> FE=fe.FE				# Same as FE=vm.getFE()
		>>> pdf=fe.fitProb			# Fitted Probability Density Values

		While plotting do:
		>>> import matplotlib.pyplot as plt	# import matplotlib as usual
		>>> plt.plot(X,pdf)			# For plotting Probability Density Function vs X-axis
		>>> plt.plot(X,FE)			# For plotting Free Energy Landscape vs X-axis

	You can check the methods help sections for more details.		

	"""
	def __init__(self, datafile, numCluster):
		if type(datafile)==str:
			self.data = np.loadtxt(datafile)				# 1D data
		else:
			self.data = datafile						# 1D data
		self.cData = np.array([np.cos(self.data), np.sin(self.data)]).T		# Circular data
		self.numCluster = numCluster						# Number of Clusters
		self.numDist = numCluster*1						# Number of Von Mises: 2 per Cluster for Good-Fit
		self.muMatrix = np.zeros([self.numDist,2])
		self.kMatrix = np.zeros([self.numDist,3])
		self.wtMatrix = np.zeros([self.numDist,1])
		self.converged = False							# Convergence of Fitting
		self.counter = 0							# Counter till convergence
		self.LL = []								# Log Likelihood
		self.TS = []								# Time Scale
		self.Centroids = None							# Cluster Centers [for k-means]
		self.Labels = []							# Cluster Labels [for k-means]
		self.Xrange = np.linspace(-np.pi, np.pi, 100)				# Angle values from -pi to pi
		self.fitProb = np.zeros([len(self.Xrange),len(self.Xrange)])		# Fitted Probability with EM
		self.FE = None								# Free Energy [FE] values from -pi to pi
		self.evolve = []							# Record the evolution of contour

	def vonMises(self, mu1, mu2, k1, k2, k3, wt, x1=0, x2=0):
		"""
		Original Von Mises probability density function [pdf] taken from
		scipy module.
			>>> from scipy.stats import vonmises

		Von Mises is just the Normal/Gaussian Distribution defined for
		circular axis/ periodic axis like angle. As a normal pdf takes a
		'mu' and 'sigma' as parameters, von mises takes 'mu' and 'kappa'.
		This 'mu' is equivalent to that of normal distribution.
		The 'kappa' is however equivalent to 1/(sigma^2).

		"""
		if type(x1)==np.ndarray and type(x2)==np.ndarray:
			n=np.exp(k1*np.cos(x1-mu1)+k2*np.cos(x2-mu2)+k3*np.cos(x1-mu1-x2+mu2))
			d=4*np.pi**2*(iv(0,k1)*iv(0,k2)*iv(0,k3)+2*sum([iv(order,k1)*iv(order,k2)*iv(order,k3) for order in range(1,1001)]))
			pdf=wt*n/d
			return pdf
			#return wt*np.exp(k*np.cos(Xdata-mu))/(2*np.pi*iv(0,k))
			#return wt*vonmises.pdf(Xdata, k, mu)
		else:
			x1=self.data[:,0]
			x2=self.data[:,1]
			n=np.exp(k1*np.cos(x1-mu1)+k2*np.cos(x2-mu2)+k3*np.cos(x1-mu1-x2+mu2))
			d=4*np.pi**2*(iv(0,k1)*iv(0,k2)*iv(0,k3)+2*sum([iv(order,k1)*iv(order,k2)*iv(order,k3) for order in range(1,1001)]))
			pdf=wt*n/d
			return pdf
			#return wt*np.exp(k*np.cos(self.data-mu))/(2*np.pi*iv(0,k))
			#return wt*vonmises.pdf(self.data, k, mu)

	def Norm(self,k1,k2,k3,deriv):
		"""
		Returns derivative of normalization factor w.r.t. k1, k2, k3 respectively. We need it to update kappa values.

		"""
		d=iv(0,k1)*iv(0,k2)*iv(0,k3)+2*sum([iv(order,k1)*iv(order,k2)*iv(order,k3) for order in range(1,1001)])
		if deriv=="k1":
			n=iv(1,k1)*iv(0,k2)*iv(0,k3)+sum([(iv(order+1,k1)+iv(order-1,k1))*iv(order,k2)*iv(order,k3) for order in range(1,1001)])
			normd=n/d
		elif deriv=="k2":
			n=iv(0,k1)*iv(1,k2)*iv(0,k3)+sum([iv(order,k1)*(iv(order+1,k2)+iv(order-1,k2))*iv(order,k3) for order in range(1,1001)])
			normd=n/d	
		elif deriv=="k3":
			n=iv(0,k1)*iv(0,k2)*iv(1,k3)+sum([iv(order,k1)*iv(order,k2)*(iv(order+1,k3)+iv(order-1,k3)) for order in range(1,1001)])
			normd=n/d
		return normd

	def initiate(self):
		"""
		Initialization of Means, Kappas and Weights for each Von Mises
		distributions in the whole mixture.
		Means are choosen in equally spaced way.
		Kappas are kept equal for all.
		Weights were also kept equal.

		"""
		Xrange = [-np.pi, np.pi]
		x_min = float(Xrange[0])
		x_max = float(Xrange[1])
		x_len = x_max-x_min
		x_step = x_len/float(self.numDist)
		gauss_start = x_min+0.5*x_step
		#k=np.random.random(self.numDist)
		for i in range(self.numDist):
			m=gauss_start+i*x_step
			wt=1/float(self.numDist)
			self.muMatrix[i,:]=m
			self.kMatrix[i,:]=np.random.random(3)
			self.wtMatrix[i,:]=wt

	def check_convergence(self, List):
		"""
		Check whether standard deviation for last 10 entries of
		log Likelihood has crossed a threshold or not. If this
		condition is satisfied we stop Expectation Maximization.	

		"""
		cutoff=np.linalg.norm(List[-10:])*10**-3
		measure=np.std(List[-10:])
		if measure < cutoff:
			self.converged = True

	def EMaximize(self):
		"""
		This part calculates the probabilities of each Von Mises given 
		that the represents the data which is done by first calculating 
		the probability of each datum that it comes from a Von Mises i.e. 
		P(datum|vonMises). Next by Bayesian statistics we get the 
		probability of each Von Mises that it represents the whole data 
		set i.e. P(vonMises|data).
		
		"""
		avg_mean=[]
		avg_sd=[]
		avg_wt=[]
		numDist=self.numDist
		data=self.data
		lenData=len(data[:,0])
		circular_data=self.cData
		while self.converged!=True:
			#print('Running...')
			self.totalDist()
			self.evolve.append(self.fitProb)
			Pxg=np.zeros([lenData,numDist])
			Pgx=np.zeros([lenData,numDist])
			PxgLog=np.zeros([lenData,1])
			new_mean=[]
			new_k=[]
			new_wt=[]
			new_logL=[]
			for i in range(numDist):
				x1=self.data[:,0]
				x2=self.data[:,1]
				m1=self.muMatrix[i,0]
				m2=self.muMatrix[i,1]
				k1=self.kMatrix[i,0]
				k2=self.kMatrix[i,1]
				k3=self.kMatrix[i,2]
				wt=self.wtMatrix[i,0]
				prob_xg=self.vonMises(m1,m2,k1,k2,k3,wt)
				#logL=self.logLikelihood(m1,m2,k1,k2,k3,wt)
				Pxg[:,i]=prob_xg
				#PxgLog[:,i]=logL
			for j in range(lenData):
				tot=sum(Pxg[j,:])
				prob_gx=(Pxg[j,:])/tot
				Pgx[j,:]=prob_gx
			# update mean standard deviation and weights of each gaussians in the gaussian-mixture
			# they are updated using the probability P(gaussian|data)
			for i in range(numDist):
				k1,k2,k3=self.kMatrix[i,:]
				m1,m2=self.muMatrix[i,:]
				wt=self.wtMatrix[i,0]
				self.wtMatrix[i,0]=(sum(Pgx[:,i])/float(lenData))
				nu1=k1*np.sin(x1)+k3*np.sin(x1-x2+m2)
				du1=k1*np.cos(x1)+k3*np.cos(x1-x2+m2)
				mu1=np.arctan(sum(Pgx[:,i]*nu1)/sum(Pgx[:,i]*du1))
				nu2=k2*np.sin(x2)-k3*np.sin(x1-x2-mu1)
				du2=k2*np.cos(x2)+k3*np.cos(x1-x2-mu1)
				mu2=np.arctan(sum(Pgx[:,i]*nu2)/sum(Pgx[:,i]*du2))
				self.muMatrix[i,:]=mu1,mu2
				RHS1 = sum(Pgx[:,i]*np.cos(x1-mu1))/sum(Pgx[:,i])
				RHS2 = sum(Pgx[:,i]*np.cos(x2-mu2))/sum(Pgx[:,i])
				RHS3 = sum(Pgx[:,i]*np.cos(x1-mu1-x2+mu2))/sum(Pgx[:,i])
				func1 = lambda k1: self.Norm(k1,k2,k3,"k1")-RHS1
				func2 = lambda k2: self.Norm(k1,k2,k3,"k2")-RHS2
				func3 = lambda k3: self.Norm(k1,k2,k3,"k3")-RHS3
				kap1 = fsolve(func1, 0)
				kap2 = fsolve(func2, 0)
				kap3 = fsolve(func3, 0)
				self.kMatrix[i,:]=kap1,kap2,kap3
				#PxgLog[:,i]=PxgLog[:,i]*Pgx[:,i]
			
			PxgLog[:,0]=np.log(np.sum(Pxg,axis=1))
			logLike=np.sum(PxgLog)
			self.LL.append(logLike)	
			self.counter+=1
			self.TS.append(self.counter)
			if self.counter>20:
				self.check_convergence(self.LL)
			
		for i in range(numDist):
			if self.kMatrix[i,0]<0:
				if self.muMatrix[i,0]<0:
					self.muMatrix[i,0]+=np.pi
				else:
					self.muMatrix[i,0]+=-np.pi
				self.kMatrix[i,0]*=-1 #abs(self.kMatrix[i,0])
			if self.kMatrix[i,1]<0:
				if self.muMatrix[i,1]<0:
					self.muMatrix[i,1]+=np.pi
				else:
					self.muMatrix[i,1]+=-np.pi
				self.kMatrix[i,1]*=-1 #abs(self.kMatrix[i,1])
			if -self.muMatrix[i,0]+self.muMatrix[i,1] > -np.pi and -self.muMatrix[i,0]+self.muMatrix[i,1] < 0:
				self.kMatrix[i,2]*=-1

	def likelihood(self, data):
		"""
		Returns the individual likelihood values for each datapoints.

		"""
		likelihood = np.zeros((data.shape[0],1))
		#print(likelihood.shape)
		x1=data[:,0]
		x2=data[:,1]
		mu1=self.muMatrix[:,0]
		mu2=self.muMatrix[:,1]
		k1=self.kMatrix[:,0]
		k2=self.kMatrix[:,1]
		k3=self.kMatrix[:,2]
		wt=self.wtMatrix[:,0]
		#print("This much")
		for i in range(len(mu1)):
			#print(self.vonMises(mu1[i],mu2[i],k1[i],k2[i],k3[i],wt[i],x1,x2).shape)
			likelihood = likelihood + self.vonMises(mu1[i],mu2[i],k1[i],k2[i],k3[i],wt[i],x1,x2).reshape(-1,1)
		return likelihood

	def log_likelihood(self, data):
		"""
		Returns the total log likelihood for a given dataset.

		"""
		LL=np.sum(np.log(self.likelihood(data=data)))
		return LL
					
	def totalDist(self):
		"""
		Returns the total probability density values along the Xaxis.
		Xaxis = [-pi, pi]

		"""
		self.fitProb = np.zeros([len(self.Xrange),len(self.Xrange)])
		X1=self.Xrange
		X2=self.Xrange
		x1,x2=np.meshgrid(X1,X2)
		mu1=self.muMatrix[:,0]
		mu2=self.muMatrix[:,1]
		k1=self.kMatrix[:,0]
		k2=self.kMatrix[:,1]
		k3=self.kMatrix[:,2]
		wt=self.wtMatrix[:,0]
		for i in range(len(mu1)):
			self.fitProb = self.fitProb + (self.vonMises(mu1[i],mu2[i],k1[i],k2[i],k3[i],wt[i],x1,x2))

	def reArrange(self, rightEnd1=None, rightEnd2=None):
		"""
		Mean List is arranged according to ascending order. kList and wtList are
		arranged according to the order of muList. leftEnd is the maximum CV got
		from traj1.nc: Periodicity only matters < leftEnd. Therefore if some cluster
		means are at +pi end and some other at -pi end, being the same identity their
		sign differs and when we start kMeans clustering it might think them as elements
		of different clusters. So, initially we add 2*pi with the -pi < elements < leftEnd.
		Thus all of them goes to +pi side and hence becomes members of same cluster.

		Purpose of Doing k-Means here is to track the individual Distributions under each
		stable states. Basically you group those distributions which are under the same
		stable state. During multiply the biasing probability to each stable states, we need
		to multiply that with individual distribution under that stable state. But direct
		multiplication will not work. From the weights which we get after Expectation Maxi-
		mization, we can calculate relative weight for each j-th distribution under i-th 
		stable state. Relative weight = wt(i,j)/[SUM_over_j {wt(i,j)}]

		"""
		mu1 = self.muMatrix[:,0]
		mu2 = self.muMatrix[:,1]
		k1 = self.kMatrix[:,0]
		k2 = self.kMatrix[:,1]
		k3 = self.kMatrix[:,2]
		wt = self.wtMatrix[:,0]
		order = np.argsort(mu1)
		self.muMatrix[:,0] = mu1[order]
		self.muMatrix[:,1] = mu2[order]
		self.kMatrix[:,0] = k1[order]
		self.kMatrix[:,1] = k2[order]
		self.kMatrix[:,2] = k3[order]
		self.wtMatrix[:,0] = wt[order]
		if rightEnd1!=None:
			for i in range(self.numDist):
				if self.muMatrix[i,0]>rightEnd1:
					self.muMatrix[i,0]-=2*np.pi
		if rightEnd2!=None:
			for i in range(self.numDist):
				if self.muMatrix[i,1]<rightEnd2:
					self.muMatrix[i,1]-=2*np.pi
		typCount=1
		while typCount!=self.numCluster:
			typCount=1
			centroid, label = kmeans2(self.muMatrix, self.numCluster)
			self.Centroids = np.sort(centroid)
			self.Labels = []
			iniGrp=0
			labLen=len(label)
			for i in range(labLen):
				self.Labels.append(iniGrp)
				if i!=labLen-1:
					if label[i+1]!=label[i]:
						iniGrp+=1
						typCount+=1
		self.Labels = np.array(self.Labels)
		#print(label,self.Labels)

	def reWeight(self, bias):
		"""
		bias is a python list.
		Relative Probabilities of different Clusters are the elements of bias.
		Before reWeighting check the order of Cluster centroid values by::
			>>> print(self.Centroids)
		
		Then put the relative probability values in bias, maintaing that order.
		For example for a Cis-Trans transition of PRO, you might have:
		1. Centroids At [0.0, +pi] then the bias would be [bias of cis, bias of trans]
		2. Centroids At [-pi, 0.0] then the bias would be [bias of trans, bias of cis]

		"""
		label = self.Labels
		for i in range(self.numCluster):
			self.wtMatrix[label==i]=self.wtMatrix[label==i]/sum(self.wtMatrix[label==i])*bias[i]

	def prob2nrg(self):
		"""
		Converts Probability to Free Energy, via the following formula:
		
		Free Energy = -kB*T*ln(Probability) kcal/mol

		"""
		kB=8.314/(4.2*1000)	# Boltzmann const. Unit = kcal/mol
		T=300			# Room Temperature. Unit = K
		FreeNrg=-kB*T*np.log(self.fitProb)
		
		self.FE=FreeNrg-np.min(FreeNrg)

	def getFE(self):
		"""
		Returns Free Energy Array along Xaxis.
		Xaxis = [-pi, pi]

		"""
		self.totalDist()
		self.prob2nrg()
		return self.FE

	def bic(self):
		"""
		BIC: Bayesian Information Crieteria

		"""
		log_likelihood=self.log_likelihood(self.data)
		#log_likelihood=np.log(self.fitProb).sum()
		num_parameters=self.numDist*6-1
		num_data=self.data.shape[0]
		#print(num_data)
		BIC=-2*log_likelihood+num_parameters*np.log(num_data)
		return BIC
	
	def aic(self):
		pass

	def trigger(self):
		"""
		Runs Expectation maximization method after initialization.		

		"""
		self.initiate()
		self.EMaximize()
		self.totalDist()

def main():
	datFile="C.dat"
	#ref="FE0.dat"
	vm=VMmix1D(datFile, 2)
	vm.run()
	vm.reArrange(2.0)
	Bias=[0.1,0.9]
	vm.reWeight([0.1,0.9])
	X=vm.Xrange
	Z=vm.getFE()
	fitMatrix=np.array([X,Z]).T
	np.savetxt("fitted.dat", fitMatrix, fmt="%.3f")

	#Xref=ref[:,0]
	#Yref=ref[:,1]-min(ref[:,1])

	plt.xlim(-np.pi,np.pi)
	#plt.plot(Xref,Yref,label="ABF (reference)", linewidth=2, color='deepskyblue')
	plt.plot(X,Z, label="Our Method", linewidth=2, color='black')
	ax1 = plt.gca()
	ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=2, fontsize=20)
	ax1.tick_params(axis='x', colors='black', width=3, labelsize=15)
	ax1.tick_params(axis='y', colors='black', width=3, labelsize=15)
	ax1.set_xlabel('Dihedral Angle: Omega [degree]', color='black', fontsize=20)
	ax1.set_ylabel('Free Energy [KCal/mol]', color='black', fontsize=20)
	ax1.spines["left"].set_linewidth(2)
	ax1.spines["right"].set_linewidth(2)
	ax1.spines["top"].set_linewidth(2)
	ax1.spines["bottom"].set_linewidth(2)
	plt.grid(True)
	#plt.show()
	plt.savefig("fitted.jpg")

if __name__ == '__main__':
	main()
