import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.vq import kmeans2
from tqdm import tqdm

def rewt(data,means,weights,method='random_forest',features=None,labels=None,bias=None,num_clusters=None):
    if method=='random_forest':
        if type(labels)!=type(None) and type(features)!=type(None) and type(bias)!=type(None):
            wt_DATA = random_forest_rewt(Data=data,Means=means,Weights=weights,Features=features,Labels=labels,Bias=bias)
        else:
            print("Following attributes are required for random-forest based reweighting:\n1. features\n2. labels\n3. bias\n")
            exit()
    elif method=='k_means_cluster':
        if type(num_clusters)!=type(None):
            wt_DATA = k_means_clustering_rewt(Data=data,Means=means,Weights=weights,Num_Clusters=num_clusters)
        else:
            print("Following attribute is required for k-means clustering based reweighting:\n1. num_clusters\n")
            exit()
    return wt_DATA

def random_forest_rewt(Data,Means,Weights,Features,Labels,Bias,test_split=0.2,rf_estimators=1000):
    TRAIN = np.hstack((Features,Labels.reshape(-1,1)))
    np.random.shuffle(TRAIN)
    testlen = int(test_split*(TRAIN.shape[0]))
    X_train = TRAIN[testlen:,:2]
    Y_train = TRAIN[testlen:,2]
    X_test = TRAIN[:testlen,:2]
    Y_test = TRAIN[:testlen,2]
    del TRAIN
    rfc = RandomForestClassifier(n_estimators=rf_estimators)
    rfc = RF_train(rfc,X_train,Y_train)
    err = RF_test(rfc,X_test,Y_test)
    print(err)
    wts = Weights
    pred_lbl = rfc.predict(Means)
    for i in range(len(np.unique(Labels))):
        wts[pred_lbl==i] = Bias[i]/np.sum(wts[pred_lbl==i])
    #wts[wts<1e-2]=0
    rwtDATA = np.sum(Data*wts,axis=1)
    return rwtDATA

def RF_train(model,TRAINx,TRAINy):
    print("Random forest training...\n")
    datlen = TRAINx.shape[0]
    full_itr = datlen//1000
    extra_rm = datlen%1000
    for i in tqdm(range(full_itr)):
        partX = TRAINx[i*1000:(i+1)*1000,:]
        partY = TRAINy[i*1000:(i+1)*1000]
        model.fit(partX,partY)
    if extra_rm!=0:
        partX = TRAINx[full_itr*1000:full_itr*1000+extra_rm,:]
        partY = TRAINy[full_itr*1000:full_itr*1000+extra_rm]
        model.fit(partX,partY)
    return model

def RF_test(model,TESTx,TESTy):
    print("Random forest testing...\n")
    tot_class = len(np.unique(TESTy))
    rfc_error = np.zeros((tot_class,tot_class))
    datlen = TESTx.shape[0]
    full_itr = datlen//1000
    extra_rm = datlen%1000
    for i in tqdm(range(full_itr)):
        partX = TESTx[i*1000:(i+1)*1000,:]
        partY = TESTy[i*1000:(i+1)*1000]
        predY = model.predict(partX)
        Err = pd.crosstab(partY,predY).to_numpy()
        rfc_error += Err
    if extra_rm!=0:
        partX = TESTx[full_itr*1000:full_itr*1000+extra_rm,:]
        partY = TESTy[full_itr*1000:full_itr*1000+extra_rm]
        predY = model.predict(partX)
        Err = pd.crosstab(partY,predY).to_numpy()
        rfc_error += Err
        rfc_error = self.rfc_error/np.sum(self.rfc_error)
    return rfc_error

def k_means_clustering_rewt(Data,Means,Weights,Num_Clusters):
    centroid,label = kmeans2(Means,Num_Clusters)
    print("Centroids of the k-means clusters are\n",centroid)
    BiasIn = input("Enter Biases seperated with Comma: ")
    BiasIn = BiasIn.replace(' ','')
    Bias = [float(x) for x in BiasIn.split(',')]
    wts = Weights
    for i in range(len(centroid)):
        wts[label==i] = Bias[i]/np.sum(wts[label==i])
    #wts[wts<1e-2]=0
    rwtDATA = np.sum(Data*wts,axis=1)
    return rwtDATA
