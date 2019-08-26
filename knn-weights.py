#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=load_iris()
X=iris['data']
y=iris['target']
X_train,X_test,y_train,y_test=train_test_split(X,y)


class KNN:
    def __init__(self,k=3, weights=None):
        self.k=k
        self.weights=weights
        
    def fit(self,X,y):
        self.train=np.hstack((X,y.reshape(X.shape[0],1)))
    
    def get_nb(self,x):
        nb=[]
        for obs in self.train:
            nb.append((obs[-1],np.linalg.norm(x-obs[:-1])))
        return nb
    
    def vote(self,nb):
        nb.sort(key=lambda x: x[1])
        nbs=nb[:self.k]
        if self.weights=='distance':
            weighted=list(map(lambda x: (x[0],1/(x[1]+0.01)), nbs))
        else:
            weighted=list(map(lambda x: (x[0],1),nbs))
        #weighted.sort(key=lambda x: x[1])
        candidates=set(map(lambda x: x[0],weighted))
        res=[]
        for c in candidates:
            w=0
            for n in weighted:
                 if n[0]==c:
                        w+=n[1]
            res.append((c,w))
        res.sort(key=lambda x: x[1],reverse=True)
        return res[0][0]
    
    def predict(self,X):
        ypred=[]
        for x in X:
            ypred.append(self.vote(self.get_nb(x)))
        return ypred
        
knn=KNN(k=2,weights='distance')

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

accuracy_score(y_test,ypred)



goodknn=KNeighborsClassifier(weights='distance')
goodknn.fit(X_train,y_train)
goodpred=goodknn.predict(X_test)
accuracy_score(y_test,goodpred)






