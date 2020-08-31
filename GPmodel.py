# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
#import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,Matern
#import logging

class GaussianProcess:
    def __init__(self, dim):
        self.dim = dim
        self.kernel =  RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
        self.beta=1e6
        self.xValues = []
        self.yValues = []
        self.yValuesNorm=[]
        self.model = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=5)
#        self.model = GaussianProcessRegressor(kernel=self.kernel,normalize_y=True,n_restarts_optimizer=5)

        
    def fitNormal(self):
        y_mean = np.mean(self.yValues)
        y_std = self.getstd()
        self.yValuesNorm= (self.yValues - y_mean)/y_std
        self.model.fit(self.xValues, self.yValuesNorm)
    def fitModel(self):
        self.model.fit(self.xValues, self.yValues)

    
    def addSample(self, x, y):
        self.xValues.append(x)
        self.yValues.append(y)

    def getPrediction(self, x):
        mean, std = self.model.predict(x.reshape(1,-1),return_std=True)
        if std[0]==0:
            std[0]=np.sqrt(1e-5)*self.getstd()
        return mean, std
    def getmean(self):
        return np.mean(self.yValues)
    def getstd(self):
        y_std=np.std(self.yValues)
        if y_std==0:
            y_std=1
        return y_std