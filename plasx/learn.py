import os

import numpy as np

import scipy

def lr_transform_base(coef, intercept, X, prob=False):
    coef = np.array(coef)
    intercept = np.array(intercept)

    assert np.squeeze(coef).ndim==1 and intercept.ndim==0

    X_eff = (X.dot(coef.reshape(-1,1)) + intercept).flatten()
    if prob:
        from scipy.special import expit
        Y_hat = expit(X_eff)
        return Y_hat
    else:
        return X_eff

def lr_transform(lr, X, prob=False):
    lr_transform_base(lr.coef_, lr.intercept_, X, prob=prob)

