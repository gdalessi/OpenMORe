'''
MODULE: operations.py

@Authors: 
    G. D'Alessio [1,2], G. Aversano [1], A. Parente[1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Additional notes:
    This cose is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

__all__ = ["get_cluster", "get_centroids", "fitPCA", "check_sanity_int", "check_sanity_NaN"]

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# -----------------
# Functions
# -----------------
def get_cluster(X, idx, index):
    positions = np.where(idx == index)
    cluster = X[positions]
    return cluster

def get_centroids(X):
    centroid = np.mean(X, axis = 0)
    return centroid

def fitPCA(X, n_eig):
   
    pca = PCA(n_components = n_eig)
    pca.fit(X)
    eigenvec = pca.components_
    return np.array(eigenvec)

def check_sanity_int(kappa):
    if isinstance(kappa, int) == True:
        return kappa
    else:
        raise Exception("The number of cluster and/or eigenvectors input must be integers. Please provide a valid input.")
        exit()

def check_sanity_NaN(X):
    if X.isna().values.any() == False:
        return X
    else:
        raise Exception("The input matrix contains NaN values. Please double-check your input.")
        exit()