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

__all__ = ["get_cluster", "get_centroids", "fitPCA"]

import numpy as np
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