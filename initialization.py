'''
MODULE: initialization.py

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

__all__ = ["initialize_clusters"]

import numpy as np
from sklearn.cluster import KMeans



# -----------------
# Functions
# -----------------
def initialize_clusters(X, method, k):

    if method == 'RANDOM' or method == 'random' or method == 'Random':
        idx = np.random.random_integers(1, k, size=(X.shape[0], 1))
    elif method == 'KMEANS' or method == 'kmeans' or method == 'Kmeans':
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        idx = kmeans.labels_
    return idx
