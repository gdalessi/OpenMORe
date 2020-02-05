'''
MODULE: preprocessing.py

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

__all__ = ["center", "scale", "center_scale"]

import numpy as np



# -----------------
# Functions
# -----------------
def center(X, method):
    # Main
    if method == 'MEAN' or method == 'mean' or method == 'Mean':
        mu = np.mean(X, axis = 0)
    elif method == 'MIN' or method == 'min' or method == 'Min':
        mu = np.min(X, axis = 0)
    return mu


def scale(X, method):
    # Main
    if method == 'AUTO' or method == 'auto' or method == 'Auto':
        sig = np.std(X, axis = 0)
    elif method == 'PARETO' or method == 'pareto' or method == 'Pareto':
        sig = np.sqrt(np.std(X, axis = 0))
    elif method == 'VAST' or method == 'vast' or method == 'Vast':
        variances = np.var(X, axis = 0)
        means = np.mean(X, axis = 0)
        sig = variances / means
    elif method == 'RANGE' or method == 'range' or method == 'Range':
        maxima = np.max(X, axis = 0)
        minima = np.min(X, axis = 0)
        sig = maxima - minima
    return sig


def center_scale(X, mu, sig):
    TOL = 1E-10
    # Main
    X0 = X - mu
    X0 = X0 / (sig + TOL)
    return X0
