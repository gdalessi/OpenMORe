import numpy as np
import os

import OpenMORe.model_order_reduction as mor 
from OpenMORe.utilities import *

from sklearn.metrics.pairwise import rbf_kernel

file_options = {
    #set the training matrix file options
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")), 
    "input_file_name"           : "turbo2D.csv",
}

settings = {
    #centering and scaling options
    "center"                        :   True,
    "centering_method"              :   "mean",
    "scale"                         :   True,
    "scaling_method"                :   "auto",
    
    #mandatory input settings below:
    "kernel_type"               : "rbf",    #only rbf is working for now
    "number_to_pick"            : 100,
    "rank"                      : 30,
     
    #optional input settings below:
    "number_of_matrices"        : 10,       #for ensemble
    "sigma"                     : 1,        #sigma for rbf

    "polynomial_degree"         : 2,        #poly d
    "polynomial_freeParameter"  : 1,        #poly c
    
    "nu_matern"                 : 1,        #matern nu
    "rho_matern"                : 2,        #matern rho
    "sigma_matern"              : 5,        #matern sigma

}

# Load the data:
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
X = X[:2000,:]

print("Dimensions of the input matrix: {}".format(X.shape))

# Compute the approx Kernel matrix with the defined settings:
# Call to the class: the input matrix and the settings are required to make it work.
model = mor.Kernel_approximation(X, settings)

# Choose the algorithm to compute K 
Kapprox = model.Nystrom_standard()     # Nystrom std algorithm 
#Kapprox = model.Nystrom_ensemble()     # Nystrom ensemble algorithm
#Kapprox = model.QRdecomposition()       # QR decomposition formulation


# Center and scale:
X_tilde = center_scale(X, center(X, settings["centering_method"]), scale(X, settings["scaling_method"]))
# Compute the exact Kernel matrix via sklearn:
X_features = rbf_kernel(X_tilde, gamma=1/settings["sigma"])

# Check for the kernel dimensions:
print("Dimensions of K via OpenMORe: {}".format(Kapprox.shape))
print("Dimensions of K via sklearn: {}".format(X_features.shape))

# Compute the error between the full and the approximated kernels
error = np.linalg.norm(Kapprox-X_features,'fro')/np.linalg.norm(X_features,'fro')

print("The relative error is: {}".format(error))