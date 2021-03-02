import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

############################################################################
# In this example it's shown how to perform dimensionality reduction and 
# feature extraction on a matrix X (turbo2D.csv) via Non-negative Matrix 
# Factorization (NMF).
############################################################################

# Dictionary to load the input matrix, found in .csv format, and the mesh (also in .csv)
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",

    "mesh_file_name"            : "mesh_turbo.csv",
}

# Dictionary with the instruction for the NMF algorithm:
settings = {
    #Preprocessing settings 
    "center"                    : True,
    "centering_method"          : 'min',
    "scale"                     : True,
    "scaling_method"            : "range",

    #set the reduced dimensionality
    "number_of_features"        : 8,

    #set the optimization algorithm to be used. Two are available:
    #'als' (Alternating Least Squares) and 'mur' (Multiplicative Update Rule)
    "optimization_algorithm"    : "als",
    
    #if 'als' is selected, there is also the option to add sparsity. Therefore,
    #two options for als_method are available: 'standard' (sparsity = False) and 'sparse'.
    #When the sparsity constraint is activated, eta and beta must also be set.     
    "als_method"                : "standard",
    "sparsity_eta"              : 0.1,
    "sparsity_beta"             : 0.3, 

    #the metric to assess the reconstruction error. It is possible to select either
    #'frobenius' (i.e., Frobenius distance between the original and the reconstructed matrix),
    #or "kld", which stands for: Kullback-Leibler Divergence.
    "optimization_metric"       : "frobenius",   
}


# Load the input matrix and the associated mesh
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
mesh = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["mesh_file_name"], delimiter= ',')

# Start the dimensionality reduction and the feature extraction step:
# call the NMF class and give in input X and the dictionary with the instructions
model = model_order_reduction.NMF(X, settings)

# Perform the feature extraction step
W,H = model.fit()

# Perform the clustering step via NMF
idx = model.cluster()


plt.rcParams.update({'font.size' : 12, 'text.usetex' : True})
for ii in range(0, settings["number_of_features"]):
    fig = plt.figure()
    axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
    axes.scatter(mesh[:,0], mesh[:,1], c=H.T[:,ii],alpha=0.5, cmap='gnuplot')
    axes.set_xlabel('$x\ [m]$')
    axes.set_ylabel('$y\ [m]$')
    plt.show()


fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=idx,alpha=0.5, cmap='gnuplot')
axes.set_xlabel('$x\ [m]$')
axes.set_ylabel('$y\ [m]$')
plt.show()