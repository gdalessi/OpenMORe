import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

import matplotlib.pyplot as plt
import numpy as np
import os

############################################################################
# In this example it's shown how to perform dimensionality reduction and 
# feature extraction on a matrix X (turbo2D.csv) via Principal Component 
# Analysis (PCA). 

# It is similar to the other function, but in this example the reduced 
# dimensionality is set by means of the denoising function described in:
# Matan, Donoho. IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",
}

# Dictionary to load the mesh, also found in .csv format
mesh_options = {
    #set the mesh file options (the path goes up twice - it's ok)
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh_turbo.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}

# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
optimalRank = denoise(X)


# Dictionary with the instruction for the PCA algorithm:
settings ={
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_eigenvectors"    : optimalRank,
    
    #enable to plot the cumulative explained variance
    "enable_plot_variance"      : True,
}


# Start the dimensionality reduction and the feature extraction step:
# call the PCA class and give in input X and the dictionary with the instructions
model = model_order_reduction.PCA(X, settings)


# Perform the dimensionality reduction via Principal Component Analysis,
# and return the eigenvectors of the reduced manifold 
PCs = model.fit()


# Compute the projection of the original points on the reduced
# PCA manifold, obtaining the scores matrix Z
Z = model.get_scores()


# Assess the percentage of explained variance if the number of PCs has not
# been set automatically, and plot the result
model.get_explained()

print("The optimal reduced-rank for the input matrix found via denoising is: {}".format(optimalRank))