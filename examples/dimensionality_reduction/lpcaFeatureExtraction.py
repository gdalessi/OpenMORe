import numpy as np
import os

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *


############################################################################
# In this example it's shown how to perform dimensionality reduction and 
# feature extraction on a matrix X (turbo2D.csv) via Local Principal Component 
# Analysis (LPCA).
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"                  :   os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"               :   "turbo2D.csv",
}

# Dictionary with the instruction for the LPCA algorithm:
settings ={
    #centering and scaling options
    "center"                        :   True,
    "centering_method"              :   "mean",
    "scale"                         :   True,
    "scaling_method"                :   "auto",

    #set the number of PCs:
    "number_of_eigenvectors"        :   5,

    #set the path to the partitioning file:
    #WARNING: the file name "idx.txt" is mandatory
    "path_to_idx"                   : file_options["path_to_file"],

    #the number of cluster where you want to plot:
    "cluster_to_plot"               :   1,

    #the local principal component you want to plot:
    "PC_to_plot"                    :   0,
}

# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

# Start the dimensionality reduction and the feature extraction step:
# call the LPCA class and give in input X and the dictionary with the instructions
model = model_order_reduction.LPCA(X, settings)

# Perform the dimensionality reduction via Local Principal Component Analysis,
# and return the local eigenvectors of the reduced manifold (LPCs), the local
# lower dimensional projection (u_scores), the local eigenvalues (Leigen) and
# the centroids for each cluster (centroids)
LPCs, u_scores, Leigen, centroids = model.fit()
X_rec_lpca = model.recover()

# Assess the reconstruction of the variables from the lower dimensional manifold
# by means of a parity plot
model.plot_parity()

# Plot the PCs in the clusters
model.plot_PCs()