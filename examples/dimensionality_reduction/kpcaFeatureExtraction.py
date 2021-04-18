import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

import matplotlib.pyplot as plt
import numpy as np
import os

############################################################################
# In this example it's shown how to perform dimensionality reduction and 
# feature extraction on a matrix X (moons.csv) via Kernel Principal Component 
# Analysis (KPCA).
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/dummy_data/")),
    "input_file_name"           : "moons.csv",
}


# Dictionary with the instruction for the KPCA algorithm:
settings ={
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_eigenvectors"    : 2,
    
    #set the kernel type
    "selected_kernel"           : "rbf",
    #set the sigma parameter if rbf is selected as kernel
    "sigma"                     : 1,

    #the following two, if set to True, allow to use approximation
    #algorithms to speed-up Kernel PCA. Suggested for large X
    "use_Nystrom"               : False,
    "fast_SVD"                  : False,
}


# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])


# Start the dimensionality reduction and the feature extraction step:
# call the KPCA class and give in input X and the dictionary with the instructions
model = model_order_reduction.KPCA(X, settings)


# Perform the dimensionality reduction via Principal Component Analysis,
# and return the eigenvectors of the reduced manifold 
U, V, Sigma = model.fit()

#Plot the first two KPCs
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(U[:,0], U[:,1])
axes.set_xlabel('$First\ PC\ [-]$')
axes.set_ylabel('$Second\ PC\ [-]$')
plt.show()