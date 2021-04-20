import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

import matplotlib 
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
    "true_labels_fileName"      : "moonsLabels.csv"
}


# Dictionary with the instruction for the KPCA algorithm:
settings ={
    #centering and scaling options
    "center"                    : False,
    "centering_method"          : "mean",
    "scale"                     : False,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_eigenvectors"    : 2,
    
    #set the kernel type
    "selected_kernel"           : "rbf",
    #set the sigma parameter if rbf is selected as kernel
    "sigma"                     : 0.03,

    #the following two, if set to True, allow to use approximation
    #algorithms to speed-up Kernel PCA. Suggested for large X
    "use_Nystrom"               : False,
    "fast_SVD"                  : True,
    "eigensFast"                : 50,   #parameter to set the number of eigens used in the fastSVD algorithm
}


# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
id_rows = readCSV(file_options["path_to_file"], file_options["true_labels_fileName"])


# Start the dimensionality reduction and the feature extraction step:
# call the KPCA class and give in input X and the dictionary with the instructions
model = model_order_reduction.KPCA(X, settings)


# Perform the dimensionality reduction via Principal Component Analysis,
# and return the eigenvectors of the reduced manifold 
U, V, Sigma = model.fit()

#Plot the data in the original input space
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(X[:,0], X[:,1], c=id_rows)
axes.set_xlabel('$x\ [-]$')
axes.set_ylabel('$y\ [-]$')
plt.show()


#Plot the first two KPCs
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(U[:,0], U[:,1], c=id_rows)
axes.set_xlabel('$First\ KPC\ [-]$')
axes.set_ylabel('$Second\ KPC\ [-]$')
plt.show()