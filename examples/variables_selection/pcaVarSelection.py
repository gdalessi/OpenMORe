import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

#######################################################################################
# In this example it's shown how to perform feature selection via PCA
#######################################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",
}

# Dictionary with the instructions for PCA feature selection class:
settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #variables selection options
    "method"                    : "procrustes",
    "number_of_eigenvectors"    : 8,
    "number_of_variables"       : 25,
    "path_to_labels"            : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "labels_name"               : "labels.csv",
    "include_temperature"       : False
}

# Load the input matrix:
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

# In this case, we only want to select the species. Therefore, we exlude T
if not settings["include_temperature"]: 
    X = X[:,1:]

# Select the Principal Variables (i.e., perform the feature selection step)
# and print the results
PVs = model_order_reduction.variables_selection(X, settings)
labels, numbers = PVs.fit()

print(labels)