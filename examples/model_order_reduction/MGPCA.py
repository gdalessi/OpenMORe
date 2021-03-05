import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

############################################################################
# In this example it's shown how to reduce the model via Manifold Generated 
# Principal Component Analysis (MGPCA). 
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"                  : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")), 
    "input_file_name"               : "turbo2D.csv",

}

# Dictionary with the settings for PCA and the PV algorithm
# The data will be centered and scaled outside the class,
# thus they don't have to be processed and the options are
# set to False.

settings ={
    #centering and scaling options
    "center"                        :   False,
    "centering_method"              :   "mean",
    "scale"                         :   False,
    "scaling_method"                :   "auto",

    #set the number of PCs:
    "number_of_eigenvectors"        :   35,

    #variables selection options
    "method"                        : "b2", 
    "number_of_variables"           : 10,
    "path_to_labels"                : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "labels_name"                   : "labels.csv",

    #choose if the reduction has to be only on the chemical species
    "include_temperature"           : False,

}


# Load the input matrix and do not consider the temperature 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
if not settings["include_temperature"]:
    X = X[:,1:]

# Center and scale the input matrix
mu = center(X, "mean")
sigma = scale(X, "auto")
X_preprocessed = center_scale(X, mu, sigma)


# Create a folder for the considered training
import datetime
now = datetime.datetime.now()
try:
    newDirName = "trainingMGPCAforPV_time:_" + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)
except FileExistsError:
    newDirName = "trainingMGPCAforPV2ndtry_time" + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)


# Write a txt to recap the prescribed settings
text_file = open("Recap_training.txt", "wt")
method_used = text_file.write("The PV method is: " + str(settings["method"]) + ". \n")
minVar = text_file.write("The number of considered variables for the model reduction is: {} \n".format(settings["number_of_variables"]))
text_file.close()


# Retrieve the PVs name, and the corresponding numbers in the original matrix X
PV = model_order_reduction.variables_selection(X_preprocessed, settings)
PVs, numbers = PV.fit()
    
# Save the selected variables in a .txt, they can be useful later on
np.savetxt("Selected_variables.txt", numbers)

# Cut the dimensionality of the original matrix, considering only
# the selected 'm' variables
X_preprocessed_cut = X_preprocessed[:,numbers]


#perform the dimensionality reduction via Principal Component Analysis,
# and return the eigenvectors of the reduced manifold (PCs).
model = model_order_reduction.PCA(X_preprocessed, settings)
PCs, ____ = model.fit()


#compute the projection of the original points on the reduced
#PCA manifold, obtaining the scores matrix Z
Z = model.get_scores()


# Compute the matrix B for the MGPCA reconstruction
Bi,resid,rank,s = np.linalg.lstsq(X_preprocessed_cut,Z)

# Now we can reconstruct the original matrix from the reduced matrix X_preprocessed_cut
X_recovered = X_preprocessed_cut @ Bi @ PCs.T


# Uncenter and unscale the reconstructed matrix 
X_ = unscale(X_recovered, sigma)
X_back = uncenter(X_, mu)


# Assess the quality of the reconstruction by means of a parity plot
matplotlib.rcParams.update({'font.size' : 14, 'text.usetex' : True})
for ii in range(X.shape[1]):
    a = plt.axes(aspect='equal')
    plt.scatter(X[:,ii], X_back[:,ii], c='darkgray', label= "$MGPCA\ reconstruction$")
    plt.xlabel('$Original\ Variable\ [-]$')
    plt.ylabel('$Reconstructed\ Variable\ [-]$')
    lims = [np.min(X[:,ii]), np.max(X[:,ii])]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims, 'k', label= "$True\ value$")
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc="best")
    plt.title("$MGPCA\ reconstruction$")
    plt.savefig('Comparison_rec_error_MGPCA.png', dpi=300)
    plt.show()