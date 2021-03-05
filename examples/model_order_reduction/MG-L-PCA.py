import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

############################################################################
# In this example it's shown how to reduce the model via Manifold Generated 
# Local Principal Component Analysis (MG-L-PCA). 
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

    #set the path to the partitioning file:
    #WARNING: the file name "idx.txt" is mandatory
    "path_to_idx"                   : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),

}

# Load the input matrix and do not consider the temperature 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
if not settings["include_temperature"]:
    X = X[:,1:]

# Load the solution vector from the clustering algorithm
idx_clustering = np.genfromtxt(settings["path_to_idx"] + '/idx.txt')


# Center and scale the input matrix
mu = center(X, "mean")
sigma = scale(X, "auto")
X_preprocessed = center_scale(X, mu, sigma)


# Create a folder for the considered training
import datetime
now = datetime.datetime.now()
try:
    newDirName = "trainingMG-L-PCAforPV_time:_" + now.strftime("%Y_%m_%d-%H%M")
    os.mkdir(newDirName)
    os.chdir(newDirName)
except FileExistsError:
    newDirName = "trainingMG-L-PCAforPV2ndtry_time" + now.strftime("%Y_%m_%d-%H%M")
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

# Apply Local Principal Component Analysis to build the reconstruction matrices
model = model_order_reduction.LPCA(X_preprocessed, settings)
LPCs, u_scores, Leigen, centroids = model.fit()

# Allocate the list to store the reconstruction matrices and allocate the memory
# for the matrix reconstructed from the reduced space
Bi = [None] * len(LPCs)
X_recovered = np.empty(X_preprocessed.shape, dtype=float)


for ii in range(0, len(LPCs)):
    Aq = LPCs[ii]

    # Cluster from the full input matrix
    cluster = get_cluster(X_preprocessed, idx_clustering, ii)
    print("cluster dimensions: {}".format(cluster.shape))

    # Cluster from the reduced input matrix
    cluster_q = cluster[:,numbers]
    print("cluster_q dimensions: {}".format(cluster_q.shape))

    # Scores matrix from the full input matrix
    Z_scal = cluster @ Aq 
    print("Zscal dimensions: {}".format(Z_scal.shape))

    # Regress the scores from the reduced input matrix
    Bi[ii],resid,rank,s = np.linalg.lstsq(cluster_q,Z_scal)
    print("Bi[ii] dimensions: {}".format(Bi[ii].shape))

    # Reconstruct ALL the variables in the given cluster by means of the local B_{i} matrix
    rec_cluster = cluster_q @ Bi[ii] @ Aq.T
    print("rec_cluster dimensions: {}".format(rec_cluster.shape))
    
    # Put back the rows in their original place
    positions = np.where(idx_clustering == ii)
    X_recovered[positions] = rec_cluster

    np.savetxt("Aq_" + str(ii) + ".txt", Aq)
    np.savetxt("B_" + str(ii) + ".txt", Bi[ii])


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