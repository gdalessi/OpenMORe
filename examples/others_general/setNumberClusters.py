import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.clustering as clustering
from OpenMORe.utilities import *


#######################################################################################
# In this example, it's shown how to set the optimal number of clusters (a posteriori) 
# via Davies Bouldin index. The LPCA clustering algorithm is used in this example.
#######################################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "turbo2D.csv",
}

# Dictionary with the instructions for the outlier removal class:
settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the initialization method (random, observations, kmeans, pkcia, uniform)
    "initialization_method"     : "uniform",

    #set the number of PCs in each cluster
    "number_of_eigenvectors"    : 12,

    #enable additional options:
    "initial_k"                 : 2,
    "final_k"                   : 10,
    "write_stats"               : False,
}

# Load the input matrix:
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
# Preprocess the input matrix: center and scale:
X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))

# Initialize the clusters selection procedure
num_of_k = np.linspace(settings["initial_k"], settings["final_k"], settings["final_k"]-settings["initial_k"]+1)
DB_scores = [None]*len(num_of_k)
idxs = [None]*len(num_of_k)

# Perform the clustering algorithm k times to see what's the best clustering solution
for ii in range(0,len(num_of_k)):
    model = clustering.lpca(X, settings)
    model.clusters = int(num_of_k[ii])
    index = model.fit() 
    
    idxs[ii] = index
    DB_scores[ii] = evaluate_clustering_DB(X_tilde, index)

print(DB_scores)
print("According to the Davies-Bouldin index, the best number of clusters to use for the given dataset is: {}".format(num_of_k[np.argmin(DB_scores)]))
print("Corresponding idx: {}".format(idxs[np.argmin(DB_scores)]))

# Save the best clustering solution in a txt
best_solution_idx = idxs[np.argmin(DB_scores)]
np.savetxt("best_idx.txt", best_solution_idx)

plt.rcParams.update({'font.size' : 12, 'text.usetex' : True})
fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.plot(num_of_k, DB_scores, c='darkblue',alpha=0.9, marker='o', linestyle='dashed', linewidth=2, markersize=12)
axes.set_xlabel('$Number\ of\ clusters\ [-]$')
axes.set_ylabel('$Davies-Bouldin\ index\ [-]$')
plt.show()