import sys
sys.path.insert(1, '../src')

import pyMORe.model_order_reduction as model_order_reduction
from pyMORe.utilities import *


file_options = {
    "path_to_file"              : "../data",
    "input_file_name"           : "flameD.csv",
}

settings ={
    #centering and scaling options
    "centering"                 : "mean",
    "scaling"                   : "auto",

    #set the number of PCs: it can be done automatically, or it can be
    #decided by the user.
    "number_of_PCs"             : "auto",
}


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])


model = model_order_reduction.PCA(X)


if settings["number_of_PCs"] is "auto":

    #if the "auto" setting is chosen, the number of PCs
    #is set on the basis of the explained variance (95% min)
    #or on the basis of the NRMSE with the reconstructed
    #matrix (<10% error with the original matrix, on average)
    model.set_PCs_method = 'var'
    model.set_PCs()
else:
    #otherwise the user input is used
    model.eigens = settings["number_of_PCs"]


#perform the dimensionality reduction via Principal Component Analysis,
#and return the eigenvectors of the reduced manifold
PCs = model.fit()


#compute the projection of the original points on the reduced
#PCA manifold, obtaining the scores matrix Z
Z = model.get_scores()


#assess the percentage of explained variance if the number of PCs has not
#been set automatically, and plot the result
model.get_explained()


#reconstruct the matrix from the reduced PCA manifold
X_recovered = model.recover()


#compare the reconstructed chosen variable "set_num_to_plot" with the
#original one, by means of a parity plot
model.set_num_to_plot = 0
model.plot_parity()
