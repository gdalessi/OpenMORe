import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/dummy_data/")),
    "input_file_name"           : "outlier_data.csv",
}

settings ={
    #centering and scaling options
    "center"                    : True,
    "centering"                 : "mean",
    "scale"                     : True,
    "scaling"                   : "auto",

    #set the number of PCs: it can be done automatically, or it can be
    #decided by the user.
    "number_of_PCs"             : 1,

    #set the method for the outlier removal procedure
    "method"                    : "orthogonal",
}


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = model_order_reduction.PCA(X)
model.eigens = settings["number_of_PCs"]
model.to_center = settings["center"]
model.centering = settings["centering"]
model.to_scale = settings["scale"]
model.scaling = settings["scaling"]

if settings["method"].lower() == "leverage":
    X_cleaned, ___, id = model.outlier_removal_leverage()
elif settings["method"].lower() == "orthogonal":
    X_cleaned, ___, ____ = model.outlier_removal_orthogonal()
elif settings["method"].lower() == "multistep":
    X_cleaned = model.outlier_removal_multistep()
else:
    raise Exception("Outlier removal method not available. Available methods: leverage, orthogonal, multistep. Exiting with error..")
    exit()

print("The matrix dimensions with outliers are: {}x{}".format(X.shape[0], X.shape[1]))
print("The matrix dimensions after the outlier removal procedure are: {}x{}".format(X_cleaned.shape[0], X_cleaned.shape[1]))

matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(X[:,0], X[:,1], c='seagreen',alpha=0.9)
axes.set_title("Full data with outliers")
axes.set_xlabel('X [-]')
axes.set_ylabel('Y [-]')
plt.show()

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(X[:,0], X[:,1], c='red',alpha=0.9)
axes.scatter(X_cleaned[:,0], X_cleaned[:,1], c='darkblue',alpha=0.9)
axes.set_title("Outliers identification (in red)")
axes.set_xlabel('X [-]')
axes.set_ylabel('Y [-]')
plt.show()


fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)

axes.scatter(X_cleaned[:,0], X_cleaned[:,1], c='darkblue',alpha=0.9)
axes.set_title("Cleaned data without outliers")
axes.set_xlabel('X [-]')
axes.set_ylabel('Y [-]')
plt.show()