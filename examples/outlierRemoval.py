import sys
sys.path.insert(1, '../src')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    "number_of_PCs"             : 15,

    #set the method for the outlier removal procedure
    "method"                    : "leverage",
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = model_order_reduction.PCA(X)
model.eigens = settings["number_of_PCs"]

if settings["method"].lower() == "leverage":
    X_cleaned, ___, ____ = model.outlier_removal_leverage()
elif settings["method"].lower() == "orthogonal":
    X_cleaned = model.outlier_removal_orthogonal()
elif settings["method"].lower() == "multistep":
    X_cleaned = model.outlier_removal_multistep()
else:
    raise Exception("Outlier removal method not available. Available methods: leverage, orthogonal, multistep. Exiting with error..")
    exit()

print("The matrix dimensions with outliers are: {}x{}".format(X.shape[0], X.shape[1]))
print("The matrix dimensions after the outlier removal procedure are: {}x{}".format(X_cleaned.shape[0], X_cleaned.shape[1]))
