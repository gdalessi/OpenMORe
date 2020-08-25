import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import PyTROModelling.model_order_reduction as model_order_reduction
from PyTROModelling.utilities import *

file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitLab/PyTROModelling/data",
    "input_file_name"           : "flameD.csv",
}

settings = {
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #variables selection options
    "method"                    : "procustes",
    "number_of_PCs"             : 8,
    "number_of_variables"       : 25,
    "path_to_labels"            : "/Users/giuseppedalessio/Dropbox/GitLab/PyTROModelling/data",
    "labels_name"               : "labels.csv",
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
X = X[:,1:]

PVs = model_order_reduction.variables_selection(X, settings)
labels = PVs.fit()

print(labels)
print(numbers)
