import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from preprocessing import *
from initialization import *

import clustering


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/python_course/LPCA/",
    "file_name"                 : "f10A25.csv",
}


settings = {
    "centering_method"          : "MEAN",
    "scaling_method"            : "AUTO",
    "initialization_method"     : "KMEANS",
    "number_of_clusters"        : 8,
    "number_of_eigenvectors"    : 15
}


try:
    print("Reading the training matrix..")
    X = pd.read_csv(file_options["path_to_file"] + file_options["file_name"], sep = ',', header = None) 
    print("The training matrix has been read successfully!")
except OSError:
    print("Could not open/read the selected file: " + file_options["file_name"])
    exit()



X_tilde = center_scale(X, center(X, method=settings["centering_method"]), scale(X, method=settings["scaling_method"]))

model = clustering.lpca(X_tilde, settings["number_of_clusters"], settings["number_of_eigenvectors"], settings["initialization_method"])
index = model.fit()

np.savetxt("idx_new_version_python.txt", index)



