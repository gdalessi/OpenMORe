import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pyMORe.clustering as clustering
from pyMORe.utilities import *
import pyMORe.classification as classification


file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "input_file_name"           : "cfdf.csv",

    "idx_name"                  : "idx.txt",
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
idx = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["idx_name"], delimiter= '\n')


file_options_classifier = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "test_file_name"            : "laminar2D.csv",
}

mesh_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
    "mesh_file_name"            : "mesh.csv",
}

try:
    print("Reading test matrix..")
    Y = np.genfromtxt(file_options_classifier["path_to_file"] + "/" + file_options_classifier["test_file_name"], delimiter= ',')
except OSError:
    print("Could not open/read the selected file: " + "/" + file_options_classifier["test_file_name"])
    exit()


classifier = classification.VQPCA(X, idx, Y)
classification_vector = classifier.fit()

matplotlib.rcParams.update({'font.size' : 6, 'text.usetex' : True})
mesh = np.genfromtxt(mesh_options["path_to_file"] + "/" + mesh_options["mesh_file_name"], delimiter= ',')

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=classification_vector, alpha=0.5)
axes.set_xlabel('X [m]')
axes.set_ylabel('Y [m]')
plt.show()
