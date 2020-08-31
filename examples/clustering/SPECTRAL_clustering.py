import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

import PyTROModelling.clustering as clustering
from PyTROModelling.utilities import *


from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs

file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitLab/PyTROModelling/data/dummy_data",
    "input_file_name"           : "moons.csv",
}


settings = {
    #centering and scaling options
    "center"                    : False,
    "centering_method"          : "mean",
    "scale"                     : False,
    "scaling_method"            : "auto",

    #clustering options: choose the number of clusters
    "number_of_clusters"        : 4,
    "sigma"                     : 1.0,

    #write clustering solution on txt
    "write_on_txt"              : False,
    "evaluate_clustering"       : False,
}

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])

model = clustering.spectralClustering(X)
model.to_center = False
model.to_scale = False
model.clusters = settings["number_of_clusters"]
model.sigma = settings["sigma"]
idx = model.fit()

plt.scatter(X[:, 0], X[:, 1], marker='o', c=idx)
plt.title("spectral clustering solution on dummy")
plt.show()