import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pyMORe.model_order_reduction as model_order_reduction
from pyMORe.utilities import *

file_options = {
    "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitLab/pyMORe/data",
    "input_file_name"           : "flameD.csv",

    "mesh_file_name"            : "mesh.csv",
}


num_of_features = 5

X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
mesh = np.genfromtxt(file_options["path_to_file"] + "/" + file_options["mesh_file_name"], delimiter= ',')

model = model_order_reduction.NMF(X)
model.encoding = num_of_features


W,H = model.fit()
idx = model.cluster()

for ii in range(0, num_of_features):
    fig = plt.figure()
    axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
    axes.scatter(mesh[:,0], mesh[:,1], c=W.T[:,ii],alpha=0.5, cmap='gnuplot')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    plt.show()


fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)
axes.scatter(mesh[:,0], mesh[:,1], c=idx,alpha=0.5, cmap='gnuplot')
axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.show()

PHC_coeff, PHC_deviations = evaluate_clustering_PHC(X, idx, method='PHC_standard')
print("PHC is equal to: {}".format(PHC_coeff))



