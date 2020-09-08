import OpenMORe.clustering as clustering
from OpenMORe.utilities import *

import matplotlib 
import matplotlib.pyplot as plt
import os



file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/dummy_data/")),
    "input_file_name"           : "moons.csv",
}


settings = {
    #centering and scaling options
    "center"                    : False,
    "centering_method"          : "mean",
    "scale"                     : False,
    "scaling_method"            : "auto",

    #clustering options: choose the number of clusters
    "number_of_clusters"        : 2,
    "sigma"                     : 0.2,

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

matplotlib.rcParams.update({'font.size' : 12, 'text.usetex' : True})

fig = plt.figure()
axes = fig.add_axes([0.2,0.15,0.7,0.7], frameon=True)

cmap = matplotlib.colors.ListedColormap(['darkred', 'midnightblue'])
sc = axes.scatter(X[:,0], X[:,1], c=idx,alpha=0.9, cmap=cmap, edgecolor ='none')
bounds = [0, 1]
axes.set_title("Spectral clustering solution")
axes.set_xlabel('X [-]')
axes.set_ylabel('Y [-]')
#plt.colorbar(sc, extendfrac='auto',spacing='uniform')
cb = plt.colorbar(sc, spacing='uniform', ticks=bounds)
cb.set_ticks(ticks=range(2))
plt.show()