import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *

import matplotlib.pyplot as plt
import numpy as np
import os

file_options = {
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "input_file_name"           : "flameD.csv",
}

mesh_options = {
    #set the mesh file options (the path goes up twice - it's ok)
    "path_to_file"              : os.path.abspath(os.path.join(__file__ ,"../../../data/reactive_flow/")),
    "mesh_file_name"            : "mesh.csv",

    #eventually enable the clustering solution plot on the mesh
    "plot_on_mesh"              : True,
}

settings ={
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_eigenvectors"    : 7,
    
    #enable to plot the cumulative explained variance
    "enable_plot_variance"      : True,
    
    #set the number of the variable whose reconstruction must be plotted
    "variable_to_plot"          : 0,

}


X = readCSV(file_options["path_to_file"], file_options["input_file_name"])



model = model_order_reduction.PCA(X, settings)
#perform the dimensionality reduction via Principal Component Analysis,
#and return the eigenvectors of the reduced manifold
PCs, ____ = model.fit()

#plot the original PC
model.plot_PCs()
#apply the varimax rotation algorithm from the utilities module
rotated = varimax_rotation(X, PCs, normalize=False)

#plot the rotated PC
fig = plt.figure()
axes = fig.add_axes([0.15,0.15,0.7,0.7], frameon=True)

x = np.linspace(1, X.shape[1], X.shape[1])
axes.bar(x, rotated[:,0])
axes.set_xlabel('Variables [-]')
axes.set_ylabel('Weights on the rotated PC number: {} [-]'.format(0))
plt.show()