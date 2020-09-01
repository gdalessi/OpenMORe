<<<<<<< HEAD
PyTROMode (Python Tools for Reduced Order Modelling) is a collection of Python 
modules for Model-Order-Reduction, clustering and classification. 
=======
PyTROMode (Python Tools for Reduced Order Modelling) is a collection of Python modules for Model-Order-Reduction, clustering 
and classification. 
>>>>>>> master
Several techniques are implemented to accomplish the aforementioned purposes, 
i.e.: Principal Component Analysis (PCA), Local Principal Component Analysis 
(LPCA), K-means, Non-negative Matrix Factorization (NMF). Moreover, a wide range 
of useful functions for general machine-learning tasks are implemented as data 
scaling, matrix sampling, clusters evaluation, (multivariate) outlier 
identification and removal.

**Requirements:**
in order to use pyMORe on your devices, the following requirements must be 
satisfied:
-	Python version >= 3.6
-	Numpy must be installed
-	Scipy must be installed
-	Matplotlib must be installed

**Installation:**
if the libraries requirements are satisfied, go to the pyMORe folder from your 
terminal, and type: `python setup.py install`.

**Usage:**
you can run a quick test running one of the examples in the “examples” folder, 
or following one of the jupyter tutorials in the “tutorials” folder. In the 
“data” folder, a collection of data (from a CFD simulation of a turbulent 
reacting jet) is also contained to run the examples and test the code.

Example: `python examples/LPCA_clustering.py`
	
A detailed description of all the functions is available in the source code. 
Several examples for different tasks (clustering, model order reduction, 
dimensionality reduction, outlier removal, etc.) are available in the examples 
folder, and they are also commented to describe the required inputs and setters.

**Warning:**
I wrote almost all the classes and functions from scratch, and I did not have 
the time to test everything accurately. Given that, there may be bugs in some 
functions. In case you spot one, please write to me so I can clean up the code.