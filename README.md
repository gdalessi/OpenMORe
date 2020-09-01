OpenMORe is a collection of Python modules for Model-Order-Reduction, clustering and classification. Several techniques are implemented to accomplish the aforementioned purposes, i.e.: Principal Component Analysis (PCA), Local Principal Component Analysis (LPCA), Kernel Principal Component Analysis (KPCA), K-means, Non-negative Matrix Factorization (NMF). 
Moreover, a wide range of useful functions for general machine-learning tasks are implemented such as data scaling, matrix sampling, clusters evaluation, multivariate outlier identification and removal, linear and non-linear feature selection. 

**Requirements**: in order to use OpenMORe on your devices, the following requirements must be satisfied:

- *Python version >= 3.6 
- Numpy must be installed 
- Scipy must be installed 
- Matplotlib must be installed*



**Installation**: if the libraries requirements are satisfied, clone/download the repo. After that, go to the OpenMORe folder from your terminal (where the file *setup.py* is located) and type: `python setup.py install`. 

**Test**: it is possible to check if the installation process was successful running the tests. To do that, go to the OpenMORe folder from your terminal and type:
- `python -m unittest tests/test_PCA.py`  
- `python -m unittest tests/test_sampling.py `
- `python -m unittest tests/test_NMF.py` 
- `python -m unittest tests/test_clustering.py `


**Usage**: if the tests were successful, you can now use OpenMORe. In the "examples" folder there are some pre-set cases, organized on the basis of the functionality (i.e., clustering/dimensionality-reduction/preprocessing&sampling/variables-selection). They are also fully commented to describe the required inputs and setters
In the “data/reactive_flow” folder, a collection of data (from a CFD simulation of a turbulent reacting jet) is also contained to run the examples and test the code, while in "data/dummy_data" you can find relatively simple data sets to test the scripts' functionality. 
A detailed description of all the classes and functions is available in the source code. 

**Warning:** I wrote almost all the classes and functions from scratch, and I did not have the time to test everything accurately. Given that, there may be bugs in some functions or classes. In case you spot one, please write to me so I can clean up the code.