OpenMORe is a collection of Python modules for Model-Order-Reduction, clustering and classification. 

**Implemented techniques:**

_Model Order Reduction:_
- Principal Component Analysis (PCA)
- Local PCA (LPCA)
- Kernel PCA (KPCA)
- Non-negative Matrix Factorization (NMF) 
- Feature selection via PCA 
- Outlier removal via PCA
- Data sampling 

_Clustering:_
- Local PCA (via Vector Quantization, _unsupervised_)
- FPCA (via conditioning vector, _supervised_)
- Kmeans (_unsupervised_)
- Spectral Clustering (_unsupervised_)

_Utilities:_
- Multivariate data preprocessing 
- Varimax Rotation 
- Clustering solution evaluation 
- Fast algorithm for SVD of massive data (approx)


**Requirements**: in order to use OpenMORe on your devices, the following requirements must be satisfied:

- Python version >= 3.6 
- Numpy must be installed 
- Scipy must be installed 
- Matplotlib must be installed
- Scikit-learn is optional (only needed from the KPCA class)


**Installation**: if the libraries requirements are satisfied, clone/download the repo. After that, go to the OpenMORe folder from your terminal (where the file *setup.py* is located) and type: `python setup.py install`. 

**Test**: it is possible to check if the installation process was successful running the tests. To do that, just type:
- `python -m unittest tests/test_PCA.py`  
- `python -m unittest tests/test_sampling.py `
- `python -m unittest tests/test_NMF.py` 
- `python -m unittest tests/test_clustering.py `

If the tests have positive response, you should get a message like: 

```
___________________________
Ran 4 tests in 0.113s

OK
___________________________
```

**Usage**: if the tests are successful, you can now use OpenMORe. In the "examples" folder there are some pre-set cases, organized according to the final aim (i.e., clustering/dimensionality-reduction/others_general/variables-selection). They are also fully commented to describe the required dictionary inputs.
In the “data/reactive_flow” folder, there is a collection of data (from a CFD simulation of a turbulent reacting jet) to run the examples and test the code, while in "data/dummy_data" you can find relatively simple data sets to test the scripts' functionality. 
A detailed description of all the classes and functions is available in the detailed documentation and in the source code. 

**Documentation**: the official documentation is available in /OpenMORe/Documentation. It is strongly suggested to read it before using the software.

For any question or problem regarding the code you can write to me at the following address: giuseppe.dalessio@ulb.ac.be 