# General
OpenMORe is a collection of Python modules for Model-Order-Reduction, clustering and classification. 

# How to cite
If you use OpenMORe for your publications, I kindly ask you to cite the following papers:
If you use clustering.py and/or classification.py:

* _D’Alessio et al., "Adaptive chemistry via pre-partitioning of composition space and mechanism reduction." Combustion and Flame 211 (2020): 68-82._
* _D'Alessio et al., "Feature extraction and artificial neural networks for the on-the-fly classification of high-dimensional thermochemical spaces in adaptive-chemistry simulations." Data-Centric Engineering 2 (2021)._

If you use model_order_reduction.py:

* *D'Alessio et al., "Analysis of turbulent reacting jets via principal component analysis." Data Analysis for Direct Numerical Simulations of Turbulent Combustion. Springer, Cham, 2020. 233-251.*


# **Implemented techniques:**

_Model Order Reduction:_
- Principal Component Analysis (PCA)
- Local PCA (LPCA)
- Kernel PCA (KPCA)
- Feature selection via PCA 
- Outlier removal via PCA
- Data sampling 

_Clustering:_
- Local PCA (via Vector Quantization, _unsupervised_)
- FPCA (via conditioning vector, _supervised_)
- Spectral Clustering (_unsupervised_)

_Utilities:_
- Multivariate data preprocessing 
- Varimax Rotation 
- Clustering solution evaluation 
- Fast algorithm for SVD 


# **Requirements**: 
In order to use OpenMORe on your devices, the following requirements must be satisfied:

- Python version >= 3.6 
- Numpy must be installed 
- Scipy must be installed 
- Matplotlib must be installed
- Pandas must be installed
- Latex must be installed (for the plots' labels)


# **Installation**: 
If the libraries requirements are satisfied, clone or download the repo. After that, go to the OpenMORe folder from your terminal (where the file *setup.py* is located) and type: `python setup.py install`. 

# **Test**: 
It is possible to check if the installation process was successful running the tests. To do that, just type:
- `python -m unittest tests/test_PCA.py`  
- `python -m unittest tests/test_sampling.py `
- `python -m unittest tests/test_clustering.py `

If the tests have positive response, you should get a message like: 

```
___________________________
Ran 4 tests in 0.113s

OK
___________________________
```

# **Usage**: 
If the tests are successful, you can now use OpenMORe. In the "examples" folder there are some pre-set cases, organized according to the final purpose (e.g., clustering, dimensionality-reduction, variables-selection, others). They are also fully commented to describe the required dictionary inputs.
In the “data/reactive_flow” folder, there is a collection of data (from a CFD simulation of a turbulent reacting jet) to run the examples and test the code, while in "data/dummy_data" you can find relatively simple data sets to test the scripts' functionality. 
A detailed description of all the classes and functions is available in the detailed documentation and in the source code. 

# **Documentation**: 
The official documentation is available in /OpenMORe/Documentation. It is strongly suggested to read it before using the software.

For any question or problem regarding the code you can write to me at the following address: giuseppe.dalessio@ulb.ac.be 
