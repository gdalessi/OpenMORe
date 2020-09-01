'''
MODULE: test_PCA.py

@Authors:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be


@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''

import unittest

import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *


class testPCA(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(30,5)
        self.nPCtest = 3
        self.kernelType = 'rbf'
        self.nVarTest = 4
        self.selMethod1 = 'procustes'
        self.selMethod2 = 'b2'
        self.selMethod3 = 'procustes_rotation'


    def tearDown(self):
        pass
    
    def test_pca(self):
        globalPCA = model_order_reduction.PCA(self.X)
        globalPCA.eigens = self.nPCtest
        globalPCA._plot_explained_variance = False
        
        PCs, eigenvalues = globalPCA.fit()
        explained = globalPCA.get_explained()


        self.assertEqual(PCs.shape[1],self.nPCtest)
        self.assertEqual(len(eigenvalues),self.nPCtest)
        self.assertIsInstance(explained, float)

    def test_kpca(self):
        kernelPCA =  model_order_reduction.KPCA(self.X)
        kernelPCA.eigens = self.nPCtest
        kernelPCA.kernel_type = self.kernelType
        kernelPCA.retained = self.nVarTest

        Z, A, S = kernelPCA.fit()
        labels = kernelPCA.select_variables()[0]


        self.assertEqual(Z.shape[1], self.nPCtest)
        self.assertEqual(A.shape[1], self.nPCtest)
        self.assertEqual(len(S), self.nPCtest)
        self.assertEqual(len(labels), self.nVarTest)

    def test_varSelection(self):
        linearSelection = model_order_reduction.variables_selection(self.X)
        linearSelection.eigens = self.nPCtest
        linearSelection.retained = self.nVarTest

        linearSelection.method = self.selMethod1
        labels1, ____ = linearSelection.fit()

        linearSelection.method = self.selMethod2
        labels2 = linearSelection.fit()

        linearSelection.method = self.selMethod3
        labels3 = linearSelection.fit()

        self.assertEqual(len(labels1), self.nVarTest)
        self.assertEqual(len(labels2), self.nVarTest)
        self.assertEqual(len(labels3), self.nVarTest)
