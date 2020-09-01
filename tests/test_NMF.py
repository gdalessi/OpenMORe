'''
MODULE: test_NMF.py

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


class testNMF(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(30,5)
        self.dimensions = 3

    def tearDown(self):
        pass

    def test_ALS_std(self):
        model = model_order_reduction.NMF(self.X)
        model.algorithm = 'als'
        model.method = 'standard'

        try:
            model.fit()
            passed = True
        except:
            passed = False 

        self.assertEqual(passed, True)

    def test_ALS_sparse(self):
        model = model_order_reduction.NMF(self.X)
        model.algorithm = 'als'
        model.method = 'sparse'

        try:
            model.fit()
            passed = True
        except:
            passed = False 

        self.assertEqual(passed, True)

    def test_MUR_frobenius(self):
        model = model_order_reduction.NMF(self.X)
        model.algorithm = 'mur'
        model.metric = 'frobenius'

        try:
            model.fit()
            passed = True
        except:
            passed = False 

        self.assertEqual(passed, True)

    def test_MUR_kld(self):
        model = model_order_reduction.NMF(self.X)
        model.algorithm = 'mur'
        model.metric = 'kld'

        try:
            model.fit()
            passed = True
        except:
            passed = False 

        self.assertEqual(passed, True)
