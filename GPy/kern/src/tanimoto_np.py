# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .stationary import Stationary
from ...core import Param
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from .kern import Kern

class Tanimoto_NP(Kern):
    
    def __init__(self, input_dim, variance=1., active_dims=None, name="tanimoto_NP"):
        super(Tanimoto_NP, self).__init__(input_dim, active_dims, name)
        self.variance = Param("variance", variance)
        self.link_parameter(self.variance)
    
    def addition_elem_wise(self, a, b):
        add = a.reshape(-1, 1) + b.reshape(1, -1)
        return np.reshape(add, (len(a), len(b)))
    
    def parameters_changed(self):
        pass

    def K(self, X, X2):

        if X2 is None:
            X2 = X

        Xs = np.square(np.linalg.norm(X, axis=1))
        X2s = np.square(np.linalg.norm(X2, axis=1))
        outer_product = X @ X2.T
        
        denominator = -outer_product + self.addition_elem_wise(Xs, X2s)
        
        return self.variance * outer_product / denominator
    
    def Kdiag(self, X):
        return self.variance*np.ones(X.shape[0])
    
    def parameters_changed(self):
        pass

    def update_gradients_diag(self, a, b):
        pass

    def update_gradients_full(self, a, b, c):
        pass