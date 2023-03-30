# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .stationary import Stationary
from ...core import Param
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from .kern import Kern
import tensorflow as tf

class Tanimoto(Kern):
    
    def __init__(self, input_dim, variance=1., active_dims=None, name="tanimoto"):
        super(Tanimoto, self).__init__(input_dim, active_dims, name)
        self.variance = Param("variance", variance)
        self.link_parameter(self.variance)
    
    def broadcasting_elementwise(self,
        op, a, b
        ) -> tf.Tensor:
        """
        Apply binary operation `op` to every pair in tensors `a` and `b`.
        :param op: binary operator on tensors, e.g. tf.add, tf.substract
        """
        flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
        return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + self.broadcasting_elementwise(tf.add, Xs, X2s)

        return (self.variance * outer_product/denominator).numpy()

    def Kdiag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return (tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))).numpy()
    
    def parameters_changed(self):
        pass

    def update_gradients_diag(self, a, b):
        pass

    def update_gradients_full(self, a, b, c):
        pass