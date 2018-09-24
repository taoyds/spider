from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val))


def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)


def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](0.), *dims)


def tensor_right_shift(tensor):
    temp = T.zeros_like(tensor)
    temp = T.set_subtensor(temp[:, 1:, :], tensor[:, :-1, :])

    return temp


def ndim_tensor(ndim, name=None):
    if ndim == 1:
        return T.vector()
    elif ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix(name=name)


# get int32 tensor
def ndim_itensor(ndim, name=None):
    if ndim == 2:
        return T.imatrix(name)
    elif ndim == 3:
        return T.itensor3(name)
    elif ndim == 4:
        return T.itensor4(name)
    return T.imatrix(name=name)


# get int8 tensor
def ndim_btensor(ndim, name=None):
    if ndim == 2:
        return T.bmatrix(name)
    elif ndim == 3:
        return T.btensor3(name)
    elif ndim == 4:
        return T.btensor4(name)
    return T.imatrix(name)