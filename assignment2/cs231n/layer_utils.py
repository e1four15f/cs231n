pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine + batchnorm transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter
    - beta: Shift paremeter)
    - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features
          
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    fc, fc_cache = affine_forward(x, w, b)
    batchnorm, batchnorm_cache = batchnorm_forward(fc, gamma, beta, bn_param)
    out, relu_cache = relu_forward(batchnorm)
    cache = (fc_cache, batchnorm_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, batchnorm_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward_alt(da, batchnorm_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_layernorm_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    Convenience layer that perorms an affine + layernorm transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter
    - beta: Shift paremeter)
    - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features
          
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    fc, fc_cache = affine_forward(x, w, b)
    layernorm, layernorm_cache = layernorm_forward(fc, gamma, beta, ln_param)
    out, relu_cache = relu_forward(layernorm)
    cache = (fc_cache, layernorm_cache, relu_cache)
    return out, cache


def affine_layernorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-layernorm-relu convenience layer
    """
    fc_cache, layernorm_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dln, dgamma, dbeta = layernorm_backward(da, layernorm_cache)
    dx, dw, db = affine_backward(dln, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_dropout_relu_forward(x, w, b, dropout_param):
    """
    Convenience layer that perorms an affine + dropout transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter
    - beta: Shift paremeter)
    - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We keep each neuron output with probability p.
          - mode: 'test' or 'train'. If the mode is train, then perform dropout;
            if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking but not
            in real networks.
        
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    fc, fc_cache = affine_forward(x, w, b)
    dropout, dropout_cache = dropout_forward(fc, dropout_param)
    out, relu_cache = relu_forward(dropout)
    cache = (fc_cache, dropout_cache, relu_cache)
    return out, cache


def affine_dropout_relu_backward(dout, cache):
    """
    Backward pass for the affine-dropout-relu convenience layer
    """
    fc_cache, dropout_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    ddo = dropout_backward(da, dropout_cache)
    dx, dw, db = affine_backward(ddo, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
