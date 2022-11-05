from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm import topi

# Global declarations of environment.

# llvm
tgt_host = "llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt = tvm.target.Target("llvm", host=tgt_host)


def make_elemwise_add(shape, tgt, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, func_name,
                               dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, func_name,
                               dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, name=func_name)
    return f


def make_relu(shape, tgt, func_name, dtype="float32"):
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.compute(
        A.shape, lambda *i: tvm.te.max(A(*i), tvm.tir.const(0, dtype)))

    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, name=func_name)
    return f


def make_relu_gradient(shape, tgt, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(
        A.shape, lambda *i: tvm.te.if_then_else(A(*i) > 0, B(*i), 0.0))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, name=func_name)
    return f

# 优化前：
# logreg: Validation set accuracy = 0.922300 Average Time per Training Epoch = 1.153565 s
#    MLP: Validation set accuracy = 0.971700 Average Time per Training Epoch = 42.437669 s


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt,
                    func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    lhs_shape = shapeA if not transposeA else shapeA[::-1]
    rhs_shape = shapeB if not transposeB else shapeB[::-1]
    assert len(lhs_shape) == len(rhs_shape)
    assert lhs_shape[1] == rhs_shape[0]
    out_shape = [lhs_shape[0], rhs_shape[1]]

    A = tvm.te.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.te.placeholder(shapeB, dtype=dtype, name="B")
    k = tvm.te.reduce_axis((0, lhs_shape[1]), name="k")
    C = None
    if (transposeA is False) and (transposeB is False):
        C = tvm.te.compute(out_shape, lambda i,
                           j: tvm.te.sum(A[i, k] * B[k, j], axis=k))
    elif (transposeA is True) and (transposeB is True):
        C = tvm.te.compute(out_shape, lambda i,
                           j: tvm.te.sum(A[k, i] * B[j, k], axis=k))
    elif (transposeA is True) and (transposeB is False):
        C = tvm.te.compute(out_shape, lambda i,
                           j: tvm.te.sum(A[k, i] * B[k, j], axis=k))
    else:
        C = tvm.te.compute(out_shape, lambda i,
                           j: tvm.te.sum(A[i, k] * B[j, k], axis=k))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, func_name, dtype="float32"):
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    assert (shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    out_shape = (N, M, H - R + 1, W - S + 1)
    di = tvm.te.reduce_axis((0, R), name="di")
    dj = tvm.te.reduce_axis((0, S), name="dj")
    dc = tvm.te.reduce_axis((0, C), name="dc")
    X = tvm.te.placeholder(shapeX, dtype=dtype, name="X")
    F = tvm.te.placeholder(shapeF, dtype=dtype, name="F")
    Y = tvm.te.compute(out_shape, lambda n, m, i, j: tvm.te.sum(
        X[n][dc][i + di][j + dj] * F[m][dc][di][dj], axis=[dc, di, dj]))
    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s, [X, F, Y], tgt, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    assert len(shape) == 2
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = topi.exp(A - topi.max(A, axis=1, keepdims=True) - 5)
    dj = tvm.te.reduce_axis((0, shape[1]), name="dj")
    C = tvm.te.compute(shape, lambda i, j: tvm.te.sum(B[i][dj], axis=dj))
    D = topi.divide(B, C)
    s = tvm.te.create_schedule(D.op)
    f = tvm.build(s, [A, D], tgt, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, func_name,
                                      dtype="float32"):
    """Hint: output shape should be (1,)"""
    assert len(shape) == 2
    Y = tvm.te.placeholder(shape, dtype=dtype, name="Y")
    Y_ = tvm.te.placeholder(shape, dtype=dtype, name="Y_")

    # D = topi.log(topi.nn.softmax(Y))
    B = topi.exp(Y - topi.max(Y, axis=1, keepdims=True) - 5)
    dj = tvm.te.reduce_axis((0, shape[1]), name="dj")
    C = tvm.te.compute(shape, lambda i,
                       j: tvm.te.sum(B[i][dj], axis=dj))
    D = topi.log(topi.divide(B, C))

    dp = tvm.te.reduce_axis((0, shape[0]))
    dq = tvm.te.reduce_axis((0, shape[1]))
    E = tvm.te.compute([1], lambda i: tvm.te.sum(-Y_[dp][dq]
                       * D[dp][dq], axis=[dp, dq]))
    F = topi.divide(E, shape[0])

    s = tvm.te.create_schedule(F.op)
    f = tvm.build(s, [Y, Y_, F], tgt, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, func_name,
                      dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, func_name,
                    dtype="float32"):
    X = tvm.te.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.te.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, name=func_name)
    return f
