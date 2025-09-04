import cupy as cp
from .Layer import Layer
from .Randomization import *
from .activation import *
import pickle

def testing3():
    print('test')

def conv2d(X:cp.ndarray, weight:cp.ndarray, bias, stride=4, padding = 0):
    N, N_C, H, W = X.shape
    F_N, F_C, F_H, F_W = weight.shape

    out_h = (H-F_H + 2 * padding) // stride + 1   
    out_w = (W-F_W + 2 * padding) // stride + 1   

    output = cp.zeros((N, F_N, out_h, out_w))
    X_padded = cp.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

    for n in range(N):
        for f in range(F_N):
            for i in range(out_h):
                for j in range(out_w):
                    region = X_padded[n,:, i *stride: i*stride+F_H, j*stride: j*stride+F_W]
                    output[n, f,i,j] = cp.sum(region * weight[f]) + bias[f]
    return output

def maxPool(X, size = 3, stride=2):
    N, C, H, W = X.shape
    out_h = (H-size) // stride + 1
    out_W = (W-size) // stride + 1
    output = cp.zeros((N, C, out_h, out_W))

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_W):
                    region = X[n, c, i * stride: i*stride +size,j*stride:j*stride+size]
                    output[n, c,i,j] = cp.max(region)
    return output

def alex_first_pass(X, filters, bias):
    A = conv2d(X, filters, bias)
    B = ReLu(A)
    C = maxPool(B)
    return A, B, C