import cupy as cp

def ReLu(X):
    return cp.maximum(X, 0)

def ReLu_derive(X):
    return X > 0

def softmax(X):
    return cp.exp(X) / sum(cp.exp(X))

def sigmoid(X):
    return 1 / (1 + cp.exp(-X))

def one_hot_encode(Y, max_val=9):
    one_hot_encode_Y = cp.zeros((Y.size, max_val + 1))
    one_hot_encode_Y[cp.arange(Y.size), Y] = 1
    one_hot_encode_Y = one_hot_encode_Y.T
    return one_hot_encode_Y

