from .Network import *
from .Alex import *
import pandas as pd
import cupy as cp
import time

def get_accuracy(Y_hat, Y):
    return cp.sum(Y_hat == Y) / Y.size

def test():
    temp = Network()
    temp.add_layer(784, 0, 0)
    temp.add_layer(10, ReLu, ReLu_derive)
    temp.add_layer(10, softmax, 0)
    temp.apply_randomization([1,2], uniform_rand)

    data = pd.read_csv('train.csv')

    data = cp.array(data)
    m, n = data.shape

    Train = data.T
    X_train = Train[1:n]
    X_train = X_train / 255.
    Y_train = Train[0]
    start = time.time()
    Y_encode = one_hot_encode(Y_train)
    print(X_train[: 0, None].shape)
    for i in range(500):
        Z1, A1, Z2, A2 = temp.forward(X_train)
        New_data = temp.forward(X_train)
        for dat in New_data:
            print(dat.shape)
        dW1, db1, dW2, db2 = temp.backward_prop([Z1, A1, Z2, A2], X_train, Y_encode)
        temp.update_params([dW1, db1, dW2, db2], 0.1)
        if i % 10 == 0:
            print('Iteration ', i)
            print('Accuracy: ', get_accuracy(cp.argmax(A2, 0), Y_train))
    end = time.time()
    print(end - start)