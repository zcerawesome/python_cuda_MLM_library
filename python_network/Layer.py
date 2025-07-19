import cupy as cp

class Layer:
    def __init__(self, weight_dimension, bias_dimension, activation_function, activation_derive, output):
        self.weight = cp.zeros((weight_dimension[0], weight_dimension[1]))
        self.bias = cp.zeros((bias_dimension[0], bias_dimension[1]))
        self.activation = activation_function
        self.activation_derive = activation_derive
        self.output = output