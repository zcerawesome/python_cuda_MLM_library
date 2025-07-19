import cupy as cp
from .Layer import Layer
from .Randomization import *
from .activation import *
import pickle

class Network:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, output, activation_function, activation_derive):
        if(len(self.layers) == 0):
            self.layers.append(Layer([0,0], [0,0], 0, 0, output))
        else:
            self.layers.append(Layer([output, self.layers[-1].output], [output, 1], activation_function, activation_derive, output))
        
    def apply_randomization(self, index, random, lower=-0.5, higher=0.5):
        if type(index) == int:
            self.layers[index].weight = random(lower, higher, self.layers[index].weight.shape)
        else:
            for indice in index:
                self.layers[indice].weight = random(lower, higher, self.layers[indice].weight.shape)

    def forward(self, X):
        results = [cp.array] * ((len(self.layers) - 1) * 2)
        for i in range(len(self.layers) - 1):
            layer = self.layers[i + 1]
            if i == 0:
                Z = layer.weight.dot(X) + layer.bias
            else:
                Z = layer.weight.dot(results[i* 2 - 1]) + layer.bias
            A = layer.activation(Z)
            results[i * 2] = Z
            results[i * 2 + 1] = A
        return results

    def backward_prop_policy(self, forward: list, X, Y, Reward):
        col = forward[-1].shape[1]
        results = [cp.array] * (len(self.layers) -1) * 2
        not_results = [cp.array] * (len(self.layers) - 1)
        i = len(self.layers) - 2
        while i >= 0:
            if i == len(self.layers) - 2:
                not_results[i] = (forward[-1] - Y) * Reward
            else:
                not_results[i] = self.layers[i + 2].weight.T.dot(not_results[i + 1]) * self.layers[i + 1].activation_derive(forward[i * 2])
            if i == 0:
                results[i * 2] = not_results[i].dot(X.T) / col
            else:
                results[i * 2] = not_results[i].dot(forward[i * 2 - 1].T) / col
            results[i * 2 + 1] = cp.array([cp.sum(not_results[i], axis=1)]).T / col
            
            i -=1

        return results

    def backward_prop(self, forward: list, X, Y):
        col = Y.shape[1]
        results = [cp.array] * (len(self.layers) -1) * 2
        not_results = [cp.array] * (len(self.layers) - 1)
        i = len(self.layers) - 2
        while i >= 0:
            if i == len(self.layers) - 2:
                not_results[i] = forward[-1] - Y
            else:
                not_results[i] = self.layers[i + 2].weight.T.dot(not_results[i + 1]) * self.layers[i + 1].activation_derive(forward[i * 2])
            if i == 0:
                results[i * 2] = not_results[i].dot(X.T) / col
            else:
                results[i * 2] = not_results[i].dot(forward[i * 2 - 1].T) / col
            results[i * 2 + 1] = cp.array([cp.sum(not_results[i], axis=1)]).T / col
            
            i -=1
        return results
    
    def update_params(self, back_prop: list, alpha):
        for i in range(int(len(back_prop) / 2)):
            index = i + 1
            self.layers[index].weight = self.layers[index].weight - back_prop[i * 2] * alpha
            self.layers[index].bias = self.layers[index].bias - back_prop[i * 2+1] * alpha

    def save_model(self, dir_path):
        data = {}
        data['model'] = self
        with open(dir_path, 'wb') as file:
            pickle.dump(data, file)
    
    def load_model(self, dir_path):
        with open(dir_path, 'rb') as file:
            data = pickle.load(file)
            self = data['model']