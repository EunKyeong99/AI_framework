import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Denser:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.random.uniform(0, 1, (n_neurons,))

    def forward(self, inputs):
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)

p1 = Layer_Denser(n_inputs=2, n_neurons=3)
p2 = Layer_Denser(n_inputs=3, n_neurons=5)

z = p1.forward(X)
q = p2.forward(z)



print(q)




