import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Denser:
    def __init__(self, n_inputs, n_neurons, initialize_method='xavier'):
        if initialize_method == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1/n_inputs)
        elif initialize_method == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2/n_inputs)
        elif initialize_method == 'gaussian':
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        else:
            self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))

        self.biases = np.zeros((n_neurons,))

    def forward(self, inputs):
        inputs = np.array(inputs)
        outputs = np.dot(inputs, self.weights) + self.biases
        return np.maximum(0, outputs)

X, y = spiral_data(samples=100, classes=3)

p1 = Layer_Denser(n_inputs=2, n_neurons=3)
p2 = Layer_Denser(n_inputs=3, n_neurons=5)

p1_xavier = Layer_Denser(n_inputs=2, n_neurons=3, initialize_method='xavier')
p2_xavier = Layer_Denser(n_inputs=3, n_neurons=5, initialize_method='xavier')

p1_he = Layer_Denser(n_inputs=2, n_neurons=3, initialize_method='he')
p2_he = Layer_Denser(n_inputs=3, n_neurons=5, initialize_method='he')

p1_gaussian = Layer_Denser(n_inputs=2, n_neurons=3, initialize_method='gaussian')
p2_gaussian = Layer_Denser(n_inputs=3, n_neurons=5, initialize_method='gaussian')

z = p1.forward(X)
q = p2.forward(z)

z_xavier = p1_xavier.forward(X)
q_xavier = p2_xavier.forward(z_xavier)

z_he = p1_he.forward(X)
q_he = p2_he.forward(z_he)

z_gaussian = p1_gaussian.forward(X)
q_gaussian = p2_gaussian.forward(z_gaussian)

print(q_xavier)
print(q_he)
print(q_gaussian)
