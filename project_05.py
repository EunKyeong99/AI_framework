import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class Activation_relu:
    def forward(self, inputs):
        return np.maximum(0, inputs)

X, y = spiral_data(samples=2, classes=3)

dense1 = Layer_Dense(2, 3)

dense_output = dense1.forward(X)

activation1 = Activation_relu()

activation_output = activation1.forward(dense_output)

print("Dense output:")
print(dense_output)

print("\nActivation (ReLU) output:")
print(activation_output)
