import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

weights1 = np.array([[0.5], [-0.2], [0.3], [0.7], [-0.5], [0.4], [-0.6], [0.1]])
weights1 = weights1.T
biases1 = np.zeros((1, 8))

weights2 = np.random.randn(8, 1) * 0.5
biases2 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

layer1_output = relu(np.dot(X, weights1) + biases1)

layer2_output = tanh(np.dot(layer1_output, weights2) + biases2)

plt.plot(X, y, label="True Sine Wave", color="blue")
plt.plot(X, layer2_output, label="NN Output", color="red")
plt.legend()
plt.title("Sine Wave Approximation using Neural Network")
plt.show()
