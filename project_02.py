import numpy as np

inputs = [
    [2.0, 4.0, 6.0, 8.5],
    [-2.0, 5.0, 1.0, -2.0],
    [-1.5, 2.7, 3.3, 0.8]
]
weights = [
    [-0.2, 0.8, -0.5, 1.0],
    [0.6, 0.91, -0.26, -0.5],
    [-0.26, 0.27, 0.27, 0.87]
]

biases = [2.0, 3.0, 0.5]

weight2 = [
    [-0.1, 0.7, -0.5],
    [0.3, 0.51, -0.26],
    [-0.26, 0.27, 0.17]
]

biase2 = [1.0, 2.0, 0.7]

layers_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layers_outputs, np.array(weight2).T) + biase2

print(layers_outputs)
print(layer2_outputs)
