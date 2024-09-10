import numpy as np

def init_weight(inputs):
    weights = np.random.uniform(-1, 1, inputs)
    return weights

def cal(inputs, weights, bias):
    inputs = np.array(inputs)
    weights = np.array(weights)
    output = np.dot(inputs, weights) + bias
    return output

def cal_neuron(num_neuron, inputs):
    outputs = []

    for i in range(num_neuron):
        weights = init_weight(inputs)
        bias = np.random.uniform(-1, 1)
        output = cal(inputs, weights, bias)
        outputs.append(cal(inputs, weights, bias))

    return outputs

num_neuron = int(input("Enter the number of neurons: "))
print("Outputs:", outputs)





