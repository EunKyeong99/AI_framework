import numpy as np

class cross_entropy():
    def forward(self, predictions, targets):
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

class Activation_Softmax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.argmax(inputs, axis=1))
        probabilities = exp_values / np.sum(exp_values, axis=1)
        return probabilities

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.2, 0.2, 0.6]
])
targets = np.array([0, 1, 2])

activation_softmax = Activation_Softmax()
softmax_output = activation_softmax.forward(softmax_outputs)

cross_entropy = cross_entropy()
loss = cross_entropy.forward(softmax_outputs, targets)

print("Categorical Cross-Entropy Loss", loss)