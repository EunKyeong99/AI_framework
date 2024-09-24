import numpy as np

class Activation_step:
    def forward(self, inputs):
        return np.where(inputs >= 0, 1, 0)

class Activation_linear:
    def forward(self, inputs):
        return inputs

class Activation_sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class Activation_relu:
    def forward(self, inputs):
        return np.maximum(0, inputs)