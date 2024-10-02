import numpy as np  # 행렬 연산 편하게
import nnfs  # 실험 환경 구축
from nnfs.datasets import vertical_data  # 세로로 배열된 데이터

nnfs.init()  # random seed 초기화


# Dense Layer
class Layer_Dense:
    # 초기화
    def __init__(self, n_inputs, n_neurons):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: 출력의 개수
        '''
        self.weights = 0.05 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward propagation
    def forward(self, inputs):
        '''
        :param inputs: 입력
        :return: forward propagation 완료된 값
        '''
        # y = ax + b
        return np.dot(inputs, self.weights) + self.biases


# ReLu activation
class Activation_ReLU:
    def forward(self, inputs):
        '''
        :param inputs: dense의 출력
        :return: relu 연산 완료 된 값
        '''
        return np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:
    # 결과값을 확율 처럼 만들고 싶을때
    def forward(self, inputs):
        '''
        :param inputs: dense의 output
        :return:  softmax 연산값
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # (300,3) 3개 짜리가 300개 있다. 3개짜리 뭉치 하나당 max를 구한다.
        probailities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # 현재 값 / 전체 값 ==> 전체에 대한 현재값의 비율 즉, 확률
        return probailities

    # 부모 클래스


# 상속 //
class Loss:
    def calculate(self, output, y):
        '''
        :param output: NN 출력
        :param y: 정답지
        :return: loss함수의 계산값
        '''
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):
    '''
    상속 : 클래스 이름(부모 클래스의 이름)
    '''

    def forward(self, y_pred, y_true):
        '''
        :param y_pred: NN 의 출력값
        :param y_true: 정답지
        :return: loss 계산값
        '''
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Create Dataset

X, y = vertical_data(samples=100, classes=10)
#  X : 300,2
#  Y : 3
# create model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 10)
activation2 = Activation_Softmax()
# loss
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
#class 10개 정확도 90%
for iteration in range(100000):

    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.01 * np.random.randn(3,10)
    dense2.biases += 0.01 * np.random.randn(1,10)

    out1 = activation1.forward(dense1.forward(X))
    out2 = activation2.forward(dense2.forward(out1))

    loss = loss_function.calculate(out2, y)

    predictions = np.argmax(out2, axis=1)
    # [0, 1, 0] => 1
    # [1, 0, 0] => 0
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print('New set of weights found. iteration', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()



