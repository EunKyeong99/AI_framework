import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

input_size = 1
output_size = 1
num_epochs = 30
learning_rate = 0.001

x_train = np.array([[60.0], [87.5], [70.0], [77.0], [101.2], [48.9], [51.2]], dtype=np.float32)
y_train = np.array([[165.0], [180.0], [168.0], [175.2], [177.2], [150.2], [151.3]], dtype=np.float32)

x_train = (x_train - np.mean(x_train))/np.std(x_train)
y_train = (y_train - np.mean(y_train))/np.std(y_train)

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('epoch {}/{}, Loss: {:4f}'.format(epoch+1, num_epochs, loss.item()))

predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.plot(x_train, predicted, label='Fitted Data')
plt.legend()
plt.show()