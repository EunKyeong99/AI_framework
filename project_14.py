import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        '''
        :param in_channels: 입력 채널 수
        :param out_channels: 출력 채널 수
        :param hidden_dim: hidden 채널 수
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x) #저차원 특징 생성
        x = self.relu(x)
        x = self.conv2(x) #저차원 특징 활용 고차원 특징 생성
        x = self.relu(x)
        x = self.pool(x) #분류를 위해 특별한 특징만 거르기

        return x

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.block1 = BasicBlock(in_channels=3, out_channels=64, hidden_dim=32)
        self.block2 = BasicBlock(in_channels=64, out_channels=128, hidden_dim=64)
        self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)  # 저차원 특징 생성
        x = self.relu(x)
        x = self.fc2(x)  # 저차원 특징 활용 고차원 특징 생성
        x = self.relu(x)
        x = self.fc3(x)

        return x
'''
ex) 손글씨 데이터셋
input : 28x28 이미지
class : 10개 (0,1,2,3,4,5,6,7,8,9)
'''

input_size = 28*28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.005



t = Compose([
    RandomCrop((32,32), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4917, 0.4822, 0.4465),
              std=(0.247, 0.243, 0.261))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=t, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=t, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('epoch {}/{}, step {}/{}, Loss: {:4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print("acc {}%".format(100*correct/total))

#Tip) hidden_size를 2의 배수로 하면 정확도가 더 올라간다?