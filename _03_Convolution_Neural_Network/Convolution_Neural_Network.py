# 在该文件NeuralNetwork类中定义你的模型
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(NeuralNetwork, self).__init__()

        # 定义卷积层和全连接层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def read_data():
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Resize((227, 227)),
                                                     torchvision.transforms.ToTensor()
                                                 ]))
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.Resize((227, 227)),
                                                   torchvision.transforms.ToTensor()
                                               ]))
    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

    return dataset_train, dataset_val, data_loader_train, data_loader_val # Assuming there are 10 classes

def main():
    model = torchvision.models.alexnet(pretrained=False, num_classes=10)
    dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()
    num_classes = 10
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for images, labels in data_loader_train:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader_val:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{10} | Accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), '../pth/model.pth')
    return model






