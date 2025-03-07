import torch
from torch import nn
import torch.nn.functional as F


class Image_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(6, 9, 5)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(6084, 128) ##256*256
        #self.fc1 = nn.Linear(1296, 128) ## 128*128
        #self.fc1 = nn.Linear(576, 128) ## 96*96
        #self.fc1 = nn.Linear(225, 128) ## 72*72
        self.fc2 = nn.Linear(128, 1)
        #self.bn1 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.conv2(x))
        x = self.pool(x)
        x = self.leakyRelu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

        
