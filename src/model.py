# Load in relevant libraries, and alias where appropriate
import torch.nn as nn
import torch.nn.functional as F

# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        self.conv1=nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # self.bn1=nn.BatchNorm2d(6)
        self.relu1=nn.ReLU()
        self.mp1=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2=nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # self.bn2=nn.BatchNorm2d(16)
        self.relu2=nn.ReLU()
        self.mp2=nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.mp1(self.relu1(self.conv1(x)))
        out = self.mp2(self.relu2(self.conv2(out)))
        out = out.reshape(out.size(0), -1) ## [B, 400]
        out = self.fc(out)
        out = self.relu3(out)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        # return out
        return F.log_softmax(out, dim=1)
