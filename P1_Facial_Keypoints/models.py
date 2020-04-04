## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # MaxPool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, 5) # Note: stride=1 by default

        # output_size = (Width - Filter(kernel_size) + 2*Padding) / Stride + 1
        # output_size = (224 - 5 + 0) / 1 + 1 = 220
        # >>> (32, 220, 220)
        # after pooling:
        # >>> (32, 110, 110)

        self.conv2 = nn.Conv2d(32, 64, 5)

        # output_size = (110 - 5 + 0) / 1 + 1 = 106
        # >>> (64, 106, 106)
        # after pooling:
        # >>> (64, 53, 53)

        self.conv3 = nn.Conv2d(64, 128, 5)

        # output_size = (53 - 5 + 0) / 1 + 1 = 49
        # >>> (128, 49, 49)
        # after pooling:
        # >>> (128, 24, 24)

        self.conv4 = nn.Conv2d(128, 256, 5)

        # output_size = (24 - 5 + 0) / 1 + 1 = 20
        # >>> (256, 20, 20)
        # after pooling:
        # >>> (256, 10, 10)

        self.conv5 = nn.Conv2d(256, 512, 5)

        # output_size = (10 - 5 + 0) / 1 + 1 = 6
        # >>> (512, 6, 6)
        # after pooling:
        # >>> (512, 3, 3)

        self.conv6 = nn.Conv2d(512, 1028, 3)

        # output_size = (3 - 3 + 0) / 1 + 1 = 1
        # >>> (1028, 1, 1)

        # Linear layers
        self.fc1 = nn.Linear(1028 * 1 * 1, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 136)

        # Dropout
        self.drop_conv1 = nn.Dropout(p=0.4)
        self.drop_conv2 = nn.Dropout(p=0.4)
        self.drop_conv3 = nn.Dropout(p=0.4)
        self.drop_conv4 = nn.Dropout(p=0.5)
        self.drop_conv5 = nn.Dropout(p=0.5)
        self.drop_conv6 = nn.Dropout(p=0.5)

        self.drop_fc1 = nn.Dropout(p=0.4)
        self.drop_fc2 = nn.Dropout(p=0.5)
        self.drop_fc3 = nn.Dropout(p=0.6)

        # Batch normalizations
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.bn_conv3 = nn.BatchNorm2d(128)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_conv5 = nn.BatchNorm2d(512)
        self.bn_conv6 = nn.BatchNorm2d(1028)

        self.bn_fc = nn.BatchNorm1d(1000)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop_conv1(self.pool(F.relu(self.bn_conv1(self.conv1(x)))))
        x = self.drop_conv2(self.pool(F.relu(self.bn_conv2(self.conv2(x)))))
        x = self.drop_conv3(self.pool(F.relu(self.bn_conv3(self.conv3(x)))))
        x = self.drop_conv4(self.pool(F.relu(self.bn_conv4(self.conv4(x)))))
        x = self.drop_conv5(self.pool(F.relu(self.bn_conv5(self.conv5(x)))))
        x = self.drop_conv6(F.relu(self.bn_conv6(self.conv6(x))))

        x = x.view(x.size(0), -1)

        # Linear layers
        x = self.drop_fc1(F.relu(self.bn_fc(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc(self.fc2(x))))
        x = self.drop_fc3(F.relu(self.bn_fc(self.fc3(x))))
        x = self.fc4(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
