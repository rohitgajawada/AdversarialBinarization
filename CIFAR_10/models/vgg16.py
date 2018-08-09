import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F

#    'D': [64, 64, 'M', 128,'D', 128, 'M', 256,'D', 256,'D', 256, 'M', 512,'D', 512,'D', 512, 'M', 512,'D', 512,'D', 512, 'M'],

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = BinConv2d(64, 64, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BinConv2d(64, 128, kernel_size=3, padding=1)
        self.dp3 = nn.Dropout2d(0.3)

        self.conv4 = BinConv2d(128, 128, kernel_size=3, padding=1)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = BinConv2d(128, 256, kernel_size=3, padding=1)
        self.dp5 = nn.Dropout2d(0.3)

        self.conv6 = BinConv2d(256, 256, kernel_size=3, padding=1)
        self.dp6 = nn.Dropout2d(0.3)

        self.conv7 = BinConv2d(256, 256, kernel_size=3, padding=1)
        self.mp7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = BinConv2d(256, 512, kernel_size=3, padding=1)
        self.dp8 = nn.Dropout2d(0.3)

        self.conv9 = BinConv2d(512, 512, kernel_size=3, padding=1)
        self.dp9 = nn.Dropout2d(0.3)

        self.conv10 = BinConv2d(512, 512, kernel_size=3, padding=1)
        self.mp10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = BinConv2d(512, 512, kernel_size=3, padding=1)
        self.dp11 = nn.Dropout2d(0.3)

        self.conv12 = BinConv2d(512, 512, kernel_size=3, padding=1)
        self.dp12 = nn.Dropout2d(0.3)

        self.conv13 = BinConv2d(512, 512, kernel_size=3, padding=1)
        self.mp13 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))

        x = self.mp2(self.relu(self.conv2(x)))
        x = self.dp3(self.relu(self.conv3(x)))
        x = self.mp4(self.relu(self.conv4(x)))
        x = self.dp5(self.relu(self.conv5(x)))
        x = self.dp6(self.relu(self.conv6(x)))
        h2 = self.mp7(self.relu(self.conv7(x)))
        #256 x 4 x 4

        x = self.dp8(self.relu(self.conv8(h2)))
        x = self.dp9(self.relu(self.conv9(x)))
        x = self.mp10(self.relu(self.conv10(x)))
        x = self.dp11(self.relu(self.conv11(x)))
        x = self.dp12(self.relu(self.conv12(x)))
        x = self.mp13(self.relu(self.conv13(x)))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, h2

from torch.autograd import Variable
x = Variable(torch.randn(1, 3, 32, 32)).cuda()
net = Net()
net = net.cuda()
out = net(x)
