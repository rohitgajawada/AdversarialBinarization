import torch.nn as nn
import torch.nn.init as init
import math

#    'D': [64, 64, 'M', 128,'D', 128, 'M', 256,'D', 256,'D', 256, 'M', 512,'D', 512,'D', 512, 'M', 512,'D', 512,'D', 512, 'M'],

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dp3 = nn.Dropout2d(0.3)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dp5 = nn.Dropout2d(0.3)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.dp6 = nn.Dropout2d(0.3)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.mp7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.dp8 = nn.Dropout2d(0.3)

        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.dp9 = nn.Dropout2d(0.3)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.mp10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.dp11 = nn.Dropout2d(0.3)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.dp12 = nn.Dropout2d(0.3)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.mp13 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv1(self.bn1(self.relu(x)))
        x = self.conv2(self.bn2(self.relu(self.mp2(x))))
        x = self.conv3(self.bn3(self.relu(self.dp3(x))))
        x = self.conv4(self.bn4(self.relu(self.mp4(x))))
        x = self.conv5(self.bn5(self.relu(self.dp5(x))))
        x = self.conv6(self.bn6(self.relu(self.dp6(x))))
        x = self.conv7(self.bn7(self.relu(self.mp7(x))))
        x = self.conv8(self.bn8(self.relu(self.dp8(x))))
        x = self.conv9(self.bn9(self.relu(self.dp9(x))))
        x = self.conv10(self.bn10(self.relu(self.mp10(x))))
        x = self.conv11(self.bn11(self.relu(self.dp11(x))))
        x = self.conv12(self.bn12(self.relu(self.dp12(x))))
        x = self.conv13(self.bn13(self.relu(self.mp13(x))))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.mp2(self.relu(self.bn2(self.conv2(x))))
        x = self.dp3(self.relu(self.bn3(self.conv3(x))))
        x = self.mp4(self.relu(self.bn4(self.conv4(x))))
        x = self.dp5(self.relu(self.bn5(self.conv5(x))))
        x = self.dp6(self.relu(self.bn6(self.conv6(x))))
        h2 = self.mp7(self.relu(self.bn7(self.conv7(x))))
        #256 x 4 x 4

        x = self.dp8(self.relu(self.bn8(self.conv8(h2))))
        x = self.dp9(self.relu(self.bn9(self.conv9(x))))
        x = self.mp10(self.relu(self.bn10(self.conv10(x))))
        x = self.dp11(self.relu(self.bn11(self.conv11(x))))
        x = self.dp12(self.relu(self.bn12(self.conv12(x))))
        x = self.mp13(self.relu(self.bn13(self.conv13(x))))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, h2
