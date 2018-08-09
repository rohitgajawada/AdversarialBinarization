import torch
import torch.nn as nn

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(256, 256, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, 2, padding=1)

        self.linear = nn.Linear(512, 1)

    def forward(self, output):
        output = self.lrelu(self.conv2(output))
        output = self.lrelu(self.conv3(output))
        output = output.view(-1, 512)
        output = self.linear(output)
        return output

# netD = netD()
# netD.cuda()
#
# from torch.autograd import Variable
#
# h2 = Variable(torch.randn(1, 256, 4, 4)).cuda()
# out = netD(h2)
# print(out.size())
