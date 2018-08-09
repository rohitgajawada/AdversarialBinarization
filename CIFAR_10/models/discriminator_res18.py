import torch
import torch.nn as nn

#((1L, 10L), (1L, 96L, 16L, 16L), (1L, 192L, 8L, 8L))

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(256, 256, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, 2, padding=1)

        self.linear = nn.Linear(512*2*2, 1)

    def forward(self, output):
        output = self.lrelu(self.conv2(output))
        output = self.lrelu(self.conv3(output))
        output = self.lrelu(self.conv4(output))

        output = output.view(-1, 512*2*2)
        output = self.linear(output)
        return output

# netD = netD()
# netD.cuda()
#
# from torch.autograd import Variable
# h1 = Variable(torch.randn(1, 96, 16, 16)).cuda()
# h2 = Variable(torch.randn(1, 192, 8, 8)).cuda()
#
# out = netD(h1, h2)
# print(out.size())
