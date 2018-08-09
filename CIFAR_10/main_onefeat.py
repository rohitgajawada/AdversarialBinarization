from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from models import nin, ninfprec, vgg16, vgg16fprec, res18, res18fprec
from models import discriminator_normal, discriminator_vgg, discriminator_res18
from util import calc_gradient_penalty
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def save_state(student, best_acc):
    print('==> Saving models ...')
    state = {
            'best_acc': best_acc,
            'student_state_dict': student.state_dict()
            }
    torch.save(state, 'models/all.pth')


def train(student, netD, teacher, student_optimizer, netD_optimizer, criterion, GANLoss, epoch, args, writer):

    student.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()

        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())

        student_optimizer.zero_grad()
        netD_optimizer.zero_grad()

        output, h2_student = student(data)
        __, h2_teacher = teacher(data)

        # discriminator backward
        adv_outputD_fake = netD(h2_student.detach())
        adv_outputD_real = netD(h2_teacher.detach())

        #L2 Loss
        l2_loss = (h2_student - h2_teacher) ** 2
        l2_loss = l2_loss.sum(dim=3).sum(dim=2).sum(dim=1).sum(dim=0)
        l2_loss /= (h2_student.size(0) * h2_student.size(1) * h2_student.size(2) * h2_student.size(3))

        #Adv Loss
        if args.losstype == 'wgangp':
            gradient_penalty = calc_gradient_penalty(netD, h2_teacher, h2_student)
            disc_adv_loss = torch.mean(adv_outputD_fake - adv_outputD_real) + gradient_penalty

        elif args.losstype == 'lsgan' or args.losstype == 'gan':
            labsize = adv_outputD_real.size()
            labels_real = Variable(torch.ones(labsize)).cuda()
            labels_fake = Variable(torch.zeros(labsize)).cuda()
            disc_adv_loss = 0.5 * (GANLoss(adv_outputD_fake, labels_fake) + GANLoss(adv_outputD_real, labels_real))

        disc_adv_loss.backward(retain_graph=True)
        netD_optimizer.step()

        # student backward
        # task component
        task_loss = criterion(output, target)

        # adversarial component
        if args.losstype == 'wgangp':
            gen_adv_loss = -1.0 * torch.mean(adv_outputD_fake)
            loss = (task_loss + args.advweight * gen_adv_loss + args.l2weight * l2_loss) / (args.advweight + args.l2weight + 1.0)

        elif args.losstype == 'lsgan' or args.losstype == 'gan':
            gen_adv_loss = GANLoss(adv_outputD_fake, labels_real)
            loss = (task_loss + args.advweight * gen_adv_loss + args.l2weight * l2_loss) / (args.advweight + args.l2weight + 1.0)

        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        student_optimizer.step()

        # print(batch_idx, disc_adv_loss.data, gen_adv_loss.data, task_loss.data)
        iternum = (epoch - 1) * len(trainloader) + batch_idx

        writer.add_scalar('Disc Loss', disc_adv_loss.data.tolist()[0], iternum)
        writer.add_scalar('Gen Loss', gen_adv_loss.data.tolist()[0], iternum)
        writer.add_scalar('Task Loss', task_loss.data.tolist()[0], iternum)
        writer.add_scalar('L2 Loss', l2_loss.data.tolist()[0], iternum)
        # print(epoch, disc_adv_loss.data.tolist()[0], gen_adv_loss.data.tolist()[0], task_loss.data.tolist()[0])

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.data[0], student_optimizer.param_groups[0]['lr']))


def test(model, best_acc, writer, studflag=True, flag="normal"):
    test_loss = 0.0
    correct = 0

    model.eval()
    bin_op.binarization()
    for data, target in testloader:

        data, target = Variable(data.cuda()), Variable(target.cuda())
        output, h2_student = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    bin_op.restore()
    acc = 100. * correct / len(testloader.dataset)

    if flag != "demo":
        writer.add_scalar('Accuracy', acc, epoch)

    if studflag == False:
        print("Teacher showing student")
    else:
        print(acc, best_acc)
        if acc > best_acc:
            best_acc = acc
            save_state(model, best_acc)

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

    return best_acc


def adjust_learning_rate(optimizer, epoch):
    update_list = [1000, 1500, 1750, 2000]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='/home/rohit.gajawada/data',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='res18',
            help='the architecture for the network: nin')
    parser.add_argument('--netD', action='store', default='res18',
            help='the architecture for the network: nin')

    parser.add_argument('--netDlr', action='store', default=0.001,
            help='the intial learning rate')
    parser.add_argument('--studlr', action='store', default=0.001,
            help='the intial learning rate')
    parser.add_argument('--losstype', action='store', default='lsgan',
            help='gan loss function')
    parser.add_argument('--advweight', type=float, default=1.0)
    parser.add_argument('--l2weight', type=float, default=1.0)

    parser.add_argument('--teacher', action='store', default='./res18.t7',
        help='the path to the pretrained full prec teacher')

    parser.add_argument('--pretrainedstudent', action='store', default=None,
            help='the path to the pretrained student')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')


    args = parser.parse_args()
    print('==> Options:',args)

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    if not os.path.isfile(args.data+'/train_data'):
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=6)


    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        student = nin.Net()
        student.cuda()
    if args.arch == 'vgg':
        student = vgg16.Net()
        student.cuda()
    if args.arch == 'res18':
        student = res18.ResNet18()
        student.cuda()

    if args.netD == 'vgg':
        netD = discriminator_vgg.netD()
        netD.cuda()
    elif args.netD == 'nin':
        netD = discriminator_normal.netD()
        netD.cuda()
    elif args.netD == 'res18':
        netD = discriminator_res18.netD()
        netD.cuda()

    if not args.pretrainedstudent:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in student.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                # m.bias.data.zero_()
        for d in netD.modules():
            if isinstance(m, nn.Conv2d):
                d.weight.data.normal_(0, 0.05)
                d.bias.data.zero_()
    else:
        print('==> Load pretrained student from', args.pretrainedstudent, '...')
        pretrained_student = torch.load(args.pretrainedstudent)
        best_acc = pretrained_student['best_acc']
        student.load_state_dict(pretrained_student['state_dict'])

    #load pretrained teacher
    if args.arch == 'nin':
        teacher = ninfprec.Net()
        teacher_state_dict = teacher.state_dict()
    elif args.arch == 'vgg':
        teacher = vgg16fprec.Net()
        teacher_state_dict = teacher.state_dict()
    elif args.arch == 'res18':
        teacher = res18fprec.ResNet18()
        teacher_state_dict = teacher.state_dict()


    pretrained_teacher = torch.load(args.teacher)
    pretrain_state_dict = pretrained_teacher['net']

    for i, keys in enumerate(zip(teacher_state_dict.keys(), pretrain_state_dict.keys())):
        newkey, oldkey = keys
        print("Transferring ", newkey, " to ", oldkey)
        teacher_state_dict[newkey] = pretrain_state_dict[oldkey]

    teacher.load_state_dict(teacher_state_dict)
    teacher.cuda()
    print("Teacher Loaded!")

    # define solver and criterion
    student_param_dict = dict(student.named_parameters())
    student_params = []

    for key, value in student_param_dict.items():
        student_params += [{'params':[value], 'lr': args.studlr, 'weight_decay':0.00001}]

    criterion = nn.CrossEntropyLoss()

    if args.losstype == 'gan':
        GANLoss = torch.nn.BCEWithLogitsLoss()
    elif args.losstype == 'lsgan':
        GANLoss = torch.nn.MSELoss()
    else:
        GANLoss = 0

    student_optimizer = optim.Adam(student_params, lr=args.studlr, weight_decay=0.00001)
    netD_optimizer = optim.Adam(netD.parameters(), lr=args.netDlr, weight_decay=0.00001)

    # define the binarization operator
    bin_op = util.BinOp(student)

    # do the evaluation if specified
    if args.evaluate:
        test(student)
        exit(0)

    best_acc = 0
    writer = SummaryWriter()

    # start training
    test(teacher, best_acc, writer, studflag=False, flag="demo")
    print("Now testing dumb student")
    test(student, best_acc, writer, flag="demo")

    for epoch in range(1, 2500):
        adjust_learning_rate(student_optimizer, epoch)
        adjust_learning_rate(netD_optimizer, epoch)

        train(student, netD, teacher, student_optimizer, netD_optimizer, criterion, GANLoss, epoch, args, writer)
        best_acc = test(student, best_acc, writer)
        sys.stdout.flush()

    writer.close()
