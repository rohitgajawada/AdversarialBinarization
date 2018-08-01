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

from models import nin, ninfprec, discriminator
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


def train(student, netD, teacher, student_optimizer, netD_optimizer, criterion, MSELoss, epoch, args, writer):

    n_critic = 1
    if args.losstype == 'wgangp':
        n_critic = 5

    student.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()

        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())

        student_optimizer.zero_grad()
        netD_optimizer.zero_grad()

        output, h1_student, h2_student = student(data)
        __, h1_teacher, h2_teacher = teacher(data)

        # discriminator backward
        adv_outputD_fake = netD(h1_student.detach(), h2_student.detach())
        adv_outputD_real = netD(h1_teacher.detach(), h2_teacher.detach())

        if args.losstype == 'wgangp':
            gradient_penalty = calc_gradient_penalty(netD, h1_teacher, h2_teacher, h1_student, h2_student)
            disc_adv_loss = torch.mean(adv_outputD_fake - adv_outputD_real) + gradient_penalty
        elif args.losstype == 'lsgan':
            labsize = adv_outputD_real.size()
            labels_real = Variable(torch.ones(labsize)).cuda() + Variable(torch.rand(labsize).cuda() * 0.1 - 0.05)
            labels_fake = Variable(torch.zeros(labsize)).cuda() + Variable(torch.rand(labsize).cuda() * 0.1 - 0.05)
            disc_adv_loss = 0.5 * (MSELoss(adv_outputD_fake, labels_fake) + MSELoss(adv_outputD_real, labels_real))


        disc_adv_loss.backward(retain_graph=True)
        netD_optimizer.step()

        # student backward
        if ((batch_idx + 1) % n_critic) == 0:

            # task component
            task_loss = criterion(output, target)

            # adversarial component
            if args.losstype == 'wgangp':
                gen_adv_loss = -1.0 * torch.mean(adv_outputD_fake)
                loss = (gen_adv_loss + 5 * task_loss) / 15.0
            elif args.losstype == 'lsgan':
                gen_adv_loss = MSELoss(adv_outputD_fake, labels_real)
                loss = gen_adv_loss + task_loss

            loss.backward()

            # restore weights
            bin_op.restore()
            bin_op.updateBinaryGradWeight()

            student_optimizer.step()

            # print(batch_idx, disc_adv_loss.data, gen_adv_loss.data, task_loss.data)
            writer.add_scalar('Disc Loss', disc_adv_loss.data.tolist()[0], batch_idx)
            writer.add_scalar('Gen Loss', gen_adv_loss.data.tolist()[0], batch_idx)
            writer.add_scalar('Task Loss', task_loss.data.tolist()[0], batch_idx)

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.data[0], student_optimizer.param_groups[0]['lr']))


def test(model, studflag=True):
    best_acc = 0
    test_loss = 0
    correct = 0

    model.eval()
    bin_op.binarization()
    for data, target in testloader:

        data, target = Variable(data.cuda()), Variable(target.cuda())
        output, h1_student, h2_student = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    bin_op.restore()
    acc = 100. * correct / len(testloader.dataset)

    if studflag == False:
        print("Teacher showing student")
    else:
        if acc > best_acc:
            best_acc = acc
            save_state(model, best_acc)

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    print('Best Accuracy: {:.2f}%\n'.format(best_acc))


def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='/home/rohit.gajawada/data',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--netD', action='store', default='basic',
            help='the architecture for the network: nin')

    parser.add_argument('--netDlr', action='store', default=0.001,
            help='the intial learning rate')
    parser.add_argument('--studlr', action='store', default=0.001,
            help='the intial learning rate')
    parser.add_argument('--losstype', action='store', default='lsgan',
            help='gan loss function')


    parser.add_argument('--teacher', action='store', default='./best_acc.pth',
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

    trainset = data.dataset(root=args.data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=2)

    testset = data.dataset(root=args.data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        student = nin.Net()
        student.cuda()

    if args.netD == 'basic':
        netD = discriminator.netD()
        netD.cuda()

    if not args.pretrainedstudent:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in student.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
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
    teacher = ninfprec.Net()
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
    MSELoss = torch.nn.MSELoss()

    student_optimizer = optim.Adam(student_params, lr=args.studlr, weight_decay=0.00001)
    netD_optimizer = optim.Adam(netD.parameters(), lr=args.netDlr, weight_decay=0.00001)

    # define the binarization operator
    bin_op = util.BinOp(student)

    # do the evaluation if specified
    if args.evaluate:
        test(student)
        exit(0)

    # start training
    test(teacher, False)
    print("Now testing dumb student")
    test(student)

    writer = SummaryWriter()

    for epoch in range(1, 320):
        adjust_learning_rate(student_optimizer, epoch)
        adjust_learning_rate(netD_optimizer, epoch)

        train(student, netD, teacher, student_optimizer, netD_optimizer, criterion, MSELoss, epoch, args, writer)
        test(student)
        sys.stdout.flush()
