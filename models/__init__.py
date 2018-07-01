import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import models.alexnet as alexnet
import models.resnet as resnet
import utils
import os

"""
Save the current state as a checkpoint
"""
def save_checkpoint(opt, nets, optimizers, best_acc, epoch):

    state = {
        'epoch': epoch + 1,
        'arch': opt.model_def,
        'state_dictS': nets['netS'].state_dict(),
        'state_dictD': nets['netD'].state_dict(),
        'best_prec1': best_acc,
        'optimS' : optim['optimS'].state_dict(),
        'optimD' : optim['optimD'].state_dict(),
    }
    filename = "savedmodels/" + opt.name + ".pth.tar"

    torch.save(state, filename)

"""
Resume from a given checkpoint
"""
def resumer(opt, nets, optims):

    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']

        nets['netS'].load_state_dict(checkpoint['state_dictS'])
        optims['netS'].load_state_dict(checkpoint['optimS'])
        nets['netD'].load_state_dict(checkpoint['state_dictD'])
        optims['netD'].load_state_dict(checkpoint['optimD'])

        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return nets, optims, opt, best_prec1

"""
Load a model from the specified opts
"""
def load_model(opt):
    if opt.pretrained_file != "":
        model = torch.load(opt.pretrained_file)
    else:
        if opt.model_def == 'alexnet':
            model = alexnet.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnet18':
            model = resnet.ResNet(opt.nClasses)
            if opt.cuda:
                model = model.cuda()


    return model
