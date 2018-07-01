import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import torch.nn as nn
import torch.optim as optim
import torch
import models.__init__ as init
import datasets.__datainit__ as init_data
from tensorboard_logger import Logger

parser = opts.myargparser()

def main():
    global opt, best_prec1
    opt = parser.parse_args()
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    opt.logdir = opt.logdir+'/'+opt.name
    logger = Logger(opt.logdir)

    best_prec1 = 0
    print(opt)

    # Initialize the model, criterion and the optimizer
    nets = init.load_model(opt)
    losses = []
    optims = []

    losses['taskloss'] = nn.NLLLoss().cuda()
    losses['ganloss'] = utils.losses(opt)
    if opt.mse:
        losses['mseloss'] = nn.MSELoss().cuda()

    optims['optimS'] = optim.Adam(nets['netS'].parameters(), lr = opt.S_lr, weight_decay = opt.weightDecay)
    optims['optimD'] = optim.Adam(nets['netD'].parameters(), lr = opt.D_lr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(nets['netS'], opt)
        utils.weights_init(nets['netD'], opt)

    # Setup trainer and validation
    trainer = train.Trainer(nets, optims, losses, opt, logger)
    validator = train.Validator(nets, losses, opt, logger)

    # Load model from a checkpoint if mentioned in opts
    if opt.resume:
        if os.path.isfile(opt.resume):
            nets, optims, opt, best_prec1 = init.resumer(opt, nets, optims)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Setup the train and validation data loaders
    dataloader = init_data.load_data(opt)
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optims, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate for S:", optims['optimS'].param_groups[0]["lr"], "Learning rate for D:", optims['optimD'].param_groups[0]["lr"])
        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)

        acc = validator.validate(val_loader, epoch, opt)
        best_prec1 = max(acc, best_prec1)
        if best_prec1 == acc:
            init.save_checkpoint(opt, nets, optims, best_prec1, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_prec1))

if __name__ == '__main__':
    main()
