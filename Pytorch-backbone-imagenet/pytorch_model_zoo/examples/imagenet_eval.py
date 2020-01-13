from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import sys

sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils


model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="D:\data\yizhou\imagenet_raw",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_decay_step', default=30, type=int, metavar='N',
                    help='number of total step to decay learning rate')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint_path', default='D:/workspace/yizhou/train/ScaleCNN/imagenet/densenet/lr0.6_epc70_bs1200_p0.1_tmp1to5_ls80_wd1/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint and saving path (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do_not_preserve_aspect_ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0


def main():
    print(pretrainedmodels.__file__)
    global args, best_prec1
    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        print("=> using pre-trained parameters '{}'".format(args.pretrained))
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                     pretrained=args.pretrained)
    else:
        model = pretrainedmodels.__dict__[args.arch]()

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
        scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
    else:
        scale = 0.875

    def make_weights_for_balanced_classes(images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight
    #dataset_weight = make_weights_for_balanced_classes(datasets.ImageFolder(traindir).imgs, 1000)
    #dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_weight, num_samples=len(dataset_weight), replacement=True)

    train_tf = pretrainedmodels.utils.TransformImage(
        model,
        scale=scale,
        random_crop=True,
        random_hflip=True,
        random_vflip=False,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
        rotate=True, pixel_jitter=True
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, train_tf),
        batch_size=args.batch_size, shuffle=True, #sampler=dataset_sampler,
        num_workers=args.workers, pin_memory=True)

    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(max(model.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    '''

    print('Images transformed from size {} to {}'.format(
        int(round(max(model.input_size) / scale)),
        model.input_size))

    val_tf = pretrainedmodels.utils.TransformImage(
        model,
        scale=scale,
        preserve_aspect_ratio=args.preserve_aspect_ratio
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if torch.cuda.device_count() >= 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    if args.checkpoint_path:
        checkpoint_name = os.path.join(args.checkpoint_path, 'selected_checkpoint.pth.tar')
        #checkpoint_name_tmp = os.path.join(args.checkpoint_path, 'tmp_checkpoint.pth.tar')
        if os.path.isfile(checkpoint_name):
            print("=> loading checkpoint '{}'".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_name)
            '''
            checkpoint_tmp = torch.load(checkpoint_name_tmp)
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'unif_noise_variable' not in k}
            checkpoint_tmp['state_dict'] = {k: v for k, v in checkpoint_tmp['state_dict'].items() if 'unif_noise_variable' in k}

            model_dict = model.state_dict()
            model_dict.update(checkpoint['state_dict'])
            model_dict.update(checkpoint_tmp['state_dict'])
            model.load_state_dict(model_dict)
            '''
            args.start_epoch = checkpoint['epoch']
            best_prec1 = -1  # checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_name, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint_path))

    #if args.evaluate:
    #    validate(val_loader, model, criterion)
    #    return

    # validate before training
    validate(val_loader, model, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch, decay_step=args.lr_decay_step)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch % 2 == 0:
            # evaluate on validation set
            prec1, prec5 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1.item() > best_prec1
            best_prec1 = max(prec1.item(), best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.checkpoint_path, 'checkpoint.pth.tar'))



def selection():
    print(pretrainedmodels.__file__)
    global args, best_prec1
    args = parser.parse_args()

    valdir = os.path.join(args.data, 'val')
    val_tf = None
    val_loader = None

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for idx in range(500):
        torch.manual_seed(idx)
        np.random.seed(idx)

        # create model
        print("=> creating model '{}'".format(args.arch))
        if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
            print("=> using pre-trained parameters '{}'".format(args.pretrained))
            model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                         pretrained=args.pretrained)
        else:
            model = pretrainedmodels.__dict__[args.arch]()

        # Data loading code


        if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
            scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
        else:
            scale = 0.875

        if not val_tf:
            val_tf = pretrainedmodels.utils.TransformImage(
                model,
                scale=scale,
                preserve_aspect_ratio=args.preserve_aspect_ratio
            )
        if not val_loader:
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, val_tf),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()

        if torch.cuda.device_count() >= 1:
            print("Using", torch.cuda.device_count(), "GPUs.")
            model = torch.nn.DataParallel(model).cuda()

        if args.checkpoint_path:
            if os.path.isfile(args.checkpoint_path):
                print("=> loading checkpoint '{}'".format(args.checkpoint_path))
                checkpoint = torch.load(args.checkpoint_path)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.checkpoint_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.checkpoint_path))

        validate(val_loader, model, criterion, selection_mode=True)
    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model.state_dict(),
    }, False, filename=os.path.join(args.checkpoint_path, 'checkpoint.pth.tar'))

    return

def train(train_loader, model, criterion, optimizer, epoch, batch_size_multipler=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    crsentrpy_losses = AverageMeter()
    KLCD_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    '''
    with open('log_p_ls10_dense169_epc150.txt', 'a') as f:
        print('Epoch: [{0}]'.format(epoch))
        for param_tensor in model.state_dict():
            if 'p_logit' in param_tensor:
                print(param_tensor, "\t", model.state_dict()[param_tensor].sigmoid(), file=f)
    '''
    # switch to train mode
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        adjust_learning_rate_perstep(optimizer, cur_epoch=epoch, cur_step=i)
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        ##target_var = torch.Tensor(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        crsentrpy_losses.update(loss.detach().clone(), input.size(0))
        '''
        KL_CD = 0.0
        for idx, m in enumerate(model.modules()):
            if hasattr(m, 'KLreg'):
                KL_CD += m.KLreg
        KL_CD = KL_CD.cuda()
        
        KLCD_losses.update(KL_CD.detach(), input.size(0))
        '''
        loss += 0.0

        loss = loss / batch_size_multipler
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach(), target.detach(), topk=(1, 5))
        losses.update(loss.detach()*batch_size_multipler, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        loss.backward()
        if i % batch_size_multipler == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            with open('log_train_ls100_dense121_epc180_ft3x3_epc180.txt', 'a') as f:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Entropy Loss {entropy_loss.val:.4f} ({entropy_loss.avg:.4f})\t'
                      'KLCD_Loss {KLCD_loss.val:.4f} ({KLCD_loss.avg:.4f})\t'
                      'Learning Rate {lr:.8f}\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, entropy_loss=crsentrpy_losses, KLCD_loss=KLCD_losses,
                    lr=optimizer.param_groups[0]['lr'], top1=top1, top5=top5), file=f)


def validate(val_loader, model, criterion, selection_mode=False):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not selection_mode:
                if i % args.print_freq == 0:
                    with open('log_train_ls100_dense121_epc180_ft3x3_epc180.txt', 'a') as f:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                               i, len(val_loader), batch_time=batch_time, loss=losses,
                               top1=top1, top5=top5), file=f)

        if not selection_mode:
            with open('log_train_ls100_dense121_epc180_ft3x3_epc180.txt', 'a') as f:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5), file=f)
        else:
            with open('selection_ls10.txt', 'a') as f:
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5), file=f)

        return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if decay_step > 0:
        lr = args.lr * (0.1 ** (epoch // decay_step))
    else:
        if epoch < 90:
            lr = args.lr
        elif epoch >= 90 and epoch < 130:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_perstep(optimizer, cur_step, cur_epoch):
    """Sets the learning rate to the initial LR decayed by Cosine"""
    t = cur_epoch * (1281167.0 / args.batch_size) + cur_step
    T = 1281167.0 / args.batch_size * args.epochs
    lr = (0.5 * (1 + np.cos(t*np.pi/T))) * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
    main()
    #selection()