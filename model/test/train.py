# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import copy
import time
import os
from model import three_view_net
from autoaugment import ImageNetPolicy
import yaml
import math
from shutil import copyfile
from utils import save_network
from data_read import ImageFolderTwo
import utils
from tripletloss import TripletLoss

version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='two_view', type=str, help='output model name')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir',default='./data/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=12, type=int, help='padding')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--warm_epoch', default=3, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--block', default='1,4', type=str, help='the num of block' )
parser.add_argument('--freeze', default='', type=str, help='stem,layer1,layer2' )
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

transform_train_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='symmetric'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='symmetric'),
    transforms.RandomAffine(180),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'satellite': transforms.Compose(transform_satellite_list) }


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = ImageFolderTwo(os.path.join(data_dir, 'satellite'), os.path.join(data_dir, 'drone'), data_transforms['satellite'], data_transforms['train'], instance_num=6)


dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)
dataset_sizes = len(image_datasets)
print(dataset_sizes)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model

def one_LPN_output(outputs, labels, criterion):
    sm = nn.Softmax(dim=1)
    score = 0
    loss = 0
    for output in outputs:
        score += sm(output)
        loss += criterion(output, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss 


def adjust_lr(epoch, epochs, opt_lr, swa_ratio=1.0):
    stop_epoch = int(epochs[1] * swa_ratio)
    start_lr = opt_lr * 0.01
    dlr = (opt_lr - start_lr) / (epochs[0] - 1)
    if epoch < epochs[0]:
        lr = start_lr + dlr * epoch

    if epoch >= epochs[0] and epoch <= stop_epoch:
        lr = 0.5 * opt_lr * (math.cos(math.pi * (epoch - epochs[0]) / (epochs[1] - epochs[0])) + 1)

    if epoch > stop_epoch:
        lr = 0.5 * opt_lr * (math.cos(math.pi * (stop_epoch - epochs[0]) / (epochs[1] - epochs[0])) + 1)
    return lr


def train_model(model, criterion, criterion_2, optimizer, num_epochs=120, swa_epoch=0, swa_ratio=1.0):
    if len(opt.freeze) > 0:
        for name, param in model.named_parameters():
            for l in opt.freeze.split(","):
                if l in name:
                    param.requires_grad = False

    since = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs + swa_epoch):
        dataloaders.dataset.shuffle_items()
        lr = adjust_lr(epoch, [opt.warm_epoch, num_epochs], opt.lr, swa_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.defaults['lr'] = lr

        print('Epoch {}/{}'.format(epoch, num_epochs + swa_epoch - 1))
        print('-' * 10)
        print("lr ", lr)
        
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0.0
        # Iterate over data.
        total_sample = 0
        for data in dataloaders:
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            now_batch_size, c, h, w = inputs.shape
            total_sample += now_batch_size

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs, features = model(inputs)
                preds, loss = one_LPN_output(outputs, labels, criterion)
                # triplet_loss = criterion_2(features[0], labels)
                # loss = (triplet_loss + loss) / (len(outputs) + 1)
                loss = loss / len(outputs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects / dataset_sizes
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print("total_sample", total_sample)

        save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

model = three_view_net(701, block=opt.block)
opt.nclasses = 701

print(model)
optimizer_ft = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)

######################################################################
# Train and evaluate
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./run.sh', dir_name+'/run.sh')
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')
# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss()
criterion_2 = TripletLoss(norm=True, margin=None)
num_epochs = 10

model = train_model(model, criterion, criterion_2, optimizer_ft, num_epochs=num_epochs)

