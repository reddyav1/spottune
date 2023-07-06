import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from spottune_models import resnet26
from models import resnet26
from visda17 import get_visda_dataloaders

from torch.autograd import Variable
from utils import AverageMeter

from pathlib import Path
import os
import time

def train():
    net.train()

    total_step = len(train_loader)
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()

    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    

        if torch.cuda.is_available():
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        images, labels = Variable(images), Variable(labels)	   

        outputs = net.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        # Loss
        loss = criterion(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))

        if i % 50 == 0:
            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Acc: {:.4f}%, Acc Avg: {:.4f}%"
                .format(epoch+1, n_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg))

        #---------------------------------------------------------------------#
        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()  
        optimizer.step()

def validate():
    net.eval()

    tasks_top1 = AverageMeter()

    for i, task_batch in enumerate(val_loader):
        images = task_batch[0] 
        labels = task_batch[1]    

        if torch.cuda.is_available():
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        images, labels = Variable(images), Variable(labels)	   

        outputs = net.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

    print(f"validation accuracy: {tasks_top1.avg}")

# save path
save_path = 'pretrained_models/visda_syn_pretrain_v2.pth'

# training parameters
n_epochs = 20
lr_milestones = [5, 10, 15]
batch_size = 128
lr = 0.1
wd = 0.0001

n_classes = [12]
criterion = nn.CrossEntropyLoss()

# get dataloaders
train_loader, val_loader = get_visda_dataloaders(train_dir='data/visda17/train', val_dir='data/visda17/validation', batch_size=batch_size)

# create network and optimizer
net = resnet26(n_classes)

if torch.cuda.is_available:
    net.cuda()

    cudnn.benchmark = True
    net = nn.DataParallel(net, device_ids=[0])

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
# network training

start_epoch = 0
for epoch in range(start_epoch, start_epoch + n_epochs):
    # TODO: add learning rate scheduling
    start_time = time.time()
    train()
    validate()

    print('Epoch lasted {0}'.format(time.time()-start_time))

# save the pretrained model
save_dir = Path(save_path).parent
os.makedirs(save_dir, exist_ok=True)
torch.save(net, save_path)