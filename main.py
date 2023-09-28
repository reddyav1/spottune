import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
import numpy as np
import json
import collections

# import imdbfolder as imdbfolder
from spottune_models import *
import models
import agent_net

from utils import *
from gumbel_softmax import *

from visda17 import get_visda_dataloaders

def train(train_loader, net, agent, net_optimizer, agent_optimizer):
    # torch.autograd.set_detect_anomaly(True)

    #Train the model
    net.train()
    agent.train()

    total_step = len(train_loader)
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()

    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]    

        if use_cuda:
            images, labels = images.to(device=device, non_blocking=True), labels.to(device=device, non_blocking=True)
        images, labels = Variable(images), Variable(labels)	   

        probs = agent(images)

        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]

        # MANIPULATE POLICY FOR DEBUG (REMOVE!)
        # policy = torch.ones_like(policy)
        # policy = torch.zeros_like(policy)        

        outputs = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        # Loss
        loss = criterion(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))

        if i % 50 == 0:
            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Acc: {:.4f}%, Acc Avg: {:.4f}%"
                .format(epoch+1, args.nb_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg))
       
        #---------------------------------------------------------------------#
        # Backward and optimize
        net_optimizer.zero_grad()
        agent_optimizer.zero_grad()

        loss.backward()  
        net_optimizer.step()
        agent_optimizer.step()
            
    return tasks_top1.avg , tasks_losses.avg

def validate(val_loader, net, agent):
    net.eval()
    agent.eval()

    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 

    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(val_loader):
            if use_cuda:
                images, labels = images.to(device=device, non_blocking=True), labels.to(device=device, non_blocking=True)
            images, labels = Variable(images), Variable(labels)

            probs = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]

            # MANIPULATE POLICY FOR DEBUG (REMOVE!)
            # policy = torch.ones_like(policy)
            # policy = torch.zeros_like(policy)   

            outputs = net.forward(images, policy)

            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
            # Loss
            loss = criterion(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))           

    print(f"validation accuracy: {tasks_top1.avg}")
    # print("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
    #     .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg))

    return tasks_top1.avg, tasks_losses.avg

def test(test_loader, net, agent):
    net.eval()
    agent.eval()

    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if use_cuda:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            images, labels = Variable(images), Variable(labels)

            probs = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]

            # MANIPULATE POLICY FOR DEBUG (REMOVE!)
            # policy = torch.ones_like(policy)
            # policy = torch.zeros_like(policy)     
            
            outputs = net.forward(images, policy)

            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
            # Loss
            loss = criterion(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))           

    print(f"test accuracy: {tasks_top1.avg}")
    # print("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
    #     .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg))

    return tasks_top1.avg, tasks_losses.avg

def load_weights_to_flatresnet(source, net, num_class, dataset):
    if source.endswith('.pth'):
        # do stuff
        net_old = torch.load(source, map_location='cpu')
        net_old = net_old.module
    else:
        checkpoint = torch.load(source, encoding="latin1")
        net_old = checkpoint['net']

    store_data = []
    t = 0
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)
            t += 1

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
            m.running_var = store_data_rv[element].clone()
            m.running_mean = store_data_rm[element].clone()
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1
    
    del net_old
    return net

def get_model(model, num_class, dataset=None):
    if model == 'resnet26':
        rnet = resnet26(num_class)
        if dataset is not None:
            if dataset == 'imagenet12':
                source = './resnet26_pretrained.t7'
            else:
                source = dataset
            rnet = load_weights_to_flatresnet(source, rnet, num_class, dataset)
    return rnet

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch SpotTune')

    parser.add_argument('--nb_epochs', default=110, type=int, help='nb epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate of net')
    parser.add_argument('--lr_agent', default=0.01, type=float, help='initial learning rate of agent')

    parser.add_argument('--datadir', default='./data/decathlon-1.0/', help='folder containing data folder')
    parser.add_argument('--imdbdir', default='./data/decathlon-1.0/annotations/', help='annotation folder')
    parser.add_argument('--ckpdir', default='./cv/', help='folder saving checkpoint')

    parser.add_argument('--seed', default=0, type=int, help='seed')

    parser.add_argument('--step1', default=40, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=60, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')

    args = parser.parse_args()

    weight_decays = [
        ("aircraft", 0.0005),
        ("cifar100", 0.0),
        ("daimlerpedcls", 0.0005),
        ("dtd", 0.0),
        ("gtsrb", 0.0),
        ("omniglot", 0.0005),
        ("svhn", 0.0),
        ("ucf101", 0.0005),
        ("vgg-flowers", 0.0001),
        ("imagenet12", 0.0001)]

    datasets = [
        ("aircraft", 0),
        ("cifar100", 1),
        ("daimlerpedcls", 2),
        ("dtd", 3),
        ("gtsrb", 4),
        ("omniglot", 5),
        ("svhn", 6),
        ("ucf101", 7),
        ("vgg-flowers", 8)]

    datasets = collections.OrderedDict(datasets)
    weight_decays = collections.OrderedDict(weight_decays)

    with open(args.ckpdir + '/weight_decays.json', 'w') as fp:
        json.dump(weight_decays, fp)

    return args

#####################################
# Prepare data loaders
# train_loaders, val_loaders, test_loaders, num_classes = imdbfolder.prepare_data_loaders(list(datasets.keys()), args.datadir, args.imdbdir, True)
if __name__ == "__main__":

    args = parse_arguments()
    num_classes = 12
    criterion = nn.CrossEntropyLoss()

    # for i, dataset in enumerate(datasets.keys()):

    i = 0
    dataset = 'vgg-flowers' 
    # print(dataset)

    device = 'cuda:0'

    pretrained_model_dir = args.ckpdir + dataset

    # if not os.path.isdir(pretrained_model_dir):
    #     os.mkdir(pretrained_model_dir)

    results = np.zeros((4, args.nb_epochs, len(num_classes)))
    # f = pretrained_model_dir + "/params.json"
    # with open(f, 'w') as fh:
    #     json.dump(vars(args), fh)     

    # num_class = num_classes[datasets[dataset]]
    n_classes = 12 # visda
    net = get_model("resnet26", n_classes, dataset='pretrained_models/visda_syn_pretrain_v2.pth') # TODO: make this configurable
    # net = get_model("resnet26", n_classes, dataset='imagenet12') # TODO: make this configurable

    agent = agent_net.resnet(sum(net.layer_config) * 2)

    # freeze the original blocks
    flag = True
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            if flag is True:
                flag = False
            else:
                m.weight.requires_grad = False

    # Display info about frozen conv layers
    conv_layers_finetune = [x[0] for x in net.named_modules() if isinstance(x[1], nn.Conv2d) and x[1].weight.requires_grad]
    conv_layers_frozen = [x[0] for x in net.named_modules() if isinstance(x[1], nn.Conv2d) and not x[1].weight.requires_grad]

    print(f"Finetuning ({len(conv_layers_finetune)}) conv layers:")
    print(conv_layers_finetune)

    print(f"Freezing ({len(conv_layers_frozen)}) conv layers:")
    print(conv_layers_frozen)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.to(device=device)
        agent.to(device=device)

        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        # net = nn.DataParallel(net, device_ids=[0])
        # agent = nn.DataParallel(agent, device_ids=[0])

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr= args.lr, momentum=0.9, weight_decay= weight_decays[dataset])
    agent_optimizer = optim.SGD(agent.parameters(), lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)

    start_epoch = 0
    best_test_acc = 0
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_net(optimizer, epoch, args)
        adjust_learning_rate_agent(agent_optimizer, epoch, args)

        # train_loader = train_loaders[datasets[dataset]]
        # val_loader = val_loaders[datasets[dataset]]

        # VisDA
        train_loader, val_loader, test_loader = get_visda_dataloaders(train_dir='data/visda17/train', val_dir='data/visda17/validation')

        st_time = time.time()
        train_acc, train_loss = train(val_loader, net, agent, optimizer, agent_optimizer)
        test_acc, test_loss = validate(test_loader, net, agent)

        # Record statistics
        # results[0:2,epoch,i] = [train_loss, train_acc]
        # results[2:4,epoch,i] = [test_loss, test_acc]

        print('Epoch lasted {0}'.format(time.time()-st_time))

        state = {
        'net': net,
        'agent': agent,
        }

        if test_acc > best_test_acc:
            print(f"Surpassed previous best validation accuracy of ({best_test_acc}).\nSaving model...")
            torch.save(state, 'spottune_visda_v3_best.ckpt') 
            best_test_acc = test_acc

    # do test (vgg-flowers only)
    # if test_loaders[datasets[dataset]] is not None: 
        # test(test_loaders[datasets[dataset]], net, agent)

    torch.save(state, 'spottune_visda_v2_latest.ckpt')
    np.save(pretrained_model_dir + '/statistics', results)
