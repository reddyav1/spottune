'''
This Script analyzes a trained SpotTune model to understand the fine-tuning decisions being made
'''

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
import shutil

# load spottuned model
# run testset through it
# print names of images along with number of fine-tuned blocks

load_path = 'spottune_visda_v2_best.ckpt'
device = 'cuda:0'

# Load the net and agent
ckpt = torch.load(load_path, map_location=device)
net = ckpt['net'].module
agent = ckpt['agent'].module

train_loader, val_loader, test_loader = get_visda_dataloaders(
    train_dir='/export/r32/data/visda17/train',
    val_dir='/export/r32/data/visda17/validation',
    test_dir='/export/r32/data/visda17/test',
)

net.eval()
agent.eval()

tasks_top1 = AverageMeter()
tasks_losses = AverageMeter() 

with torch.no_grad():
    for i, (images, labels, paths) in enumerate(test_loader):
        if torch.cuda.is_available():
            images, labels = images.to(device=device,non_blocking=True), labels.to(device=device, non_blocking=True)
        images, labels = Variable(images), Variable(labels)

        probs = agent(images)
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]

        outputs = net.forward(images, policy)

        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        print(f"Batch accuracy: {tasks_top1.avg}")

        finetune_counts = torch.sum(policy, dim=1)

        # focus on a class
        class_idx = 5
        finetune_counts = finetune_counts[labels==class_idx]
        paths = [paths[idx] for idx in (labels==class_idx).nonzero()]

        min_count, max_count = torch.min(finetune_counts), torch.max(finetune_counts)

        counts_of_interest = [min_count, min_count+1, max_count, max_count-1]

        for count in counts_of_interest:
            locations = (finetune_counts==count).nonzero()
            img_paths = [paths[idx] for idx in locations]
            print(f"paths that have {count} finetune blocks:")
            print(img_paths)

            dir_name = f"img_examples_best/knife/{int(count)}_blocks" 
            os.makedirs(dir_name, exist_ok=True)
            
            for img_path in img_paths:
                shutil.copy(img_path, dir_name)
        break
