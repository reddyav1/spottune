"""
This script supports experiments on VisDA related to sim2real generalization
with and without targeting. We are trying to show that targeting can improve
generalization by preserving the pre-trained model's ability to process
real data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import wandb
import argparse
from pathlib import Path


from main import get_model
import agent_net
from visda17 import get_visda_dataloaders, downsample_dataset
from utils import AverageMeter
from gumbel_softmax import *

# load the resnet26 model (optionally with IN weights)
# make a training dataset with synthetic VisDA data
# make a validation dataset with real ViSDA test
# do spottune training (or optionally turn it off)

def train_epoch(train_loader, net, optimizer, agent=None, agent_optimizer=None, log_interval=10):
    net.train()
    if agent:
        agent.train()

    device = next(net.parameters()).device

    total_step = len(train_loader)
    train_acc = AverageMeter()
    train_loss = AverageMeter()

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        
        images, labels = images.to(device=device), labels.to(device=device)  

        if agent:
            policy = get_policy(agent, images)
            outputs = net.forward(images, policy)
        else:
            outputs = net.forward(images)
        
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        train_acc.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        train_loss.update(loss.item(), labels.size(0))

        if batch_idx % log_interval == 0:
            wandb.log({'train_acc': train_acc.val, 'train_loss': train_loss.val})

        # Backward and optimize
        optimizer.zero_grad()
        if agent:
            agent_optimizer.zero_grad()

        loss.backward()  

        optimizer.step()
        if agent:
            agent_optimizer.step()

def validate_epoch(val_loader, net, agent=None):
    net.eval()
    if agent:
        agent.eval()

    device = next(net.parameters()).device

    val_acc = AverageMeter()

    for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
        images, labels = images.to(device=device), labels.to(device=device)

        if agent:
            policy = get_policy(agent, images)
            outputs = net.forward(images, policy)
        else:
            outputs = net.forward(images)
        
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        val_acc.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))


    wandb.log({'val_acc': val_acc.avg})

def setup_network(net, spottune_enabled=False):
    # freeze the original blocks
    flag = True
    if spottune_enabled:
        for name, m in net.named_modules():
            # freeze conv layers that don't have "parallel_block", except first one
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

def get_policy(agent, inputs):
    probs = agent(inputs)
    action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
    policy = action[:,:,1]

    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--initialization', default='pretrained_models/resnet26_pretrained.t7')
    parser.add_argument('--n_epochs', type=int, default=110)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_agent', type=float, default=0.01)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--spottune', action='store_true', default=False)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='checkpoints/')

    args = parser.parse_args()

    run = wandb.init(
        project='targeted-generalization',
        config=vars(args)
    )

    n_classes = 12 # visda
    net = get_model("resnet26", n_classes, dataset=args.initialization)
    setup_network(net, spottune_enabled=args.spottune)

    device = torch.device('cuda:' + str(args.gpu_idx))
    net = net.to(device=device)

    train_loader, val_loader, test_loader = get_visda_dataloaders(
        train_dir='/cis/net/r32/data/visda17/train',
        val_dir='/cis/net/r32/data/visda17/validation',
        test_dir='/cis/net/r32/data/visda17/test',
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        train_fraction=args.train_fraction
    )

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    if args.spottune:
        print("using spottune!")
        agent = agent_net.resnet(sum(net.layer_config) * 2)
        agent = agent.to(device=device)
        agent_optimizer = optim.SGD(agent.parameters(), 
            lr= args.lr_agent,
            momentum=0.9, 
            weight_decay=0.001) 
        agent_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            agent_optimizer,
            T_max=args.n_epochs)
    else:
        print("NOT using spottune!")
        agent = None
        agent_optimizer = None
        agent_scheduler = None

    validate_epoch(test_loader, net, agent)
    for epoch in trange(args.n_epochs):
        train_epoch(train_loader, net, optimizer, agent=agent, agent_optimizer=agent_optimizer, log_interval=args.log_interval)
        validate_epoch(test_loader, net, agent)

        if args.save_freq > 0:
            if epoch % args.save_freq == 0:
                torch.save(
                    {'net': net, 'agent': agent},
                    Path(args.save_path) / (run.name + '_epoch' + str(epoch) + '.ckpt')
                )

        wandb.log({'net_lr': scheduler.get_last_lr()[0]})
        scheduler.step()

        if args.spottune:
            wandb.log({'agent_lr': agent_scheduler.get_last_lr()[0]})
            agent_scheduler.step()