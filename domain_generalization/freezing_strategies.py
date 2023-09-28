"""
Testing generalization performance of pre-trained ResNet-26 on VisDA real test
with various parameter freezing strategies 
"""

# Do linear probe with both synthetic training data and real training data

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
import wandb
import argparse

from main import get_model
from visda17 import get_visda_dataloaders
from targeted_synth_training import train_epoch, validate_epoch

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parser = argparse.ArgumentParser()

parser.add_argument('--ft_strategy', default='full')
parser.add_argument('--model', default='resnet26')
parser.add_argument('--init', default='imagenet12')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size_train', type=int, default=128)
parser.add_argument('--batch_size_test', type=int, default=256)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--train_first_conv', action='store_true', default=False)

args = parser.parse_args()

wandb.init(
        project='targeted-generalization',
        config=vars(args)
)

num_classes = 12

if args.model == 'resnet26':
    net = get_model('resnet26', num_class=num_classes, dataset='imagenet12' if not args.init == 'scratch' else None)
elif args.model == 'resnet18':
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, num_classes)
else:
    raise ValueError("Bad model name specified.")

n_params = count_trainable_params(net)
print(f"Network trainable parameter count: {n_params/1.0e6:.1f} M")
net.to(device='cuda:'+str(args.gpu_idx))

# Freeze params
strategy_to_name = { # maps ft strategy to unfrozen layer name components 
    'linear_probe': ['linear', 'fc'],
    'last-1': ['linear', 'fc', 'blocks.2.3'],
    'last-2': ['linear', 'fc', 'blocks.2.3', 'blocks.2.2'],
    'last-3': ['linear', 'fc', 'blocks.2.3', 'blocks.2.2', 'blocks.2.1'],
    'bn+last-3': ['linear', 'fc', 'bn', 'blocks.2.3', 'blocks.2.2', 'blocks.2.1'],
    'bn_only': ['linear', 'fc', 'bn'],
    'bn_nolinear': ['bn'],
    'full': [''],
}

# function that checks if a layer needs training
needs_training = lambda name: any(
    substring in name for substring in strategy_to_name[args.ft_strategy]
)

# if args.ft_strategy == 'linear_probe':
#     needs_training = lambda name : name in ['linear.weight', 'linear.bias', 'fc.weight', 'fc.bias']
# elif args.ft_strategy == 'last-1':
#     needs_training = lambda name : '2.3' in name
# elif args.ft_strategy == 'last-2':
#     needs_training = lambda name : '2.3' in name or '2.2' in name
# elif args.ft_strategy == 'last-3':
#     needs_training = lambda name : '2.3' in name or '2.2' in name or '2.1'
# elif args.ft_strategy == 'bn_only':
#     needs_training = lambda name : 'bn' in name
# elif args.ft_strategy == 'full':
#     needs_training = lambda name : True
# else:
#     ValueError("Invalid fine-tuning strategy.")

frozen = []
unfrozen = []
for name, m in net.named_parameters():
    if needs_training(name) or (name == 'conv1.weight' and args.train_first_conv):
        m.requires_grad = True
        unfrozen.append(name)
    else:
        m.requires_grad = False
        frozen.append(name)


n_trainable_params = count_trainable_params(net)
wandb.log({"trainable_params" : n_trainable_params})
print("frozen params: ", frozen)
print("\ntrainable params: ", unfrozen)
print("trainable param count: ", n_trainable_params)

train_loader, val_loader, test_loader = get_visda_dataloaders(
    train_dir='/export/r32/data/visda17/train',
    val_dir='/export/r32/data/visda17/validation',
    test_dir='/export/r32/data/visda17/test'
)

lr = 0.01
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=1e-4)

print("training...")
for i in range(args.n_epochs):
    train_epoch(train_loader, net, optimizer)
    print("validating...")
    validate_epoch(test_loader, net)