'''Train CIFAR10 with PyTorch.''' # Source: https://github.com/kuangliu/pytorch-cifar/tree/master
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from sam import SAM
from ssam import SSAM
import projgrad
import os
import argparse
from copy import deepcopy

from models import resnet
from utils import progress_bar

from models.model_directory import MODEL_GETTERS


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--ssam_lambda', default=0.5)
parser.add_argument('--model', default='resnet18')
parser.add_argument('--pretrained', default=True)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--dataset', default='cifar10')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)
    
    classes = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushrooms', 'oak_tree', 'oranges', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 
    'worm')

# Model
print('==> Building model..')
net = MODEL_GETTERS[args.model](num_classes=len(classes), pretrained=args.pretrained)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

start_epoch=0
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    opt_state_dict = checkpoint['opt']
    sch_state_dict = checkpoint['sch']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, last_epoch=start_epoch-1)    
elif args.optimizer == 'sam':
    optimizer = SAM(net.parameters(), optim.SGD, rho=0.05, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200, last_epoch=start_epoch-1)    
else:
    optimizer = SSAM(net.parameters(), optim.SGD, rho=0.05, lr=args.lr, momentum=0.9, weight_decay=5e-4, lam=args.ssam_lambda)
    if args.resume:
        optimizer.load_state_dict(opt_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200, last_epoch=start_epoch-1)    
    if args.resume:
        scheduler.load_state_dict(sch_state_dict)

# Training
def train(epoch):
    def enable_bn(model):
        if isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d):
            model.backup_momentum = model.momentum
            model.momentum = 0
        
    def disable_bn(model):
        if isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d):
            model.momentum = model.backup_momentum
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    if args.optimizer == 'ssam':
        avg_incr_sharp = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        enable_bn(net)
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        if args.optimizer == 'sam' or args.optimizer == 'ssam':
            inputs_, targets_ = deepcopy(inputs), deepcopy(targets)
            inputs_1, targets_1 = deepcopy(inputs), deepcopy(targets)
            inputs_2, targets_2 = deepcopy(inputs), deepcopy(targets)
        if args.optimizer == 'ssam':
            inputs_prep = deepcopy(inputs)

        outputs = net(inputs)
        # Copy of network to apply criterion to SAM update
        if args.optimizer == 'ssam':
            copy_of_net = deepcopy(net)
            copy_of_optimizer = SAM(copy_of_net.parameters(), optim.SGD, rho=0.05, lr=args.lr, momentum=0.9, weight_decay=5e-4)
            outputs_1 = copy_of_net(inputs_1)
        if args.optimizer == 'sam' or args.optimizer == 'ssam':
            if args.optimizer == 'ssam':
                loss_f = torch.mean(net(inputs_prep.cuda()))    
                loss_f.backward()
                optimizer.prep(zero_grad=True)
            loss = criterion(outputs, targets)
            if args.optimizer == 'ssam':
            # Apply SAM update to copies 
                loss_1 = criterion(outputs_1, targets_1)
                loss_1.backward()
                copy_of_optimizer.first_step(zero_grad=True)
                sam_loss = criterion(copy_of_net(inputs_2), targets_2) 
            loss.backward()
            if args.optimizer == 'ssam':
                optimizer.first_step(zero_grad=True, n_iter=5)
            else:
                optimizer.first_step(zero_grad=True)
            disable_bn(net)
            ssam_loss = criterion(net(inputs_), targets_)
            criterion(net(inputs_), targets_).backward()
            optimizer.second_step(zero_grad=True)
            if args.optimizer == 'ssam':
                avg_incr_sharp += ssam_loss.item() - sam_loss.item()
        elif args.optimizer == 'sgd':
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.optimizer != 'ssam':
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg Incr. Sharp %.3f'
                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, avg_incr_sharp/(batch_idx+1)))
    if args.optimizer == 'ssam':
        return train_loss, 100.*correct/total, avg_incr_sharp
    else:
        return train_loss, 100.*correct/total
        

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'opt': optimizer.state_dict(),
            'sch': scheduler.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}.pth')
        best_acc = acc
    return acc

for epoch in range(start_epoch, start_epoch+args.epochs):
    if args.optimizer == 'ssam':
        train_loss, train_acc, avg_incr_sharp = train(epoch)
    else:
        train_loss, train_acc = train(epoch)
    test_acc = test(epoch)
    scheduler.step()
    print("Epoch: ", epoch, "Best Acc: ", best_acc)
    with open(f"logs/model:{args.model}_pretrained:{args.pretrained}_dataset:{args.dataset}_opt:{args.optimizer}.txt", "a") as f:
        if epoch == 0:
            if args.optimizer == 'ssam':
                f.write("epoch, train_loss, train_acc, test_acc, avg_incr_sharp\n")
            else:
                f.write("epoch, train_loss, train_acc, test_acc\n")
        if args.optimizer == 'ssam':
            f.write(f"{epoch},{train_loss},{train_acc},{test_acc},{avg_incr_sharp}\n")
        else:
            f.write(f"{epoch},{train_loss},{train_acc},{test_acc}\n")
