import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

'''
Read in and prepare the MNIST and CIFAR-10 datasets.
Separate file for medical data.

For MNIST data:
https://nextjournal.com/gkoehler/pytorch-mnist

Note that for MNIST: the values 0.1307 and 0.3081 used for the Normalize() transformation below are the 
global mean and standard deviation of the MNIST dataset, we'll take them as a given here.


For CIFAR-10 data:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

MNIST = True
CIFAR = False


if MNIST:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    trainset = torchvision.datasets.MNIST(root = './data', train=True, download=True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

    testset = torchvision.datasets.MNIST('/files/', train=False, download=True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)

if CIFAR:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
