from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as tmp
from DifferentiableBoundaryTrees import DeepBoundaryTree
import HalfMoon
import MNIST
import MNIST_3D
from multiprocessing import Process, Lock, Array
from tensorflow.examples.tutorials.mnist import input_data

# Training Settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--k', type=int, default=-1, metavar='N',
                    help='k-value(default: -1)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=8473223, metavar='S',
                    help='random seed (default: 8473223)')

if __name__ == '__main__':
    args = parser.parse_args()
    #torch.manual_seed(289714)
    #torch.manual_seed(8464333)
    #torch.manual_seed(8473223)
    torch.manual_seed(args.seed)

    #dbt_model_mnist = DeepBoundaryTree([(784, 400), (400, 400), (400, 20)], k=-1)
    #MNIST.train(args, dbt_model_mnist)

    dbt_model_hm = DeepBoundaryTree()
    HalfMoon.train(args, dbt_model_hm)

