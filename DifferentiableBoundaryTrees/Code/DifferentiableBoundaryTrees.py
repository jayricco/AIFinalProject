import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.colors as color
import copy
import networkx as nx
plt.switch_backend(u'MacOSX')

class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        self.l1 = nn.Linear(2, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 30)
        self.l4 = nn.Linear(30, 2)
        self.tanh = torch.nn.ReLU()

    def forward(self, x):
        a1 = self.tanh(self.l1(x))
        a2 = self.tanh(self.l2(a1))
        a3 = self.tanh(self.l3(a2))
        a4 = self.l4(a3)
        return a4


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Transform(Bottle, TransformNetwork):
    pass


class DeepBoundaryTree(nn.Module):
    def __init__(self, k=-1):
        super(DeepBoundaryTree, self).__init__()
        self.mod = Transform()
        self.root = 0
        self.k = k
        self.count = 1
        self.data = {self.root: None}
        self.cnodes = {self.root: []}
        self.sm = nn.Softmax()
        self.lsm = nn.LogSoftmax()
        self.pd = torch.nn.PairwiseDistance()
        self.graph = nx.DiGraph()
        self.edge_labs = {}
        self.n_colors = []

    def clear_tree(self):
        self.count = 1
        del self.data
        del self.cnodes
        self.data = {self.root: None}
        self.cnodes = {self.root: []}
        self.graph.clear()
        self.edge_labs = {}
        self.n_colors = []

    def query_tree(self, x, training=False, internal=False):
        current = self.root
        qx = self.mod.forward(x)
        path = []
        last = None
        parent = None
        sib_sm = None
        classes_sib = None
        prob = None
        prob2 = None
        while True:
            neighborhood = [current] + self.cnodes[current]

            distances = torch.cat([self.pd(self.mod.forward(self.data[c][0]), qx) for c in neighborhood], 0).transpose(1, 0)
            if len(self.cnodes[current]) != 0:
                distances2 = torch.cat([self.pd(self.mod.forward(self.data[current][0]), self.mod.forward(self.data[c][0])) for c in self.cnodes[current]], 0).transpose(
                1, 0)
                classes2 = torch.cat([self.data[c][2] for c in self.cnodes[current]], 0)
            else:
                distances2 = None
                classes2 = None
            classes = torch.cat([self.data[c][2] for c in neighborhood], 0)
            sm_dist = F.log_softmax(-distances)#self.sm(-distances)
            min_dist, closest = torch.min(distances, 1)
            max_prob, closest_by_prob = torch.max(sm_dist, 1)
            if prob is None:
                prob = max_prob
            if prob is not None:
                prob += max_prob

            closest_node = neighborhood[closest_by_prob.int().data[0][0]]
            if closest_node == current:
                if distances2 is not None:
                    prob2 = torch.log(torch.sum(torch.mm(self.sm(-distances2).expand_as(classes2).t(), classes2), 1)).t()

                    print(prob2)
                if training and prob2 is not None:
                    prob = prob.expand_as(prob2).add(prob2)
                else:
                    prob = prob.expand_as(torch.zeros([1, 2]))
                break


            current = closest_node

        if training:

            return closest_node, prob
        else:
            if internal:
                return closest_node, min_dist
            return closest_node

    def train_tree(self, x, y, y_oh):

        # If the root hasn't been initialized yet
        if self.data[self.root] is None:
            self.data[self.root] = (x, y, y_oh, None)
            self.graph.add_node(str(self.root))
            self.n_colors.append(float(self.data[self.root][1].data[0]))
            return

        # Otherwise, train as normal.
        closest_node, min_dist = self.query_tree(x, internal=True)
        if not torch.equal(self.data[closest_node][1].data, y.data):
            self.data[self.count] = (x, y, y_oh, closest_node)
            self.cnodes[self.count] = []
            self.cnodes[closest_node].append(self.count)
            self.graph.add_node(str(self.count))
            e = (str(closest_node), str(self.count))
            self.graph.add_edge(e[0], e[1])
            self.edge_labs[e] = round(min_dist.data[0][0], 4)
            self.n_colors.append(float(self.data[self.count][1].data[0]))
            self.count += 1

    def forward(self, x):
        closest_node, prob = self.query_tree(x, training=True)
        return prob

# ==================