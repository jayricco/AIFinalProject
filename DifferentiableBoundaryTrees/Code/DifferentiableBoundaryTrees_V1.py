import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.colors as color
plt.switch_backend(u'QT5Agg')


class Distance(nn.Module):
    def forward(self, xj, xq):
        return F.pairwise_distance(x1=xj, x2=xq, p=2)


class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        self.l1 = nn.Linear(2, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 30)
        self.l4 = nn.Linear(30, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        a1 = self.relu(self.l1(x))
        a2 = self.relu(self.l2(a1))
        a3 = self.relu(self.l3(a2))
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
    def __init__(self):
        super(DeepBoundaryTree, self).__init__()
        self.mod = Transform()
        self.dist = Distance()
        self.root = 0
        self.count = 1
        self.data = {self.root: None}
        self.cnodes = {self.root: []}
        self.sm = nn.Softmax()

    def clear_tree(self):
        self.count = 1
        del self.data
        del self.cnodes
        self.data = {self.root: None}
        self.cnodes = {self.root: []}

    def query_tree(self, x, training=False):
        current = self.root
        qx = self.mod.forward(x)
        path = []
        last = None
        while True:
            if training:
                if last is not None:
                    path.append(last)
            neighborhood = [current] + self.cnodes[current]

            distances = torch.cat([self.dist.forward(self.mod.forward(self.data[c][0]), qx) for c in neighborhood], 0).transpose(1, 0)
            classes = torch.cat([self.data[c][2] for c in neighborhood], 0)
            sm_dist = self.sm(-distances)
            if training:
                parent = self.data[current][3]
                if parent is not None:
                    sib_nbhd = [parent] + self.cnodes[parent]
                    sib_dist = torch.cat([self.dist.forward(self.mod.forward(self.data[c][0]), qx) for c in sib_nbhd], 0).transpose(1, 0)
                    sib_sm = self.sm(-sib_dist)
                    classes_sib = torch.cat([self.data[c][2] for c in sib_nbhd], 0)
            min_dist, closest = torch.min(distances, 1)
            max_prob, closest_by_prob = torch.max(sm_dist, 1)

            closest_node = neighborhood[closest.int().data[0][0]]
            if closest_node == current:
                if training:
                    if parent is not None:
                        path.append((sib_sm, classes_sib, 'Final'))
                    if parent is None and len(path) == 0:
                        path.append((sm_dist, classes, 'Final'))
                break
            if training:
                last = max_prob/torch.sum(sm_dist)
            current = closest_node
        if training:
            return closest_node, path
        else:
            return closest_node

    def train_tree(self, x, y, y_oh):
        # If the root hasn't been initialized yet
        if self.data[self.root] is None:
            self.data[self.root] = (x, y, y_oh, None)
            return
        # Otherwise, train as normal.
        closest_node = self.query_tree(x)
        if not torch.equal(self.data[closest_node][1].data, y.data):
            self.data[self.count] = (x, y, y_oh, closest_node)
            self.cnodes[self.count] = []
            self.cnodes[closest_node].append(self.count)
            self.count += 1

    def forward(self, x):
        closest_node, path = self.query_tree(x, training=True)
        probability_sum_term_1 = None
        probability_sum_term_2 = None
        for path_elem in path:
            if len(path_elem) == 1:
                if probability_sum_term_1 is None:
                    probability_sum_term_1 = torch.log(path_elem[0] + 0.0001)
                else:
                    probability_sum_term_1.add_(torch.log(path_elem[0] + 0.0001))
            elif path_elem[2] == "Final":
                probability_sum_term_2 = torch.log(torch.sum(torch.mm(path_elem[0], path_elem[1]), 0) + 0.0001)
                if probability_sum_term_1 is not None:
                    probability_sum_term_2.add_(probability_sum_term_1.expand_as(probability_sum_term_2))
        return probability_sum_term_2

# ======================================================================================================================

if __name__ == "__main__":
    import random
    import numpy as np

    dataset = ds.make_moons(n_samples=1000)
    test_set = ds.make_moons(n_samples=1000)

    dbt = DeepBoundaryTree()
    criterion = nn.CrossEntropyLoss(size_average=False)
    optim = torch.optim.Adam(params=dbt.parameters(), lr=1e-4)


    start = 0
    finish = 20
    i = 0
    loss_track = []
    while True:
        print("Training Tree...")

        r = range(start, finish)
        random.shuffle(r)
        for (ind, index) in enumerate(r):
            X = Variable(torch.from_numpy(np.asarray([dataset[0][index, :]], dtype=np.float32)))
            Y = Variable(torch.from_numpy(np.asarray([dataset[1][index]])), requires_grad=False)

            y_oh = np.zeros([1, 10], dtype=np.float32)
            y_oh[0, Y.data[0]] = 1.0
            Y_oh = Variable(torch.from_numpy(y_oh))
            dbt.train_tree(X, Y, Y_oh)
        print("# NODES IN TREE: %d" % dbt.count)

        print("Doing The Backprops...")

        dbt.train()
        r = range(finish, finish+10)
        random.shuffle(r)
        loss_sum = 0.0
        avg_loss_last = 0.0

        for (ind, index) in enumerate(r):
            X = Variable(torch.from_numpy(np.asarray([dataset[0][index, :]], dtype=np.float32)))
            Y = Variable(torch.from_numpy(np.asarray([dataset[1][index]])), requires_grad=False)
            y_oh = np.zeros([1, 10], dtype=np.float32)
            y_oh[0, Y.data[0]] = 1.0
            Y_oh = Variable(torch.from_numpy(y_oh))

            optim.zero_grad()
            out = dbt.forward(X)
            loss = criterion(out, Y.long()[0])
            loss_sum += loss.data[0]
            #print("LOSS %f | %d/%d" % (loss.data[0], iter, 1000))
            loss.backward()
            optim.step()

        avg_loss = loss_sum/len(r)
        loss_track.append(avg_loss)
        if np.abs(avg_loss_last - avg_loss) < 0.001:
            break

        print("AVG LOSS: %f @ Iter: %d" % (avg_loss, i))
        avg_loss_last = avg_loss
        dbt.clear_tree()
        if finish + 30 > 1000:
            start = 0
            finish = 20
        else:
            start = finish
            finish = start + 20
        i += 1
        print("--------------------------------------------")

    print("TESTING!")

    x = np.zeros([1000, 2])
    y = np.zeros([1000])
    x_t = np.zeros([1000, 2])
    accuracy = 0.0
    num_right = 0.0
    for i in range(1000):
        X = Variable(torch.from_numpy(np.asarray([test_set[0][i, :]], dtype=np.float32)))
        print(dbt.mod.forward(X).data[0].numpy())
        x[i, :] = test_set[0][i, :]
        x_t[i, :] = dbt.mod.forward(X).data[0].numpy()
        y[i] = test_set[1][i]
        closest_node = dbt.query_tree(X)
        y_pred = dbt.data[closest_node][1].data[0]
        if y_pred == test_set[1][i]:
            num_right += 1.0
        accuracy = (accuracy + (num_right/float(i+1)))/2.0

    print("TOTAL ACCURACY: %f" % (accuracy*100.0))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(x[:, 0], x[:, 1], marker='+', c=y, cmap=color.ListedColormap(['r', 'b']))
    ax1.set_title("Original")
    ax2.scatter(x_t[:, 0], x_t[:, 1], marker='+', c=y, cmap=color.ListedColormap(['r', 'b']))
    ax2.set_title("Transformed")
    ax3.plot(np.asarray(loss_track, dtype=np.float32))
    ax3.set_title("Loss over Iterations")
    plt.show()
