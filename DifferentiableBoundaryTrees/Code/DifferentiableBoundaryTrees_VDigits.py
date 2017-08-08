import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as color
import networkx as nx
import time
import csv
plt.switch_backend(u'MacOSX')
plt.interactive(True)

class Distance(nn.Module):
    def forward(self, xj, xq):
        return F.pairwise_distance(x1=xj, x2=xq, p=2)


class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        self.l1 = nn.Linear(64, 100)
        self.l2 = nn.Linear(100, 59)

        self.l3 = nn.Linear(59, 60)
        self.l4 = nn.Linear(60, 3)
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
    def __init__(self, k=-1):
        super(DeepBoundaryTree, self).__init__()
        self.mod = Transform()
        self.k = k
        self.dist = Distance()
        self.root = 0
        self.count = 1
        self.data = {self.root: None}
        self.cnodes = {self.root: []}
        self.sm = nn.Softmax()
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
        while True:
            if training:
                if last is not None:
                    path.append(last)

            if self.k == -1 or len(self.cnodes[current]) < self.k:
                neighborhood = [current] + self.cnodes[current]
            else:
                neighborhood = self.cnodes[current]

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

    for current_k in [3, 5, 10, 20, 50, 100]:
        dir_name = './DBT_Digits/'
        time_str = time.strftime("%Y%m%d-%H%M%S")
        attr_str = "data_file-tb_" + str(20) + "-bpb_" + str(10) + "-"
        format_str = '.csv'
        f_name_tr = dir_name + attr_str + time_str + "_TRAIN" + format_str
        f_name_te = dir_name + attr_str + time_str + "_TEST" + format_str
        tr_fi = open(f_name_tr, 'w+b')
        te_fi = open(f_name_te, 'w+b')
        train_data = {"K": [], "Iter": [], "TrainTime": [], "Nodes": [], "BackPropTime": [],
                      "Loss": []}  # , "TestTime": [], "Accuracy": []}
        test_data = {"K": [], "NumTrain": [], "TrainTime": [], "Nodes": [], "TestTime": [], "Accuracy": []}

        train_writer = csv.DictWriter(tr_fi, train_data.keys())
        test_writer = csv.DictWriter(te_fi, test_data.keys())
        train_writer.writeheader()
        test_writer.writeheader()
        dataset = ds.load_digits(return_X_y=True)
        test_set = ds.load_digits()
        dbt = DeepBoundaryTree(k=current_k)
        criterion = nn.CrossEntropyLoss(size_average=False)
        optim = torch.optim.Adam(params=dbt.parameters(), lr=1e-4)
        loss_fig = plt.figure(figsize=(15, 5))
        ax2 = loss_fig.add_subplot(121, projection='3d')
        ax3 = loss_fig.add_subplot(122)
        tree_fig = plt.figure(figsize=(5, 5))
        tree_fig.add_axes()

        start = 0
        finish = 200
        i = 0
        loss_track = []
        while True:
            print("Training Tree...")

            r = range(start, finish)
            random.shuffle(r)
            start_time = time.clock()
            for (ind, index) in enumerate(r):
                X = Variable(torch.from_numpy(np.asarray([dataset[0][index, :]], dtype=np.float32)))
                Y = Variable(torch.from_numpy(np.asarray([dataset[1][index]])), requires_grad=False)
                y_oh = np.zeros([1, 10], dtype=np.float32)
                y_oh[0, Y.data[0]] = 1.0
                Y_oh = Variable(torch.from_numpy(y_oh))
                dbt.train_tree(X, Y, Y_oh)
            training_time = time.clock() - start_time
            print("# NODES IN TREE: %d" % dbt.count)


            # SHOW TREE
            if i == 0 or i % 4 == 0:
                x = np.zeros([1000, 64])
                y = np.zeros([1000])
                x_t = np.zeros([1000, 3])
                for tsi in range(1000):
                    X = Variable(torch.from_numpy(np.asarray([dataset[0][tsi, :]], dtype=np.float32)))
                    x[tsi, :] = dataset[0][tsi, :]
                    x_t[tsi, :] = dbt.mod.forward(X).data[0].numpy()
                    y[tsi] = dataset[1][tsi]

                ax2.clear()
                ax3.clear()
                ax2.scatter(x_t[:, 0], x_t[:, 1], marker='+', c=y/100.0, linewidth=0.5, cmap='hsv')
                ax2.view_init(elev=60., azim=(i % 360.0))
                ax2.set_title("Transformed Input Space")
                ax3.plot(np.asarray(loss_track, dtype=np.float32))
                ax3.set_title("Loss over Iterations")

                tree_fig.clf(keep_observers=True)
                p = nx.spring_layout(dbt.graph)
                nx.draw_networkx(dbt.graph, pos=p, node_color=dbt.n_colors, cmap='hsv', vmin=0.0,
                                 vmax=1.0)
                nx.draw_networkx_edge_labels(dbt.graph, pos=p, font_size=7, label_pos=0.5, edge_labels=dbt.edge_labs)
                loss_fig.savefig('%d.png'%i)


            print("Doing The Backprops...")
            dbt.train()
            r = range(finish, finish+10)
            random.shuffle(r)
            loss_sum = 0.0
            avg_loss_last = 0.0
            start_time = time.clock()
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
            bp_time = time.clock() - start_time
            avg_loss = loss_sum/len(r)
            train_data["K"] = dbt.k
            train_data["Iter"] = i
            train_data["TrainTime"] = training_time
            train_data["Nodes"] = dbt.count
            train_data["BackPropTime"] = bp_time
            train_data["Loss"] = avg_loss
            loss_track.append(avg_loss)
            if np.abs(avg_loss_last - avg_loss) < 0.001:
                break
            train_writer.writerow(train_data)
            print("AVG LOSS: %f @ Iter: %d" % (avg_loss, i))
            avg_loss_last = avg_loss
            dbt.clear_tree()
            if finish + 30 > 1000:
                start = 0
                finish = 200
            else:
                start = finish
                finish = start + 100
            i += 1
            print("--------------------------------------------")

        print("TESTING!")

        for n_e in [10, 50, 100, 1000]:
            start_time = time.clock()
            for i in range(n_e):
                X = Variable(torch.from_numpy(np.asarray([dataset[0][i, :]], dtype=np.float32)))
                Y = Variable(torch.from_numpy(np.asarray([dataset[1][i]])), requires_grad=False)
                y_oh = np.zeros([1, 10], dtype=np.float32)
                y_oh[0, Y.data[0]] = 1.0
                Y_oh = Variable(torch.from_numpy(y_oh))
                dbt.train_tree(X, Y, Y_oh)
            test_tt = time.clock() - start_time
            x = np.zeros([1000, 64])
            y = np.zeros([1000])
            x_t = np.zeros([1000, 3])
            accuracy = 0.0
            num_right = 0.0
            start_time = time.clock()
            for i in range(1000):
                X = Variable(torch.from_numpy(np.asarray([test_set[0][i, :]], dtype=np.float32)))
                x[i, :] = test_set[0][i, :]
                x_t[i, :] = dbt.mod.forward(X).data[0].numpy()
                y[i] = test_set[1][i]
                closest_node = dbt.query_tree(X)
                y_pred = dbt.data[closest_node][1].data[0]
                if y_pred == test_set[1][i]:
                    num_right += 1.0
                accuracy = (accuracy + (num_right/float(i+1)))/2.0
            true_testt = time.clock() - start_time
            print("TOTAL ACCURACY: %f" % (accuracy*100.0))
            test_data["K"] = dbt.k
            test_data["TrainTime"] = test_tt
            test_data["NumTrain"] = n_e
            test_data["Nodes"] = dbt.count
            test_data["TestTime"] = true_testt
            test_data["Accuracy"] = ((num_right / 1000) * 100.0)
            test_writer.writerow(test_data)
            ax2.clear()
            ax3.clear()
            ax2.scatter(x_t[:, 0], x_t[:, 1], marker='x', c=y, s=10, linewidth=0.5, cmap=color.ListedColormap(['b', 'r', 'g', 'k', 'y', 'm', 'b', 'r', 'g', 'm']))
            ax2.set_title("Transformed")
            ax3.plot(np.asarray(loss_track, dtype=np.float32))
            ax3.set_title("Loss over Iterations")
            tree_fig.clf(keep_observers=True)
            p = nx.spring_layout(dbt.graph)
            nx.draw_networkx(dbt.graph, pos=p, node_color=dbt.n_colors, cmap=color.ListedColormap(['b', 'r', 'g', 'k', 'y', 'm', 'b', 'r', 'g', 'm']), vmin=0.0,
                             vmax=1.0)
            nx.draw_networkx_edge_labels(dbt.graph, pos=p, font_size=7, label_pos=0.5, edge_labels=dbt.edge_labs)
