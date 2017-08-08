from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import csv
import sklearn.utils as utils
import time
from tensorflow.examples.tutorials.mnist import input_data

num_classes = 10
num_features = 784
num_outputs = 20
train_batch = 500
bp_batch = 100


dir_name = "./DBT_MNIST/"
model_fmt = '.pt'
time_str = time.strftime("%Y%m%d-%H%M%S")
attr_str = "data_file-tb_"+str(train_batch)+"-bpb_"+str(bp_batch)+"-"
format_str = '.csv'
f_name_tr = dir_name + attr_str + time_str + "_TRAIN" + format_str
f_name_te = dir_name + attr_str + time_str + "_TEST" + format_str
tr_fi = open(f_name_tr, 'w+b')
te_fi = open(f_name_te, 'w+b')
train_data = {"K": [], "Iter": [],  "TrainTime": [], "Nodes": [], "BackPropTime": [], "Loss": []}#, "TestTime": [], "Accuracy": []}
test_data = {"K": [], "NumTrain": [], "TrainTime": [], "Nodes": [], "TestTime": [], "Accuracy": []}

train_writer = csv.DictWriter(tr_fi, train_data.keys())
test_writer = csv.DictWriter(te_fi, test_data.keys())
mypid = os.getpid()


def train(args, model):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_set = zip(mnist.train.images, mnist.train.labels)
    test_set = zip(mnist.test.images, mnist.test.labels)
    #optimizer = torch.optim.SGD(params=model.mod.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()
    train_writer.writeheader()
    train_epoch(args, model, train_set, optimizer, criterion)
    tr_fi.close()

    test_writer.writeheader()
    test_epoch(model, train_set, test_set, test_sizes=[10, 100, 1000, 2000, 5000], test_ks=[-1, 2, 5, 10, 20, 50])
    te_fi.close()


def train_epoch(args, model, train_set, optimizer, criterion):
    i = 1
    loss_track = []
    y_onehot = torch.FloatTensor(1, num_classes)
    avg_loss_last = 0.0
    best_loss = float('inf')
    best_loss_last = float('inf')
    while True:
        print("[%d] Performing Iteration: %d " % (mypid, i))
        dataset = utils.resample(train_set, replace=True, n_samples=train_batch+bp_batch)
        data_pre = []
        target_pre = []
        for d in dataset:
            data_pre.append(torch.from_numpy(np.asarray([d[0]], dtype=np.float32)))
            target_pre.append(torch.from_numpy(np.asarray([d[1]], dtype=np.int64)))
        data = torch.cat(data_pre, dim=0)
        target = torch.cat(target_pre, dim=0)
        data = data.view(train_batch + bp_batch, 1, num_features)
        target = target.view(train_batch + bp_batch, 1)
        train_d = zip(data[0:train_batch, :], target[0:train_batch, :])
        backprop_d = zip(data[train_batch:train_batch + bp_batch, :], target[train_batch:train_batch + bp_batch, :])

        print("[%d] Training Tree..." % mypid)

        time_s = time.clock()
        for (batch_idx, (data, target)) in enumerate(train_d):
            X = Variable(data)
            Y = Variable(target, requires_grad=False)
            y_onehot.zero_()
            y_onehot.scatter_(1, target.view(1, 1), 1)
            Y_oh = Variable(y_onehot)
            model.train_tree(X, Y, Y_oh)
            if ((batch_idx + 1) % int(np.ceil(train_batch/10.0))) == 0:
                print("[{0}] Processed Ex: {1}/{2}".format(mypid, batch_idx + 1, train_batch))
        time_f = time.clock()
        training_time = time_f - time_s
        print("[%d] Tree Training Complete! (Time: %f sec, # Nodes: %d)\nStarting Backpropagation Step..." % (mypid, training_time, model.count))

        loss_sum = 0.0
        time_s = time.clock()
        for (batch_idx, (data, target)) in enumerate(backprop_d):
            optimizer.zero_grad()
            X = Variable(data)
            Y = Variable(target, requires_grad=False)
            loss = criterion(model.forward(X), Y)
            loss_sum += loss.data.float()[0]
            loss.backward()
            optimizer.step()
            if ((batch_idx + 1) % int(np.ceil(bp_batch / 10.0))) == 0:
                print("[{0}] Processed Ex: {1}/{2}".format(mypid, batch_idx + 1, bp_batch))
        time_f = time.clock()
        bp_time = time_f - time_s
        avg_loss = loss_sum / bp_batch
        print("[%d] Backpropagation Step Complete! (Time: %f sec, Average Loss: %f (delta: %f))\n" % (mypid, bp_time, avg_loss, np.abs(avg_loss_last - avg_loss)))

        train_data["K"] = model.k
        train_data["Iter"] = i
        train_data["TrainTime"] = training_time
        train_data["Nodes"] = model.count
        train_data["BackPropTime"] = bp_time
        train_data["Loss"] = avg_loss
        model.clear_tree()
        train_writer.writerow(train_data)
        if np.abs(avg_loss_last - avg_loss) <= 0.005 and avg_loss <= 0.001:
            print("[%d] Training Finished!\n" % mypid)
            break
        avg_loss_last = avg_loss
        best_loss = min((best_loss, avg_loss))
        if best_loss < best_loss_last:
            torch.save(obj=model.mod, f=(dir_name + "k_" + str(model.k) + "-seed_" + str(args.seed) + model_fmt))
        best_loss_last = best_loss
        i += 1

def test_epoch(model, train_set_master, test_set, **args):
    #train with a specific k, test across all k's
    sizes = args["test_sizes"]
    ks = args["test_ks"]
    for tk in ks:
        model.k = tk
        print("[{0}] Testing with k = {1}".format(mypid, model.k))
        for n_e in sizes:
            model.clear_tree()
            y_onehot = torch.FloatTensor(1, num_classes)

            print("[{0}] Training tree with {1} examples...".format(mypid, "all" if n_e >= len(train_set_master) else n_e))

            dataset = utils.resample(train_set_master, replace=False, n_samples=len(train_set_master) if n_e > len(train_set_master) else n_e)
            data_pre = []
            target_pre = []
            for d in dataset:
                data_pre.append(torch.from_numpy(np.asarray([d[0]], dtype=np.float32)))
                target_pre.append(torch.from_numpy(np.asarray([d[1]], dtype=np.int64)))
            data = torch.cat(data_pre, dim=0)
            target = torch.stack(target_pre, dim=0)
            train_set = zip(data, target)
            start_t = time.clock()
            for batch_idx, (data, target) in enumerate(train_set):

                X = data.view(1, num_features)

                Y = target
                y_onehot.zero_()
                y_onehot.scatter_(1, Y.view(1, 1), 1)
                X = Variable(X)
                Y = Variable(Y, requires_grad=False)
                Y_oh = Variable(y_onehot)
                model.train_tree(X, Y, Y_oh)
                if ((batch_idx+1) % int(np.ceil(n_e/10.0))) == 0:
                    print("[{0}] Processed Ex: {0}/{1}".format(mypid, batch_idx+1, len(train_set)))
            finish_t = time.clock()
            test_tt = finish_t - start_t
            print("All Done! (Tree Size: {0}, Time: {1})".format(model.count, test_tt))

            print("Performing Test!")
            num_right = 0.0
            num_total = 0.0
            start_t = time.clock()
            for batch_idx, (data, target) in enumerate(test_set):
                data = torch.from_numpy(np.asarray([data], dtype=np.float32))
                target = torch.from_numpy(np.asarray([target], dtype=np.int64))
                X = Variable(data.view(1, num_features))
                Y = Variable(target, requires_grad=False)
                closest_node = model.query_tree(X)
                y_pred = model.data[closest_node][1]
                if y_pred == Y:
                    num_right += 1.0
                num_total += 1
                if ((batch_idx+1) % 1000) == 0:
                    print("[{0}] Processed Ex: {1}/{2}".format(mypid, batch_idx+1, len(test_set)))
            finish_t = time.clock()
            true_testt = finish_t - start_t
            test_data["K"] = model.k
            test_data["TrainTime"] = test_tt
            test_data["NumTrain"] = n_e
            test_data["Nodes"] = model.count
            test_data["TestTime"] = true_testt
            test_data["Accuracy"] = ((num_right/num_total) * 100.0)
            test_writer.writerow(test_data)
            print("[%d] Testing Done! (Final Accuracy: %f, Time: %f)" % (mypid, ((num_right/num_total) * 100.0), true_testt))