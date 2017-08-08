import numpy as np
import copy
import sys
import time

class BoundaryTree(object):
    @staticmethod
    def distance(x1, x2):
        return np.sqrt(np.sum(np.square(np.subtract(x2, x1))))

    def __init__(self, k, root_x, root_y):
        self.max_children = k
        self.root_node_id = 0
        self.count = 1 # We've already initialized root as 0, so start with 1.

        # Make sure all of the data is the same type, so we'll force it - here
        # and anywhere else we get data from the "outside".
        root_data = (np.asarray(root_x), np.asarray(root_y))

        self.data = {self.root_node_id: root_data}
        self.children = {self.root_node_id: []}

    def query(self, test_x, internal=False):
        current_node_id = self.root_node_id
        while True:
            children = self.children[current_node_id]
            if self.max_children == -1 or len(children) < self.max_children:
                children = copy.copy(children)
                children.append(current_node_id)

            closest_node_id = min(children, key=lambda child_node_id: BoundaryTree.distance(self.data[child_node_id][0], test_x))
            if closest_node_id == current_node_id:
                break
            current_node_id = closest_node_id
        if internal:
            return current_node_id # return the node id
        else:
            return self.data[current_node_id][1] # return the class label itself, for convenience.

    def train(self, new_x, new_y):
        closest_node_id = self.query(new_x, internal=True)
        if not np.array_equal(self.data[closest_node_id][1], new_y):
            # create a new node in the tree
            new_node_id = self.count
            self.count = self.count + 1
            self.data[new_node_id] = (np.asarray(new_x), np.asarray(new_y))
            self.children[new_node_id] = []
            # link it, as a child of the closest
            self.children[closest_node_id].append(new_node_id)

if __name__ == "__main__":
    import random
    import cifar10
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy.misc import toimage
    import csv

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] /= 4
    fig_size[1] /= 4
    plt.rcParams["figure.figsize"] = fig_size

    random.seed(4242)
    data = cifar10.load_training_data()
    images, classes, onehot = data
    images_test, classes_test, onehot_test = cifar10.load_test_data()
    nums_to_run = [1, 10, 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000]
    branching_factor_list = [10, 20, 50, 100]
    #---------------------------------------------
    for bfi in range(len(branching_factor_list)):
        branching_factor = branching_factor_list[bfi]
        with open('CIFAR10_data_' + str(branching_factor) + '.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(('Branching Factor', 'Num Examples', 'Accuracy', 'Training Time (s)', 'Test Time (s)'))
            for i in range(len(nums_to_run)):
                num_to_use = nums_to_run[i]
                for j in range(10):
                    num_examples = len(images)
                    selection_list = range(num_examples)
                    random.shuffle(selection_list)
                    training_examples = [(images[index], classes[index]) for index in selection_list]
                    root_example = training_examples[random.randint(0, num_examples - 1)]
                    #boundary_tree = BoundaryTree(k=-1, root_x = root_example[0], root_y = root_example[1])
                    boundary_tree = BoundaryTree(k=branching_factor, root_x = root_example[0], root_y = root_example[1])
                    iter_count = 1
                    max_iters = min(num_examples, num_to_use) # You can set this to whatever you want

                    print("Beginning Training...")
                    start_time_train = time.time()

                    previous_percent = -1
                    for (ex_feats, true_class) in training_examples:
                        #plt.imshow(toimage(ex_feats))
                        #plt.show()

                        boundary_tree.train(ex_feats, true_class)
                        percent_complete = round((iter_count/float(num_examples)*100.0), 1)
                        if int(percent_complete) != previous_percent:
                            if percent_complete % 1 == 0 or iter_count == 1:
                                sys.stdout.write("%d percent complete " % int(percent_complete))
                                for i in range(int(percent_complete/2)):
                                    sys.stdout.write("|")
                                sys.stdout.write("      \r")
                                sys.stdout.flush()
                                previous_percent = int(percent_complete)
                        iter_count += 1
                        if iter_count >= max_iters:
                            break
                    print("")
                    print("Training Done!")
                    end_time_train = time.time() - start_time_train
                    print("--- %s seconds ---" % (end_time_train))

                    #images_test, classes_test, onehot_test = cifar10.load_test_data()
                    num_examples_test = len(images_test)
                    selection_list_test = range(num_examples_test)
                    random.shuffle(selection_list_test)
                    test_examples = [(images_test[index], classes_test[index]) for index in selection_list_test]
                    num_correct = 0.0
                    iter_num = 0.0
                    print("Running Monte Carlo Accuracy Test...")
                    start_time_test = time.time()

                    for (ex_feats, true_class) in test_examples:
                        class_guess = boundary_tree.query(ex_feats)
                        if np.array_equal(class_guess, true_class):
                            num_correct += 1.0
                        iter_num += 1.0
                        sys.stdout.write("Accuracy (@ Iteration %d): %.5f      \r" % (iter_num, (num_correct/iter_num)*100.0))
                        sys.stdout.flush()
                    end_time_test = time.time() - start_time_test
                    print("")
                    print("Testing Complete!\nFINAL ACCURACY: %.5f percent correct." % ((num_correct/iter_num)*100.0))
                    writer.writerow((branching_factor, max_iters, num_correct/iter_num, end_time_train, end_time_test))
                    with open('results-' + str(num_to_use) + '.txt', 'a') as f:
                        f.write("FINAL ACCURACY: %.5f percent correct with %d examples. Training: %.5f seconds. Testing: %.5f seconds.\n" % ((num_correct/iter_num)*100.0, max_iters, end_time_train, end_time_test))
