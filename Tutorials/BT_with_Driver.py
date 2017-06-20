import numpy as np
import copy


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
    from mnist import MNIST

    data = MNIST('./MNIST_data/', return_type='numpy')
    x_train, y_train = data.load_training()
    num_examples = len(x_train)
    selection_list = range(num_examples)
    random.shuffle(selection_list)
    training_examples = [(x_train[i], y_train[i]) for i in selection_list]
    root_example = training_examples[random.randint(0, num_examples)]
    boundary_tree = BoundaryTree(k=-1, root_x = root_example[0], root_y = root_example[1])

    iter_count = 1
    max_iters = num_examples # You can set this to whatever you want
    print("Beginning Training...")
    for (ex_feats, true_class) in training_examples:
        boundary_tree.train(ex_feats, true_class)
        percent_complete = round((iter_count/float(num_examples)*100.0), 1)
        if percent_complete % 1 == 0 or iter_count == 1:
            print("%d percent complete" % int(percent_complete))
        iter_count += 1
        if iter_count >= max_iters:
            break
    print("Training Done!")

    x_test, y_test = data.load_testing()
    num_examples = len(x_test)
    selection_list = range(num_examples)
    random.shuffle(selection_list)
    test_examples = [(x_test[i], y_test[i]) for i in selection_list]
    num_correct = 0.0
    iter_num = 0.0
    print("Running Monte Carlo Accuracy Test...")
    for (ex_feats, true_class) in test_examples:
        class_guess = boundary_tree.query(ex_feats)
        if np.array_equal(class_guess, true_class):
            num_correct += 1.0
        iter_num += 1.0
        print("Accuracy ( @ Iteration %d): %.5f" % (iter_num, (num_correct/iter_num)*100.0))
    print("Testing Complete!\nFINAL ACCURACY: %.5f percent correct." % ((num_correct/iter_num)*100.0))
