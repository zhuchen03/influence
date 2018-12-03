from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
 
from scipy.stats import pearsonr
from load_mnist import load_mnist

import influence.experiments as experiments
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.smooth_hinge import SmoothHinge
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS

from tensorflow.contrib.learn.python.learn.datasets import base
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import sys


import pdb
import pickle


def load_mnist(data_type='train', binary=True, class_inds = [1,7]):
    '''
    load_mnist('train') for training data
    load_mnist('test') for testing data
    '''
    MNIST_path = '/scratch0/GoGradients/data/MNIST'
    if binary:
        assert len(class_inds) == 2
    X = np.load(os.path.join(MNIST_path, "{}Images.npy".format(data_type)))
    Y = np.load(os.path.join(MNIST_path, "{}Labels.npy".format(data_type)))
    if len(Y.shape) > 1:
        Y = np.argmax(Y, axis=1)
    N, C, W, H = X.shape
    X = X.reshape(N, C*W*H)/255.
    mask = [Y[i] in class_inds for i in range(N)]
    X = X[mask, ...]
    Y = Y[mask]
    if binary:
        for i in range(len(Y)):
            if Y[i] == class_inds[0]:
                Y[i] = 0
            else:
                Y[i] = 1
    return X, Y.reshape(-1)

def get_mnist_dset(class_inds=[1,7]):

    X_train, y_train = load_mnist('train', class_inds=class_inds)
    X_test, y_test = load_mnist('test', class_inds=class_inds)

    train = DataSet(X_train, y_train)
    test = DataSet(X_test, y_test)

    return base.Datasets(train=train, validation=test, test=test)

def get_wrong_flags(model, data, label):
    logits = model.sess.run(model.logits, feed_dict=model.all_train_feed_dict)
    pred = np.sign(logits.reshape(-1))
    return pred != label

def plot_infuence(gt, pred, train_sample_idx, res_dir):
    plt.figure()
    plt.plot(gt, pred, 'b.')
    plt.title("Influence of sample {} on subset".format(train_sample_idx))
    plt.savefig(os.path.join(res_dir, "trainsample{}_influence.png".format(train_sample_idx)))

class_inds = [1,7]
data_sets = get_mnist_dset(class_inds)
num_train = data_sets.train._x.shape[0]

res_dir = './LR_influence/mnist'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

num_classes = 2
input_dim = 784
# weight_decay = 0.5 / data_sets.train._x.shape[0]
weight_decay = 1e-4
use_bias = True
batch_size = 100
initial_learning_rate = 1e-3
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

num_params = input_dim + 1

np.random.seed(92)

# Get weights from hinge

tf.reset_default_graph()

model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=1000,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist_logreg_chen')

model.train()
# model.load_checkpoint(iter_to_load=0)
model.saver.save(model.sess, model.checkpoint_file, global_step=0)

train_idx = np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_most_confusing_idxes_C1.npy')[:30]
train_idx = np.concatenate([train_idx, np.random.randint(0, len(data_sets.train.labels), size=(30,))])

test_idx_list = np.random.randint(0, data_sets.train._x.shape[0], 100)

total_inf_list = []
for remove_idx in train_idx:
    actual_loss_diff_list = []
    influence = []
    for test_idx in test_idx_list:
        actual_loss_diff, predicted_loss_diff, indices_to_remove = experiments.rem_sv_inf_on_train_ind(
            model,
            test_idx,
            iter_to_load=0,
            force_refresh=False,
            num_to_remove=remove_idx,  # data_sets.train.x.shape[0],
            remove_type='random',
            random_seed=0)
        print("*********************************", remove_idx, test_idx)
        print(actual_loss_diff)
        print(predicted_loss_diff)
        actual_loss_diff_list.append(actual_loss_diff)
        influence.append(predicted_loss_diff)
    total_inf = np.sum(np.abs(influence))
    total_inf_list.append(total_inf)
    inf_dict = {"total_inf": total_inf, "subset_idx": test_idx_list, "train_idx": remove_idx,
                "infs": influence, "ground_truth": actual_loss_diff_list}
    pickle.dump(inf_dict, open(os.path.join(res_dir, 'influence_{}.pkl'.format(remove_idx)), 'w'),
                pickle.HIGHEST_PROTOCOL)
    plot_infuence(actual_loss_diff_list, influence, remove_idx, res_dir)

plt.figure()
plt.plot([i for i in range(len(train_idx))], total_inf_list)
plt.savefig(os.path.join(res_dir, "mnist_totalinfluence.png"))
pickle.dump({"remove_idxes": train_idx, "total influences": total_inf_list},
            open(os.path.join(res_dir, 'influence_totals.pkl'), 'w'),
                pickle.HIGHEST_PROTOCOL)

