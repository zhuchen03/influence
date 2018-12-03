from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import matplotlib
matplotlib.use('agg')
   
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
from matplotlib import pyplot as plt

import pdb
import pickle
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_dist(weights, pts):
    # each row of pts represents a data
    x = np.abs(np.sum(pts * weights.reshape(-1)[:-1].reshape(1, -1), 1)+weights[-1])
    return x

def get_wrong_flags(model, data, label):
    logits = model.sess.run(model.logits, feed_dict=model.all_train_feed_dict)
    pred = np.sign(logits.reshape(-1))
    return pred != label

def plot_infuence(gt, pred, train_sample_idx):
    plt.figure()
    plt.plot(gt, pred, 'b.')
    plt.title("Influence of sample {} on subset".format(train_sample_idx))
    plt.savefig("figures/mnist_trainsample{}_influence.png".format(train_sample_idx))

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
                Y[i] = -1
            else:
                Y[i] = 1
    return X, Y.reshape(-1)

def get_mnist_dset(class_inds=[1,7]):

    X_train, y_train = load_mnist('train', class_inds=class_inds)
    X_test, y_test = load_mnist('test', class_inds=class_inds)

    train = DataSet(X_train, y_train)
    test = DataSet(X_test, y_test)

    return base.Datasets(train=train, validation=test, test=test)


class_inds = [1,7]
data_sets = get_mnist_dset(class_inds)
num_train = data_sets.train._x.shape[0]

svm_weight_dir, svm_bias_dir = '/scratch0/GoGradients/code/svm_figures/mnist_{}_{}/svm_weight.npy'.format(class_inds[0], class_inds[1]), \
                               '/scratch0/GoGradients/code/svm_figures/mnist_{}_{}/svm_bias.npy'.format(class_inds[0], class_inds[1])
res_dir = './svm_influence/mnist_{}_{}'.format(class_inds[0], class_inds[1])
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

num_classes = 2
input_dim = data_sets.train._x.shape[1]
# need to make it equivalent to setting C = 1, since it takes mean when computing the loss
weight_decay = 0.5 / data_sets.train._x.shape[0]
use_bias = True
batch_size = 100
initial_learning_rate = 1e-3
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

temps = [5e-4]#, 0.00001, 0.001, 0.1]
num_temps = len(temps)

num_params = input_dim + 1

np.random.seed(92)

# Get weights from hinge

tf.reset_default_graph()

temp = 0
model = SmoothHinge(
    use_bias=use_bias,
    temp=temp,
    input_dim=input_dim,
    weight_decay=weight_decay,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='smooth_hinge_mnist_17_t-%s' % temp)

# model.train()
# model.load_checkpoint(iter_to_load=0)
model.set_params_from_npy(svm_weight_dir, svm_bias_dir)
model.saver.save(model.sess, model.checkpoint_file, global_step=0)

hinge_W = model.sess.run(model.params)[0]

model_margins = model.sess.run(model.margin, feed_dict=model.all_test_feed_dict)
# Look at np.argsort(model_margins)[:10] to pick a test example
# train_idx = np.argsort(model_margins)[:10]
# print("Margins: {}, {}".format(train_idx, model_margins[train_idx]))
# pdb.set_trace()
train_idx = np.load('/scratch0/GoGradients/code/svm_figures/mnist_{}_{}/train_most_confusing_idxes_C1.npy'.format(class_inds[0], class_inds[1]))
train_idx = np.concatenate([train_idx, np.random.randint(0, len(data_sets.train.labels), size=(100,))])


params = np.zeros([num_temps, num_params])
margins = np.zeros([num_temps, num_train])
influences = np.zeros([num_temps, num_train])


# actual_loss_diffs = np.zeros([num_temps, num_to_remove])
# predicted_loss_diffs = np.zeros([num_temps, num_to_remove])
# indices_to_remove = np.zeros([num_temps, num_to_remove], dtype=np.int32)
# random_idxes = list(np.random.randint(0,num_train, 100))
bound_dist = get_dist(hinge_W, data_sets.train._x)
sorted_idxes = bound_dist.argsort()
wrong_flags = get_wrong_flags(model, data_sets.train._x, data_sets.train._labels)

chosen_subset = sorted_idxes[wrong_flags[sorted_idxes]]
test_idx_list = chosen_subset
print("Dists: {}, flags: {}".format(bound_dist[test_idx_list], wrong_flags[chosen_subset]))

for counter, temp in enumerate(temps):

    tf.reset_default_graph()

    model = SmoothHinge(
        use_bias=use_bias,
        temp=temp,
        input_dim=input_dim,
        weight_decay=weight_decay,
        num_classes=num_classes,
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output',
        log_dir='log',
        model_name='smooth_hinge_mnist_t-%s' % temp)
  
    if temp == 0:
        model.load_checkpoint(iter_to_load=0)
    else:
        params_feed_dict = {}
        params_feed_dict[model.W_placeholder] = hinge_W
        model.sess.run(model.set_params_op, feed_dict=params_feed_dict)
        model.print_model_eval()
        model.saver.save(model.sess, model.checkpoint_file, global_step=0)
    total_inf_list = []
    for remove_idx in train_idx:
        actual_loss_diff_list = []
        influence = []
        for test_idx in test_idx_list:
            if temp == 0:
                actual_loss_diff, predicted_loss_diff, indices_to_remove = experiments.test_retraining_on_train(
                        model,
                        test_idx,
                        iter_to_load=0,
                        force_refresh=False,
                        num_steps=2000,
                        remove_type='given',
                        num_to_remove=remove_idx)
            else:
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
        inf_dict = {"total_inf":total_inf, "subset_idx":test_idx_list, "train_idx": remove_idx,
                    "infs": influence, "ground_truth": actual_loss_diff_list}
        pickle.dump(inf_dict, open(os.path.join(res_dir, 'influence_{}_temp{}.pkl'.format(remove_idx, temp)), 'w'),
                    pickle.HIGHEST_PROTOCOL)
        plot_infuence(actual_loss_diff_list, influence, remove_idx)

    plt.figure()
    plt.plot([i for i in range(len(train_idx))], total_inf_list)
    plt.savefig(os.path.join(res_dir, "temp{}_totalinfluence.png".format(temp)))
    pickle.dump({"remove_idxes":train_idx, "total influences":total_inf_list})
    # cur_params, cur_margins = model.sess.run([model.params, model.margin], feed_dict=model.all_train_feed_dict)
    # cur_influences = model.get_influence_on_train_loss(
    #     test_indices=[test_idx],
    #     train_idx=np.arange(num_train),
    #     force_refresh=False)
    #
    # params[counter, :] = np.concatenate(cur_params)
    # margins[counter, :] = cur_margins
    # influences[counter, :] = cur_influences
    #
    # if temp == 0:
    #     actual_loss_diffs[counter, :], predicted_loss_diffs[counter, :], indices_to_remove[counter, :] = experiments.test_retraining(
    #         model,
    #         test_idx,
    #         iter_to_load=0,
    #         force_refresh=False,
    #         num_steps=2000,
    #         remove_type='maxinf',
    #         num_to_remove=num_to_remove)
#
# np.savez(
#     'output/hinge_results',
#     temps=temps,
#     indices_to_remove=indices_to_remove,
#     actual_loss_diffs=actual_loss_diffs,
#     predicted_loss_diffs=predicted_loss_diffs,
#     influences=influences
# )