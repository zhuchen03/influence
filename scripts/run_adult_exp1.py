from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model

import scipy
import sklearn

import influence.experiments as experiments1
from influence.nlprocessor import NLProcessor
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_spam import load_spam

import tensorflow as tf

from influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

def load_adult_dataset():

    train_set = np.load('/scratch0/GoGradients/data/adult/train.npy')  #_transform_withlabel
    test_set = np.load('/scratch0/GoGradients/data/adult/test.npy')

    X_train, y_train = train_set[:,:-1], (train_set[:,-1]+1)/2
    X_test, y_test = test_set[:,:-1], (test_set[:,-1]+1)/2 #.reshape(-1,1)

    train = DataSet(X_train, y_train)
    test = DataSet(X_test, y_test)

    return base.Datasets(train=train, validation=test, test=test)

np.random.seed(42)


# data_sets = load_spam()
data_sets = load_adult_dataset()

num_classes = 2

input_dim = data_sets.train.x.shape[1]
print('======+++++++++',input_dim)
weight_decay = 0.0001
# weight_decay = 1000 / len(lr_data_sets.train.labels)
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='spam_logreg')

tf_model.train()


test_idx = 9
actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove = experiments1.test_retraining_1(
    tf_model,
    test_idx,
    iter_to_load=0,
    force_refresh=False,
    num_to_remove=10, #data_sets.train.x.shape[0],
    remove_type='maxinf',
    random_seed=0)


# np.random.seed(0)
# ## Retraining the model by removing just one training point and computing the loss for all the test points
# for test_idx in range(10):
#     print('======', test_idx)
#     actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove = experiments.test_retraining_2(
#     tf_model,
#     test_idx,
#     iter_to_load=0,
#     force_refresh=False,
#     num_to_remove=0,           #training data point index which is removed
#     remove_type='random',
#     random_seed=0)



np.savez(
    'output/adult_all_train_2a.npz', 
    actual_loss_diffs=actual_loss_diffs, 
    predicted_loss_diffs_cg=predicted_loss_diffs_cg,
#     predicted_loss_diffs_lissa=predicted_loss_diffs_lissa,
    indices_to_remove=indices_to_remove
    )
