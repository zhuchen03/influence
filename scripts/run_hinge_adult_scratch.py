from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  
   
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

import pdb
import pickle
import os

def load_adult_dataset():

    train_set = np.load('/scratch0/GoGradients/data/adult/train_transform_withlabel.npy')
    test_set = np.load('/scratch0/GoGradients/data/adult/test_transform_withlabel.npy')

    X_train, y_train = train_set[:,:-1], train_set[:,-1]
    X_test, y_test = test_set[:,:-1], test_set[:,-1] #.reshape(-1,1)

    train = DataSet(X_train, y_train)
    test = DataSet(X_test, y_test)

    return base.Datasets(train=train, validation=test, test=test)

data_sets = load_adult_dataset()
num_train = data_sets.train._x.shape[0]

svm_weight_dir, svm_bias_dir = '/scratch0/GoGradients/code/svm_figures/svm_weight.npy', '/scratch0/GoGradients/code/svm_figures/svm_bias.npy'
res_dir = './svm_influence'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

num_classes = 2
input_dim = 20
weight_decay = 0.01
use_bias = True
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

temps = [0.001, 0.1]
num_temps = len(temps)

num_params = input_dim + 1


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
    model_name='smooth_hinge_17_t-%s' % temp)

model.train()
# model.load_checkpoint(iter_to_load=0)
# model.set_params_from_npy(svm_weight_dir, svm_bias_dir)
model.saver.save(model.sess, model.checkpoint_file, global_step=0)

hinge_W = model.sess.run(model.params)[0]

model_margins = model.sess.run(model.margin, feed_dict=model.all_test_feed_dict)
# Look at np.argsort(model_margins)[:10] to pick a test example
train_idx = np.argsort(model_margins)[:10]
print("Margins: {}".format(train_idx))

np.random.seed(92)


params = np.zeros([num_temps, num_params])
margins = np.zeros([num_temps, num_train])
influences = np.zeros([num_temps, num_train])


# actual_loss_diffs = np.zeros([num_temps, num_to_remove])
# predicted_loss_diffs = np.zeros([num_temps, num_to_remove])
# indices_to_remove = np.zeros([num_temps, num_to_remove], dtype=np.int32)

test_idx_list = list(set(list(train_idx) + list(np.random.randint(0,num_train, 10))))

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
        model_name='smooth_hinge_17_t-%s' % temp)
  
    if temp == 0:
        model.load_checkpoint(iter_to_load=0)
    else:
        params_feed_dict = {}
        params_feed_dict[model.W_placeholder] = hinge_W
        model.sess.run(model.set_params_op, feed_dict=params_feed_dict)
        model.print_model_eval()
        model.saver.save(model.sess, model.checkpoint_file, global_step=0)

    for remove_idx in train_idx:
        actual_loss_diff_list = []
        influence = []
        for test_idx in test_idx_list:
            if temp == 0:
                actual_loss_diffs, predicted_loss_diff, indices_to_remove[counter, :] = experiments.test_retraining_on_train(
                        model,
                        test_idx,
                        iter_to_load=0,
                        force_refresh=False,
                        num_steps=2000,
                        remove_type='given',
                        num_to_remove=remove_idx)
            else:
                actual_loss_diff, predicted_loss_diffs_cg, indices_to_remove = experiments.rem_sv_inf_on_train_ind(
                    model,
                    test_idx,
                    iter_to_load=0,
                    force_refresh=False,
                    num_to_remove=remove_idx,  # data_sets.train.x.shape[0],
                    remove_type='random',
                    random_seed=0)
                print("*********************************", remove_idx, test_idx)
                print(actual_loss_diff)
                print(predicted_loss_diffs_cg)
                actual_loss_diff_list.append(actual_loss_diff)
                influence.append(predicted_loss_diffs_cg)
        total_inf = np.sum(np.abs(influence))
        inf_dict = {"total_inf":total_inf, "subset_idx":test_idx_list, "train_idx": remove_idx,
                    "infs": influence, "ground_truth": actual_loss_diff_list}
        pickle.dump(inf_dict, open(os.path.join(res_dir, 'influence_{}_temp{}.pkl'.format(remove_idx, temp)), 'w'),
                    pickle.HIGHEST_PROTOCOL)
        
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

np.savez(
    'output/hinge_results', 
    temps=temps,
    indices_to_remove=indices_to_remove,
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs=predicted_loss_diffs,
    influences=influences
)