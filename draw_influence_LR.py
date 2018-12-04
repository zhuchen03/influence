import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

# code for plotting total influences computed by SVMs

# train_idx = np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_most_confusing_idxes_C1.npy')[:30]
# hinge_losses = 1 - np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_margins_C1.npy')
# hinge_losses = (hinge_losses>0) * hinge_losses
# res_dir = './LR_influence/mnist'
# hinge_scale = 100
# figure_title = "Influence vs {}\n(on MNIST, influence computed with Logistic Regression)"

train_idx = np.load('/scratch0/GoGradients/code/svm_figures/train_most_confusing_idxes_C1.npy')[:30]
minmax_p = np.load('/scratch0/GoGradients/code/exp/ploting/exp2/max.npy')
minmax_idx = minmax_p.argsort()[::-1][:30]
minmin_p = np.load('/scratch0/GoGradients/code/exp/ploting/exp2/min.npy')
# minmin_idx = minmin_p.argsort()[::-1][:30] # choosing max
minmin_idx = minmin_p.argsort()[:30] # choosing min
# minmax_res_dir = 'LR_influence/adult_minmax'
# minmin_res_dir = 'LR_influence/adult_minmin'
minmax_res_dir = 'LR_influence/adult_minmax_rho1'
minmin_res_dir = 'LR_influence/adult_minmin_rho1'

# hinge_losses = 1 - np.load('/scratch0/GoGradients/code/svm_figures/train_margins_C1.npy')
# hinge_losses = (hinge_losses>0) * hinge_losses
hinge_losses = -np.load('/scratch0/GoGradients/code/svm_figures/train_margins_C1.npy')
# res_dir = './LR_influence/adult'
random_dir = 'LR_influence/adult'
res_dir = 'LR_influence/adult_maxhingepts_2'
hinge_scale = 100
figure_title = "{} vs {}\n(on Adult, influence computed with Logistic Regression)"

out_dir = 'LR_influence/adult_rho1'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

influence_list, hinge_list, gt_list = [], [], []

def load_data(datadict, influence_list, hinge_list, gt_list):
    total_inf = datadict['total_inf']
    hinge = hinge_losses[idx]
    influence_list.append(total_inf)
    hinge_list.append(hinge)

    gt = np.sum(np.abs(datadict['ground_truth']))
    gt_list.append(gt)

for idx in train_idx:
    datadict = pickle.load(open(os.path.join(res_dir, 'influence_{}.pkl'.format(idx))))
    load_data(datadict, influence_list, hinge_list, gt_list)

# load the random samples
file_list = os.listdir(random_dir)
random_idxes = []
for fname in file_list:
    if 'pkl' not in fname:
        continue
    try:
        idx = int(fname.split('_')[1].split('.')[0])
    except:
        continue
    if idx not in train_idx:
        random_idxes.append(idx)
        datadict = pickle.load(open(os.path.join(random_dir, fname)))
        load_data(datadict, influence_list, hinge_list, gt_list)

# get the values from minmax and minmin
minmax_hinge_list, minmax_gt_list, minmax_influence_list = [], [], []
for idx in minmax_idx:
    datadict = pickle.load(open(os.path.join(minmax_res_dir, 'influence_{}.pkl'.format(idx))))
    load_data(datadict,minmax_influence_list,  minmax_hinge_list, minmax_gt_list)

minmin_hinge_list, minmin_gt_list, minmin_influence_list = [], [], []
for idx in minmin_idx:
    datadict = pickle.load(open(os.path.join(minmin_res_dir, 'influence_{}.pkl'.format(idx))))
    load_data(datadict, minmin_influence_list, minmin_hinge_list, minmin_gt_list)

def plotit(random_list1, random_list2, maxhinge_list1, maxhinge_list2,
           minmax_list1, minmax_list2, minmin_list1, minmin_list2,
           xlabel, ylabel, res_dir, xlim=None, xscale='linear', yscale='linear'):
    # plot it
    plt.figure()
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.scatter(random_list1, random_list2, c=[1,0,0,0.3])
    plt.scatter(maxhinge_list1, maxhinge_list2, c=[1,0,1,0.3])
    plt.scatter(minmax_list1, minmax_list2, c=[0,1,0,0.3])
    plt.scatter(minmin_list1, minmin_list2, c=[0,0,1,0.3])
    plt.legend(["Randomly Selected Points", "Largest Hinge Points", "Largest p_{minmax}", "Smallest p_{minmin}"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(figure_title.format(xlabel, ylabel))
    plt.autoscale()
    plt.savefig(os.path.join(res_dir, "{}_vs_{}.png".format(ylabel, xlabel)))

plotit(hinge_list[len(train_idx):], influence_list[len(train_idx):],
       hinge_list[:len(train_idx)], influence_list[:len(train_idx)],
       minmax_hinge_list, minmax_influence_list,
       minmin_hinge_list, minmin_influence_list,
       xlabel='-Margin', ylabel='InfluenceFunction',
       res_dir=out_dir)

# plt.figure()
# plt.scatter(gt_list[len(train_idx):], influence_list[len(train_idx):], c=[1,0,0,0.3])
# plt.scatter(gt_list[:len(train_idx)], influence_list[:len(train_idx)], c=[1,0,1,0.3])
# plt.scatter(minmax_gt_list, minmax_influence_list, c=[0,1,0,0.3])
# plt.scatter(minmin_gt_list, minmin_influence_list, c=[0,0,1,0.3])
# plt.legend(["Randomly Selected Points", "Largest Hinge Points", "Largest p_{minmax}", "Largest p_{minmin}"])
# plt.xlabel("Ground Truth")
# plt.ylabel("Estimated Influence")
# plt.title(figure_title.format("Influence", "Ground Truth"))
# plt.savefig(os.path.join(res_dir, "influence_vs_gt.png"))

plotit(gt_list[len(train_idx):], influence_list[len(train_idx):],
       gt_list[:len(train_idx)], influence_list[:len(train_idx)],
       minmax_gt_list, minmax_influence_list,
       minmin_gt_list, minmin_influence_list,
       xlabel='GroundTruth', ylabel='InfluenceFunction',
       res_dir=out_dir)

plotit(hinge_list[len(train_idx):], gt_list[len(train_idx):],
       hinge_list[:len(train_idx)], gt_list[:len(train_idx)],
       minmax_hinge_list, minmax_gt_list,
       minmin_hinge_list, minmin_gt_list,
       xlabel='-Margin', ylabel='GroundTruth',
       res_dir=out_dir)

plotit(minmax_p[random_idxes], gt_list[len(train_idx):],
       minmax_p[train_idx], gt_list[:len(train_idx)],
       minmax_p[minmax_idx], minmax_gt_list,
       minmax_p[minmin_idx], minmin_gt_list,
       xlabel='p_{minmax}', ylabel='GroundTruth',
       res_dir=out_dir, xscale='log')

plotit(minmin_p[random_idxes], gt_list[len(train_idx):],
       minmin_p[train_idx], gt_list[:len(train_idx)],
       minmin_p[minmax_idx], minmax_gt_list,
       minmin_p[minmin_idx], minmin_gt_list,
       xlabel='p_{minmin}', ylabel='GroundTruth',
       res_dir=out_dir, xscale='log')

plotit(hinge_list[len(train_idx):], minmax_p[random_idxes],
       hinge_list[:len(train_idx)], minmax_p[train_idx],
       minmax_hinge_list, minmax_p[minmax_idx],
       minmin_hinge_list, minmax_p[minmin_idx],
       xlabel='-Margin', ylabel='p_{minmax}',
       res_dir=out_dir, yscale='log')