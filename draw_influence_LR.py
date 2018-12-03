import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

# code for plotting total influences computed by SVMs

# train_idx = np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_most_confusing_idxes_C1.npy')[:30]
# hinge_losses = 1 - np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_margins_C1.npy')
# hinge_losses = (hinge_losses>0) * hinge_losses
# res_dir = './LR_influence/mnist'
# hinge_scale = 100
# figure_title = "Influence vs {}\n(on MNIST, influence computed with Logistic Regression)"

train_idx = np.load('/scratch0/GoGradients/code/svm_figures/train_most_confusing_idxes_C1.npy')[:30]
hinge_losses = 1 - np.load('/scratch0/GoGradients/code/svm_figures/train_margins_C1.npy')
hinge_losses = (hinge_losses>0) * hinge_losses
res_dir = './LR_influence/adult'
hinge_scale = 100
figure_title = "Influence vs {}\n(on Adult, influence computed with Logistic Regression)"

influence_list, hinge_list, gt_list = [], [], []

def load_data(datadict):
    total_inf = datadict['total_inf']
    hinge = hinge_losses[idx]
    influence_list.append(total_inf)
    hinge_list.append(hinge)

    gt = np.sum(np.abs(datadict['ground_truth']))
    gt_list.append(gt)

for idx in train_idx:
    datadict = pickle.load(open(os.path.join(res_dir, 'influence_{}.pkl'.format(idx))))
    load_data(datadict)

# load the random samples
file_list = os.listdir(res_dir)
for fname in file_list:
    if 'pkl' not in fname:
        continue
    try:
        idx = int(fname.split('_')[1].split('.')[0])
    except:
        continue
    if idx not in train_idx:
        datadict = pickle.load(open(os.path.join(res_dir, fname)))
        load_data(datadict)

# plot it
plt.figure()
plt.scatter(hinge_list[len(train_idx):], influence_list[len(train_idx):], c=[0,0,1,0.3])
plt.scatter(hinge_list[:len(train_idx)], influence_list[:len(train_idx)], c=[1,0,0,0.3])
plt.legend(["Randomly Selected Points", "Largest Hinge Points"])
plt.xlabel("Hinge Loss")
plt.ylabel("Estimated Influence")
plt.title(figure_title.format("Hinge Loss"))
plt.savefig(os.path.join(res_dir, "influence_vs_hinge.png"))

plt.figure()
plt.scatter(gt_list[len(train_idx):], influence_list[len(train_idx):], c=[0,0,1,0.3])
plt.scatter(gt_list[:len(train_idx)], influence_list[:len(train_idx)], c=[1,0,0,0.3])
plt.legend(["Randomly Selected Points", "Largest Hinge Points"])
plt.xlabel("Ground Truth")
plt.ylabel("Estimated Influence")
plt.title(figure_title.format("Ground Truth"))
plt.savefig(os.path.join(res_dir, "influence_vs_gt.png"))
