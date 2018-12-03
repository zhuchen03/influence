import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

# code for plotting total influences computed by SVMs

# train_idx = np.load('/scratch0/GoGradients/code/svm_figures/train_most_confusing_idxes_C1.npy')
# hinge_losses = 1 - np.load('/scratch0/GoGradients/code/svm_figures/train_margins_C1.npy')
# hinge_losses = (hinge_losses>0) * hinge_losses
# res_dir = './svm_influence'
# temp = 0.0005
# hinge_scale = 100
# figure_title = "Influence vs Hinge Loss\n(on Adult, influence computed with Soft SVM)"

train_idx = np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_most_confusing_idxes_C1.npy')
hinge_losses = 1 - np.load('/scratch0/GoGradients/code/svm_figures/mnist_1_7/train_margins_C1.npy')
hinge_losses = (hinge_losses>0) * hinge_losses
res_dir = './svm_influence/mnist_1_7'
temp = 0.0005
figure_title = "Influence vs Hinge Loss\n(on MNIST, influence computed with Soft SVM)"

influence_list, hinge_list = [], []
for idx in train_idx:
    try:
        datadict = pickle.load(open(os.path.join(res_dir, 'influence_{}_temp{}.pkl'.format(idx, temp))))
    except:
        print("File {} not found. skipping. ".format(os.path.join(res_dir, 'influence_{}_temp{}.pkl'.format(idx, temp))))
    total_inf = datadict['total_inf']
    hinge = hinge_losses[idx]
    influence_list.append(total_inf)
    hinge_list.append(hinge)

# load the random samples
file_list = os.listdir(res_dir)
temp_kw = "temp%s"%(temp)
for fname in file_list:
    if temp_kw in fname:
        idx = int(fname.split('_')[1])
        if idx not in train_idx:
            datadict = pickle.load(open(os.path.join(res_dir, fname)))
            total_inf = datadict['total_inf']
            hinge = hinge_losses[idx]
            influence_list.append(total_inf)
            hinge_list.append(hinge)

# plot it
plt.figure()
plt.scatter(hinge_list[len(train_idx):], influence_list[len(train_idx):], c=[0,0,1,0.3])
plt.scatter(hinge_list[:len(train_idx)], influence_list[:len(train_idx)], c=[1,0,0,0.3])
plt.legend(["Randomly Selected Points", "Largest Hinge Points"])
plt.xlabel("Hinge Loss")
plt.ylabel("Estimated Influence")
plt.title(figure_title)
plt.savefig(os.path.join(res_dir, "influence_vs_hinge.png"))
