import numpy as np
import scipy.stats
from scipy.io import loadmat

import scipy.ndimage.filters
import matplotlib.pyplot as plt
import sys
import os

from pyGLMHMM import GLMHMM
from datetime import datetime

if __name__ == "__main__":
    import scipy.stats
    import scipy.ndimage.filters

    # PATH = '/Users/andrewsong/1_Research/data/Parenting/GLMHMM/GLMHMM.mat'
    DATAPATH ='../data/GLMHMM.mat'
    info = loadmat(DATAPATH)

    subj = 'F42'
    maxiter = 10
    numOfbins = 30 # (10 Hz x 3 seconds)
    prune_nan = True
    filter_offset = 1
    num_states = 1
    num_emissions = 7
    num_feedbacks = 8

    numOfanimals = info['animals'].shape[-1]
    output = {}
    features = {}
    animal_list = []

    ##############################
    # preprocessing feature inputs
    #
    # features: feature dictionary of (features x T)
    # output: output (behavior) dictionary of (features x T)

    for idx in range(numOfanimals):

        T = info['features_animal'][0][idx].shape[0]

        if subj == 'all':
            animal_list.append(idx)
        else:
            if subj in info['animals'][0][idx][0]:
                animal_list.append(idx)

        if prune_nan:
            indices = np.isnan(info['features_animal'][0][idx][:,-1])
            min_idx = np.min(np.where(indices==False)[0])
            max_idx = np.max(np.where(indices==False)[0])
        else:
            min_idx = 0
            max_idx = T

        output[idx] = info['ethogram_animal_string'][0][idx][min_idx:max_idx, :].flatten()

        dict = []
        temp = info['features_animal'][0][idx][:,0]
        # temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,1]
        # temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,3]
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,4]
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,5]
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,6]
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,7]
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,8]
        dict.append(temp)

        features[idx] = np.stack(dict, axis=0)
        features[idx] = features[idx][:, min_idx:max_idx]
        features[idx] = scipy.stats.zscore(features[idx], axis=1, nan_policy='raise')


    ##############
    # Regressor (We want to build the Toeplitz matrix)
    for idx in range(numOfanimals):
        feature = features[idx]
        numOffeatures, T = feature.shape
        feature_conv_mat = np.zeros((numOffeatures * numOfbins + filter_offset, T + numOfbins - 1))

        for feature_idx in range(numOffeatures):
            for bin_idx in range(numOfbins):
                feature_conv_mat[feature_idx * numOfbins + bin_idx, bin_idx:T+bin_idx] = feature[feature_idx]

        feature_conv_mat[-filter_offset:, :] = 1

        features[idx] = feature_conv_mat[:, :T]


    print("ANIMAL ", animal_list)
    regressors = []
    target = []

    for idx in animal_list:
        regressors.append(features[idx])
        target.append(output[idx])

    ###################
    # Run the estimator

    estimator = GLMHMM.GLMHMMEstimator(
                                    num_samples = len(animal_list),
                                    num_states = num_states,
                                    num_emissions = num_emissions,
                                    num_feedbacks = num_feedbacks,
                                    num_filter_bins = numOfbins,
                                    num_steps = 1,
                                    filter_offset = filter_offset
                                )

    output = estimator.fit(regressors, target, [])

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    PATH = '../results/{}'.format(random_date)
    os.makedirs(PATH)

    print("Num of iterations: ", len(output))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(output[-1]['loglik'],linewidth=2)
    ax.set_xlabel('Epoch')
    plt.savefig(os.path.join(PATH, 'likelihood.png'))
