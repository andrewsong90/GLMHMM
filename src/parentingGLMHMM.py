import numpy as np
import scipy.stats
from scipy.io import loadmat

import scipy.ndimage.filters
import matplotlib.pyplot as plt
import sys
import os

from pyGLMHMM import GLMHMM

if __name__ == "__main__":
    import scipy.stats
    import scipy.ndimage.filters

    # PATH = '/Users/andrewsong/1_Research/data/Parenting/GLMHMM/GLMHMM.mat'
    PATH ='../data/GLMHMM.mat'
    info = loadmat(PATH)

    subj = 'F42'
    numOfbins = 10 # (10 Hz x 3 seconds)
    prune_nan = True
    filter_offset = 1
    num_states = 3

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
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,1]
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
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
    num_emissions = output[0].shape[0]
    num_feedbacks = features[0].shape[0]

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

    print("Num of iterations: ", len(output))
    print(output[-1]['loglik'])
