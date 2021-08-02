import numpy as np
import scipy.stats
from scipy.io import loadmat, savemat

import scipy.ndimage.filters
import matplotlib.pyplot as plt
import sys
import os
import time

from pyGLMHMM import GLMHMM
from datetime import datetime

if __name__ == "__main__":
    import scipy.stats
    import scipy.ndimage.filters

    # PATH = '/Users/andrewsong/1_Research/data/Parenting/GLMHMM/GLMHMM.mat'
    DATAPATH ='../data/GLMHMM.mat'
    info = loadmat(DATAPATH)

    subj = 'F43'
    subj_test = ['F45', 'F42', 'FV4', 'MV1']
    numOfbins = 30 # (10 Hz x 3 seconds)
    prune_nan = True
    filter_offset = 1   # Bias. Always set it to 1
    num_states = 1
    num_emissions = 7
    num_feedbacks = 8
    max_optim_iter = 3
    max_iter = 2
    fs = 10
    L2_smooth = True
    smooth_lambda = 0.05
    random_state = 9000

    numOfanimals = info['animals'].shape[-1]
    animal_names = []
    output = {}
    features = {}
    animal_list = []
    animal_list_test = []
    stamps = {}

    ##############################
    # preprocessing feature inputs
    #
    # features: feature dictionary of (features x T)
    # output: output (behavior) dictionary of (features x T)

    behavior_names = [name[0] for name in info['behavior_cleaned'][0]]
    behavior_names = ['idle'] + behavior_names

    for idx in range(numOfanimals):

        T = info['features_animal'][0][idx].shape[0]

        animal_names.append(info['animals'][0, idx])

        if subj in info['animals'][0][idx][0]:
                animal_list.append(idx)

        for indiv in subj_test:
            if indiv in info['animals'][0][idx][0]:
                animal_list_test.append(idx)

        if prune_nan:
            indices = np.isnan(info['features_animal'][0][idx][:,-1])
            min_idx = np.min(np.where(indices==False)[0])
            max_idx = np.max(np.where(indices==False)[0])
        else:
            min_idx = 0
            max_idx = T

        output[idx] = info['ethogram_animal_string'][0][idx][min_idx:max_idx, :].flatten()
        stamps[idx] = [min_idx, max_idx]

        dict = []
        temp = info['features_animal'][0][idx][:,0]
        dict.append(temp)

        temp = info['features_animal'][0][idx][:,1]
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
        features[idx] = scipy.stats.zscore(features[idx], axis=1)


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

    print("ANIMAL train ", animal_list)
    print("ANIMAL test ", animal_list_test)

    regressors = []
    target = []
    regressors_test = []
    target_test = []

    for idx in animal_list:
        regressors.append(features[idx])
        target.append(output[idx])

    for idx in animal_list_test:
        regressors_test.append(features[idx])
        target_test.append(output[idx])

    fig, ax = plt.subplots(figsize=(20,6))
    for idx in range(len(target)):
        ax.plot(target[idx], label="{}".format(idx))
    ax.legend()
    plt.savefig('ethogram_{}.png'.format(subj))

    ###################
    # Run the estimator
    ###################

    estimator = GLMHMM.GLMHMMEstimator(
                                        random_state = random_state,
                                        num_samples = len(animal_list),
                                        num_states = num_states,
                                        num_emissions = num_emissions,
                                        num_feedbacks = num_feedbacks,
                                        num_filter_bins = numOfbins,
                                        num_steps = 1,
                                        max_iter = max_iter,
                                        max_optim_iter = max_optim_iter,
                                        filter_offset = filter_offset,
                                        L2_smooth = L2_smooth,
                                        smooth_lambda = smooth_lambda
                                    )

    s= time.time()

    output = estimator.fit(regressors, target, [])

    e = time.time()
    print("=============")
    print("Total elapsed time: ", e-s)
    print("=============")

    #################
    # Testing
    #################
    output_predict = estimator.predict(regressors_test, target_test)
    for idx in range(len(animal_list_test)):
        print("Likelihood for {}: ".format(idx), output_predict['forward_ll'][idx] - output_predict['chance_ll'][idx])

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    PATH = '../results/{}_state{}_lambda{}_iter{}'.format(
                                                    random_date,
                                                    num_states,
                                                    smooth_lambda,
                                                    max_iter
                                                )
    os.makedirs(PATH)

    titlesize= 17
    fontsize = 14

    print("Num of iterations: ", len(output))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(output[-1]['loglik'],linewidth=2)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_title("Negative log-likelihood {:.2e}".format(output[-1]['loglik'][-1]), fontsize=titlesize)
    plt.savefig(os.path.join(PATH, 'likelihood.png'), bbox_inches='tight')

    ####################
    # Train data
    ####################
    forward_ll_arr = output[-1]['prob_emission']

    for idx, animal_idx in enumerate(animal_list):

        fig, ax = plt.subplots(2, 1, figsize=(18, 10))
        forward_ll = forward_ll_arr[idx]

        min_idx = stamps[animal_idx][0]
        max_idx = stamps[animal_idx][1]

        for emit_idx in range(len(forward_ll)):
            ax[0].plot(min_idx/fs + np.arange(min_idx, max_idx)/fs, forward_ll[emit_idx], label='{} ({})'.format(behavior_names[emit_idx], emit_idx))
        ax[0].legend(fontsize=11, loc=4)
        ax[0].set_title("Forward prediction likelihood", fontsize=titlesize)
        ax[0].set_ylabel('p(Behavior)', fontsize=fontsize)

        ax[1].plot(min_idx/fs + np.arange(min_idx, max_idx)/fs, target[idx])
        ax[1].set_title("Actual observations", fontsize=titlesize)
        ax[1].set_xlabel("Time (s)", fontsize=fontsize)
        ax[1].set_ylabel("Behavior", fontsize=fontsize)

        ax[0].tick_params('both', labelsize=fontsize)
        ax[1].tick_params('both', labelsize=fontsize)

        plt.savefig(os.path.join(PATH, 'forward_likelihood_animal_{}.png'.format(animal_names[animal_idx][0])), bbox_inches='tight')

    #####################
    # Test data
    #####################
    forward_ll_arr = output_predict['prob_emission']

    for idx, animal_idx in enumerate(animal_list_test):

        fig, ax = plt.subplots(2, 1, figsize=(18, 10))
        forward_ll = forward_ll_arr[idx]

        min_idx = stamps[animal_idx][0]
        max_idx = stamps[animal_idx][1]

        for emit_idx in range(len(forward_ll)):
            ax[0].plot(min_idx/fs + np.arange(min_idx, max_idx)/fs, forward_ll[emit_idx], label='{} ({})'.format(behavior_names[emit_idx], emit_idx))
        ax[0].legend(fontsize=11, loc=4)
        ax[0].set_title("Forward prediction likelihood", fontsize=titlesize)
        ax[0].set_ylabel('p(Behavior)', fontsize=fontsize)

        ax[1].plot(min_idx/fs + np.arange(min_idx, max_idx)/fs, target_test[idx])
        ax[1].set_title("Actual observations", fontsize=titlesize)
        ax[1].set_xlabel("Time (s)", fontsize=fontsize)
        ax[1].set_ylabel("Behavior", fontsize=fontsize)

        ax[0].tick_params('both', labelsize=fontsize)
        ax[1].tick_params('both', labelsize=fontsize)

        plt.savefig(os.path.join(PATH, 'forward_likelihood_animal_{}.png'.format(animal_names[animal_idx][0])), bbox_inches='tight')

    emit_w_final = output[-1]['emit_w']
    emit_w_init = estimator.emit_w_init_
    trans_w_final = output[-1]['trans_w']
    trans_w_init = estimator.trans_w_init_

    ##################
    # Emission filters
    ##################

    for state_idx in range(num_states):
        fig, ax = plt.subplots(num_emissions-1, 1, figsize=(14,20))
        for emit_idx in range(num_emissions-1):
            ax[emit_idx].plot(emit_w_init[state_idx, emit_idx, :],"-g", label='Init')
            ax[emit_idx].plot(emit_w_final[state_idx, emit_idx, :-1], "-r", label='Learned')
            ax[emit_idx].tick_params('both', labelsize=fontsize)
            ax[emit_idx].set_title('Bias {:.2f}'.format(emit_w_final[state_idx, emit_idx, -1]))

            ax[emit_idx].set_ylabel("Emission {}".format(emit_idx+1), fontsize=fontsize)
            for idx in range(num_feedbacks+1):
                ax[emit_idx].axvline(numOfbins * idx, linestyle='--')

            if emit_idx == 0:
                ax[emit_idx].set_title("Emission filters for state {}".format(state_idx), fontsize=titlesize)

            if emit_idx == num_emissions -2:
                ax[emit_idx].set_xlabel("Features", fontsize=fontsize)
                ax[emit_idx].legend(fontsize=14)

        plt.savefig(os.path.join(PATH, 'state{}_emission_filters.png'.format(state_idx)), bbox_inches='tight')

    ####################
    # Transition filters
    ####################

    for state_idx1 in range(num_states):
        fig, ax = plt.subplots(num_states, 1, figsize=(14,20))

        for state_idx2 in range(num_states):
            ax[state_idx2].plot(trans_w_init[state_idx1, state_idx2, :],"-g", label='Init')
            ax[state_idx2].plot(trans_w_final[state_idx1, state_idx2, :-1], "-r", label='Learned')
            ax[state_idx2].tick_params('both', labelsize=fontsize)
            ax[state_idx2].set_title('Bias {:.2f}'.format(trans_w_final[state_idx1, state_idx2, -1]), fontsize=fontsize-1)

            ax[state_idx2].set_ylabel("State {}".format(state_idx2), fontsize=fontsize)
            for idx in range(num_feedbacks+1):
                ax[state_idx2].axvline(numOfbins * idx, linestyle='--')

            if state_idx2 == 0:
                ax[state_idx2].set_title("Transition filters for state {}".format(state_idx1), fontsize=titlesize)

            if state_idx2 == num_states-1:
                ax[state_idx2].set_xlabel("Features", fontsize=fontsize)
                ax[state_idx2].legend(fontsize=14)

        plt.savefig(os.path.join(PATH, 'state{}_transition_filters.png'.format(state_idx1)), bbox_inches='tight')

    ###########################
    # Save the relevant results
    result = {}
    result['behavior'] = target
    for idx in range(len(animal_list)):
        result['forward_ll_{}'.format(idx)] = forward_ll_arr[idx]
    result['animal_list'] = [animal_names[idx] for idx in animal_list]
    result['animal_list_test'] = [animal_names[idx] for idx in animal_list_test]

    result['emit_w_init'] = emit_w_init
    result['emit_w_final'] = emit_w_final
    result['trans_w_init'] = trans_w_init
    result['trans_w_final'] = trans_w_final

    savemat(os.path.join(PATH, './result.mat'), result)
