"""
Andrew Song
"""

import numpy as np

def _chance_likelihood(stim):
    """
    Chance likelihood (Frequency of each observation)
    """
    num_emissions = stim[0]['num_states']
    prob_obs = np.zeros(num_emissions)
    numOfobs = 0
    likelihood_arr = []

    for trial in range(0, len(stim)):

        y = stim[trial]['emit']
        T = len(y)
        numOfobs += T

        for idx in range(T):
            prob_obs[y[idx]] += 1

    prob_obs /= numOfobs

    for trial in range(0, len(stim)):

        y = stim[trial]['emit']
        T = len(y)
        likelihood = 0

        for idx in range(T):
            likelihood += np.log(prob_obs[y[idx]])

        likelihood_arr.append(likelihood)

    result = {
        'likelihood': likelihood_arr
    }


    return result

def _chance_likelihood_seq(stim):
    """
    Chance likelihood sequential
    """
    return likelihood_arr
