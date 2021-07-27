"""
Andrew Song
"""

import numpy as np

def _forward_likelihood(emit_w, stim):

    num_emissions = stim[0]['num_emissions']
    num_states = emit_w.shape[0]
    num_bins = stim[0]['data'].shape[0]

    prob_emission_arr = []
    likelihood_arr = []

    for trial in range(0, len(stim)):

        data = stim[trial]['data']
        y = stim[trial]['emit']

        gamma = stim[trial]['gamma']
        xi = stim[trial]['xi']
        T = data.shape[1]

        prob_emission = np.zeros((num_emissions, T))
        loglikelihood = 0

        # Assume filtpower is states x emissions x time
        filtpower = np.zeros((num_states, num_emissions, T))
        for i in range(num_states):
            # Convert into states x bins x time and sum across bins
            temp = np.reshape(
                                np.sum(
                                        np.reshape(np.tile(np.expand_dims(emit_w[i], axis = 2), (1, 1, T)), (num_emissions-1, num_bins, T), order = 'F')\
                                        * np.tile(np.reshape(stim[trial]['data'], (1, num_bins, T), order = 'F'), (num_emissions-1, 1, 1)),
                                        axis = 1
                                    ),
                                (num_emissions-1, T)
                            )

            filtpower[i, 1:, :] = temp

        for idx in range(T-1):

            prediction = np.zeros(num_states)

            for i in range(num_states):
                temp = 0
                for j in range(num_states):
                    temp += xi[j, i, idx] * gamma[j,idx]
                prediction[i] = temp

            prediction /= np.sum(prediction)

            denom = 1 + np.sum(np.exp(filtpower[..., idx+1]), axis=1)

            prob_conditional = np.zeros((num_states, num_emissions))
            prob_conditional[:, 0] = 1

            for obs in range(1, num_emissions):
                prob_conditional[:, obs] = np.exp(filtpower[:, obs, idx+1])
            prob_conditional /= np.expand_dims(np.sum(prob_conditional, axis=1), axis=1)

            prob_emission[:, idx+1] = np.matmul(prob_conditional.T, prediction.reshape(num_states, 1)).flatten()

            loglikelihood += np.log(prob_emission[y[idx+1], idx+1])

        prob_emission_arr.append(prob_emission)
        likelihood_arr.append(loglikelihood/T)

    result = {
        'emission': prob_emission_arr,
        'likelihood': likelihood_arr
    }

    return result
