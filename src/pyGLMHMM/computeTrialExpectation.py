import copy
import numpy as np
from numba import jit, njit

eps = 1e-16

@njit
# @jit
def _compute_trial_expectation(prior, likelihood, transition):
    # Forward-backward algorithm, see Rabiner for implementation details
	# http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf

    t = 0
    T = likelihood.shape[1]
    num_states = likelihood.shape[0]

    # E-step
    # alpha1 is the forward probability of seeing the sequence
    alpha1 = np.zeros((len(prior), T))
    alpha2 = np.zeros((len(prior), T))
    scale = np.zeros((len(prior), T))
    scale_a = np.ones((T, 1))
    score = np.zeros((T, 1))

    alpha1[:, 0] = prior * likelihood[:, 0]
    alpha1[:, 0] = alpha1[:, 0] / np.sum(alpha1[:, 0], axis = 0)
    scale[:, 0] = alpha1[:, 0]

    alpha2[:, 0] = prior

    for t in range(1, T):
        alpha1[:, t] = np.dot(transition[:, :, t].T, alpha1[:, t - 1])
        scale[:, t] = alpha1[:, t] / np.sum(alpha1[:, t], axis = 0)
        alpha1[:, t] = alpha1[:, t] * likelihood[:, t]

        # Use this scaling component to try to prevent underflow errors
        scale_a[t] = np.sum(alpha1[:, t], axis = 0)
        alpha1[:, t] = alpha1[:, t] / scale_a[t]

        alpha2[:, t] = np.dot(transition[:, :, t].T, alpha2[:, t - 1])
        alpha2[:, t] = alpha2[:, t] / np.sum(alpha2[:, t], axis = 0)
        score[t] = np.sum(alpha2[:, t] * likelihood[:, t], axis = 0)

    # beta is the backward probability of seeing the sequence
    beta = np.zeros((len(prior), T))	# beta(i, t) = Pr(O(t + 1:T) | X(t) = i)
    beta[:, -1] = np.ones(len(prior)) / len(prior)

    scale_b = np.ones((T, 1))

    for t in range(T - 2, -1, -1):
        beta[:, t] = np.dot(transition[:, :, t + 1], (beta[:, t + 1] * likelihood[:, t + 1]))
        scale_b[t] = np.sum(beta[:, t], axis = 0)
        beta[:, t] = beta[:, t] / scale_b[t]

    # If any of the values are 0, it's defacto an underflow error so set it to eps
    for i in alpha1:
        i[i==0] = eps
    for i in beta:
        i[i==0] = eps

    # alpha1[alpha1 == 0] = eps
    # beta[beta == 0] = eps

    # gamma is the probability of seeing the sequence, found by combining alpha and beta

    # gamma = np.log(alpha1) + np.log(beta) - np.tile(np.log(np.cumsum(scale_a, axis = 0)).T, (num_states, 1)) - np.tile(np.log(np.flip(np.cumsum(np.flip(scale_b, axis = 0), axis = 0), axis = 0)).T, (num_states, 1))

    gamma = np.log(alpha1) + np.log(beta)\
            - np.log(np.cumsum(scale_a)).repeat(num_states).reshape(-1, num_states).T\
            - np.log(np.flip(np.cumsum(np.flip(scale_b)))).repeat(num_states).reshape(-1, num_states).T
    # print("+++++++++++++++++ ", gamma[:, :20])

    gamma = np.exp(gamma)
    # print("+++++++++++++++++ ", gamma[:, :20])
    # print("! ", scale_b.shape)
    for i in gamma:
        i[i==0] = eps
    # gamma[gamma == 0] = eps

    # gamma = gamma / np.tile(np.sum(gamma, axis = 0), (num_states, 1))
    gamma = gamma / np.sum(gamma, axis = 0).repeat(num_states).reshape(-1, num_states).T


    # xi is the probability of seeing each transition in the sequence
    xi = np.zeros((len(prior), len(prior), T - 1))
    # transition2 = copy.copy(transition[:, :, 1:])
    transition2 = np.copy(transition[:, :, 1:])

    for s1 in range(0, num_states):
        for s2 in range(0, num_states):
            xi[s1, s2, :] = np.log(likelihood[s2, 1:]) + np.log(alpha1[s1, 0:-1]) + np.log(transition2[s1, s2, :].T) + np.log(beta[s2, 1:])\
                            - np.log(np.cumsum(scale_a[0:-1]))\
                            - np.log(np.flip(np.cumsum(np.flip(scale_b[1:]))))
            xi[s1, s2, :] = np.exp(xi[s1, s2, :])

    for arr in xi:
        for i in arr:
            i[i==0] = eps
    # xi[xi == 0] = eps

    # Renormalize to make sure everything adds up properly
    # a = xi / np.tile(np.expand_dims(np.expand_dims(np.sum(np.sum(xi, axis = 0), axis = 0), axis = 0), axis = 0), (num_states, num_states, 1))
    # print(np.sum(xi, axis=(0,1)).repeat(len(prior)*len(prior)).reshape(-1, len(prior), len(prior)).shape)
    denom = np.sum(np.sum(xi, axis=0), axis=0)
    xi = xi / denom.repeat(len(prior)*len(prior)).reshape(-1, len(prior), len(prior)).T

    # print("A ", a[...,:3])
    # print("B ", b[...,:3])

    if xi.shape[2] == 1:
        xi = np.reshape(xi, (xi.shape[0], xi.shape[1], 1))

    # Save the prior initialization state for next time
    prior = gamma[:, 0]

    return prior, gamma, xi, alpha1, alpha2, scale, scale_a, score
