import numpy as np

def viterbi(emit_w, trans_w, X, y, state_init=None):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt_list: list of np.ndarray - Optimal state sequence of length N
    """

    num_states, num_emissions, num_total_bins = emit_w.shape
    num_emissions_eff = num_emissions -1
    tiny = np.finfo(0.).tiny

    S_opt_list = []

    for trial in range(0, len(stim)):

        ################################
        # Compute emission probabilities
        ################################

        T = X[trial].shape[1]
        prob_emission = np.zeros((num_states, num_emissions, T))

        for state_idx in range(numOfstates):

            mat1 = np.zeros((num_emissions_eff, num_total_bins, T))
            for idx in range(T):
                mat1[..., idx] = emit_w[state_idx]

            mat2 = np.zeros((num_emissions_eff, num_total_bins, T))
            for idx in range(num_emissions_eff):
                mat2[idx] = np.reshape(X[trial], (1, num_total_bins, T), order = 'F')

            filtpower = np.reshape(
                                        np.sum(
                                                np.reshape(
                                                            mat1, (num_emissions_eff, num_total_bins, T), order = 'F'
                                                        ) * mat2,
                                                axis = 1
                                            ),
                                        (num_emissions_eff, T), order = 'F'
                                    )   # (num_emissions -1 x T)

            filtpower = np.vstack((np.zeros(1,T), filtpower))   # (num_emissions x T)

            prob_emission[state_idx] = np.exp(filtpower)/np.sum(np.exp(filtpower), axis=0)

        ################################
        # Compute transition probabilities
        ################################

        prob_transition = np.zeros((num_states, num_emissions, T-1))

        for state_idx in range(num_states):
            # Use data from 1:end-1 or 2:end?
            filtpower = np.sum(
                                np.tile(
                                            np.expand_dims(trans_w[state_idx], axis = 2), (1, 1, T-1)
                                        ) *\
                                np.tile(
                                            np.reshape(X[trial][:, 1:], (1, num_total_bins, T-1), order = 'F'),
                                            (num_states, 1, 1)
                                        ),
                                axis = 1
                            )

            filtpower[state_idx] = 0    # Want the self-transition effects to be zero
            prob_transition[state_idx] = np.exp(filtpower)/np.sum(np.exp(filtpower), axis=0)

        ############################
        # Viterbi Algorithm
        ############################
        if state_init is None:
            prob_init = np.ones(num_states)/num_states
        else:
            prob_init = state_init[trial]

        A_log = np.log(prob_transition + tiny)  # (num_states x num_states x T)
        C_log = np.log(prob_init + tiny)    # (num_states)
        B_log = np.log(prob_emission + tiny)    # (num_states x num_emissions x T)

        # Initialize D and E matrices
        D_log = np.zeros((num_states, T))
        E = np.zeros((num_states, T-1)).astype(np.int32)
        D_log[:, 0] = C_log + B_log[:, y[trial][0], 0]

        # Compute D and E in a nested loop
        for n in range(1, T):
            for i in range(num_states):
                temp_sum = A_log[:, i, n] + D_log[:, n-1]
                D_log[i, n] = np.max(temp_sum) + B_log[i, y[trial][n], n]
                E[i, n-1] = np.argmax(temp_sum)

        # Backtracking
        S_opt = np.zeros(T).astype(np.int32)
        S_opt[-1] = np.argmax(D_log[:, -1])
        for n in range(T-2, -1, -1):
            S_opt[n] = E[int(S_opt[n+1]), n]

        S_opt_list.append(S_opt)

    return S_opt_list
