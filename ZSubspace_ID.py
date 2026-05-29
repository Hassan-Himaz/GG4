import numpy as np
from ZLDSParams import LDSParams



class Subspace_ID:

    @staticmethod
    def n4sid(inputs: np.ndarray, outputs: np.ndarray, latent_dim: int,
              horizon: int) -> LDSParams:
        '''
        Deterministic-stochastic subspace ID (state-sequence form, VODM Alg. 1).
        Non-iterative, no init, no local optima. Returns A,B,C,D,Q,R,S and the
        singular-value spectrum (the cliff in `sv` gives the model order).
        Convention: x_{t+1}=Ax_t+Bu_t+w, y_t=Cx_t(+Du_t)+v.  D should be ~0.
        Recovered up to the same similarity transform as EM -> compare invariants only.

        inputs     : (T, input_dim)   drive signal (the SAME one that generated `outputs`)
        outputs    : (T, output_dim)  measured emissions
        latent_dim : state order n
        horizon    : block-Hankel horizon i (> latent_dim); default max(n+2, 2n)
        '''
        T, input_dim = inputs.shape
        output_dim = outputs.shape[1]
        n = latent_dim
        i = horizon if horizon is not None else max(n + 2, 2 * n)   # must exceed n
        num_windows = T - 2 * i + 1                                 # effective samples

        def hankel(data, num_blocks, start):     # block-Hankel: num_blocks rows, num_windows cols
            dim = data.shape[1]
            H = np.empty((num_blocks * dim, num_windows))
            for row in range(num_blocks):
                H[row*dim:(row+1)*dim] = data[start+row : start+row+num_windows].T
            return H

        def oblique(target, along, onto):        # row(target) onto row(onto) along row(along)
            perp = lambda X: X - (X @ along.T) @ np.linalg.pinv(along @ along.T) @ along
            return perp(target) @ np.linalg.pinv(perp(onto)) @ onto

        # --- i-projection: state sequence at time i ---
        past = np.vstack([hankel(inputs,  i, 0),
                          hankel(outputs, i, 0)])
        proj_i = oblique(hankel(outputs, i, i),       # future outputs
                         hankel(inputs,  i, i),       # along future inputs
                         past)                        # onto the past

        # --- shifted projection: state sequence at time i+1 ---
        past_shifted = np.vstack([hankel(inputs,  i+1, 0),
                                  hankel(outputs, i+1, 0)])
        proj_i1 = oblique(hankel(outputs, i-1, i+1),
                          hankel(inputs,  i-1, i+1),
                          past_shifted)

        # --- factor proj_i = observability @ states, via SVD truncated to n -  --
        left_sv, sing_vals, _ = np.linalg.svd(proj_i, full_matrices=False)
        observability   = left_sv[:, :n] * np.sqrt(sing_vals[:n])     # (output_dim*i, n)
        observability_m = observability[:-output_dim]                 # drop last block row

        states_i  = np.linalg.pinv(observability)   @ proj_i          # (n, num_windows)
        states_i1 = np.linalg.pinv(observability_m) @ proj_i1         # (n, num_windows)

        inputs_i  = hankel(inputs,  1, i)            # input  block row at time i
        outputs_i = hankel(outputs, 1, i)            # output block row at time i

        # --- one regression:  [states_i1; outputs_i] = [[A B],[C D]] [states_i; inputs_i] ---
        regressors = np.vstack([states_i, inputs_i])
        targets    = np.vstack([states_i1, outputs_i])
        theta = targets @ np.linalg.pinv(regressors)
        A, B = theta[:n, :n], theta[:n, n:]
        C, D = theta[n:, :n], theta[n:, n:]

        residual = targets - theta @ regressors
        cov = (residual @ residual.T) / num_windows

        return LDSParams(A=A,B =B,C= C,Q = cov[:n,:n],R = cov[n:,n:],mu_0 = np.zeros(n),P_0=np.eye(n))
    
    @staticmethod
    def stochastic_n4sid(outputs: np.ndarray,
                        latent_dim: int,
                        horizon: int | None = None) -> LDSParams:
        '''
        Stochastic subspace ID — identifies (A, C, Q, R) from outputs alone.

        No B, no D. The input contribution is unobserved and gets absorbed into
        Q (process noise). Use as an initialisation seed for blind/augmented EM,
        not as a final estimate.

        Parameters
        ----------
        outputs    : (T, output_dim)  measured emissions
        latent_dim : state order n
        horizon    : block-Hankel horizon i (> latent_dim); default max(n+2, 2n)
        '''
        T = outputs.shape[0]
        output_dim = outputs.shape[1]
        n = latent_dim
        i = horizon if horizon is not None else max(n + 2, 2 * n)
        num_windows = T - 2 * i + 1
        if num_windows <= 0:
            raise ValueError(f"Trial too short for horizon {i}: need T > {2 * i}, got T = {T}")

        def hankel(data, num_blocks, start):
            dim = data.shape[1]
            H = np.empty((num_blocks * dim, num_windows))
            for row in range(num_blocks):
                H[row*dim:(row+1)*dim] = data[start+row : start+row+num_windows].T
            return H

        def orthogonal(target, onto):
            # Orthogonal projection of row(target) onto row(onto).
            return target @ np.linalg.pinv(onto) @ onto

        # --- projection at time i: future outputs onto past outputs ---
        Y_p     = hankel(outputs, i, 0)         # past   (i * output_dim, num_windows)
        Y_f     = hankel(outputs, i, i)         # future (i * output_dim, num_windows)
        proj_i  = orthogonal(Y_f, Y_p)

        # --- shifted projection at time i+1 ---
        Y_p_sh  = hankel(outputs, i+1, 0)       # past shifted by one
        Y_f_sh  = hankel(outputs, i-1, i+1)     # future shifted by one (one block shorter)
        proj_i1 = orthogonal(Y_f_sh, Y_p_sh)

        # --- factor proj_i = observability @ states, truncated SVD to rank n ---
        left_sv, sing_vals, _ = np.linalg.svd(proj_i, full_matrices=False)
        observability   = left_sv[:, :n] * np.sqrt(sing_vals[:n])     # (i * output_dim, n)
        observability_m = observability[:-output_dim]                 # drop last block row

        states_i  = np.linalg.pinv(observability)   @ proj_i          # (n, num_windows)
        states_i1 = np.linalg.pinv(observability_m) @ proj_i1         # (n, num_windows)

        outputs_i = hankel(outputs, 1, i)        # output block row at time i

        # --- regression: [states_i1; outputs_i] = [[A], [C]] @ states_i ---
        # No B, D — there are no observed inputs to regress against.
        regressors = states_i                                          # (n, num_windows)
        targets    = np.vstack([states_i1, outputs_i])                 # (n + output_dim, num_windows)
        theta = targets @ np.linalg.pinv(regressors)                   # (n + output_dim, n)
        A = theta[:n, :]                                               # (n, n)
        C = theta[n:, :]                                               # (output_dim, n)

        residual = targets - theta @ regressors
        cov = (residual @ residual.T) / num_windows
        Q = cov[:n, :n]
        R = cov[n:, n:]

        # Placeholder B, D for LDSParams API compatibility — augmented EM will learn the real ones.
        # If LDSParams.B/D shapes depend on input_dim, pass them through from the caller instead.
        return LDSParams(
            A=A, B=np.zeros((n, 0)), C=C,
            Q=Q, R=R,
            mu_0=np.zeros(n), P_0=np.eye(n),
        )