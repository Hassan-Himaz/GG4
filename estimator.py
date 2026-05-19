import numpy as np
from PEM_framework import PEM_Framework
from Illustrator import Illustrator
from Simulator import Simulator
import scipy.linalg as la


#class to hold all different state space estimation techniques


class Estimator:
    def __init__(self, data_set,ilustrator:Illustrator):
        self.data_set = data_set
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = data_set.shape
        self.observation = data_set
        self.ilustrator = ilustrator




    def _slice_and_flatten(self, lag: int) -> tuple[np.ndarray, np.ndarray]:
        """Private helper to split and flatten time series into t and t+lag segments."""
        if not 1 <= lag < self.timestep_cnt:
            raise ValueError(f"Lag must be between 1 and {self.timestep_cnt - 1}.")
            
        x_t = self.observation[:, :-lag, :]
        x_t_plus_lag = self.observation[:, lag:, :]
        
        return x_t.reshape(-1, self.neuron_cnt), x_t_plus_lag.reshape(-1, self.neuron_cnt)

    #-------------------------------------------------------------------

    #hankel matrix method


    def estimate_ssid_matrices(self, state_dim: int, horizon: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the hidden system matrix A and observation matrix C 
        using Subspace System Identification (SSID).
        
        Parameters
        ----------
        state_dim : int
            The desired dimension of the hidden state subspace (n).
            Must be less than or equal to (neuron_cnt * horizon).
        horizon : int
            The number of time steps to look into the past/future to build 
            the Hankel prediction matrix. Higher captures slower dynamics.
            
        Returns
        -------
        A : np.ndarray (state_dim x state_dim)
            The hidden state transition matrix.
        C : np.ndarray (neuron_cnt x state_dim)
            The observation mapping matrix.
        """

        # Ensure we are working with mean-centered data across trials
        # Data shape: (Trials * Timepoints, Neurons)
        total_timepoints = self.trial_cnt * self.timestep_cnt
        y_flat = self.observation.reshape(total_timepoints, self.neuron_cnt)
        y_centered = y_flat - np.mean(y_flat, axis=0)
        
        # 1. Construct the Block Hankel Matrix
        # We need enough continuous time points per trial. For simplicity, we assume
        # data continuity or perform it per trial. Here we treat the flattened array 
        # as a continuous stream for structural simplicity.
        N = total_timepoints - 2 * horizon + 1
        if N <= 0:
            raise ValueError("Data is too short for the requested horizon size.")
            
        # Build past and future Hankel structures
        H = np.zeros((2 * horizon * self.neuron_cnt, N))
        for i in range(2 * horizon):
            H[i * self.neuron_cnt : (i + 1) * self.neuron_cnt, :] = y_centered[i : i + N, :].T
            
        Y_p = H[: horizon * self.neuron_cnt, :]  # Past observations
        Y_f = H[horizon * self.neuron_cnt :, :]  # Future observations

        # 2. Project Future onto Past (Geometric Projection)
        # This captures the subspace shared between past and future
        # Projection = Y_f * Y_p^T * inv(Y_p * Y_p^T) * Y_p
        R_ff = np.dot(Y_f, Y_p.T)
        R_pp = np.dot(Y_p, Y_p.T) + np.eye(Y_p.shape[0]) * 1e-6 # Ridge stability
        projection = np.dot(R_ff, la.solve(R_pp, Y_p))

        # 3. Singular Value Decomposition (SVD) to isolate the State Subspace
        U, Sigma, Vt = la.svd(projection, full_matrices=False)
        
        # Truncate to the chosen hidden state dimension
        U_n = U[:, :state_dim]
        Sigma_n = np.diag(Sigma[:state_dim])
        Vt_n = Vt[:state_dim, :]

        # 4. Extract the Hidden State Sequences
        # The extended observability matrix O_i and estimated states X
        O_i = np.dot(U_n, la.sqrtm(Sigma_n))
        X = np.dot(la.sqrtm(Sigma_n), Vt_n)
        
        # Shifted state sequences for regression: X(t) and X(t+1)
        X_t = X[:, :-1]
        X_t_plus_1 = X[:, 1:]

        # 5. Compute System Matrices (A and C) via Ordinary Least Squares
        # X(t+1) = A * X(t) -> A = X(t+1) * X(t)^T * inv(X(t) * X(t)^T)
        A = np.dot(X_t_plus_1, X_t.T) @ la.inv(np.dot(X_t, X_t.T) + np.eye(state_dim) * 1e-6)
        
        # C is extracted from the top block of the Observability Matrix O_i
        C = O_i[: self.neuron_cnt, :]

        return A, C
    


    def estimate_system_matrices(self, latent_dim: int = 2, n_iter: int = 10, control_function=None) -> dict:
        """
        Attempts to estimate the linear dynamical system matrices (A, C, Q, R, and B) 
        from the observation data using Expectation-Maximization and State Regression.

        Parameters
        ----------
        latent_dim : int
            The assumed number of hidden state dimensions.
        n_iter : int
            The number of EM iterations to perform.
        control_function : callable, optional
            A function `f(time, data_history)` that returns the control input u_t.
            Defaults to None (assumes zero input/no B matrix).

        Returns
        -------
        dict
            A dictionary containing the estimated matrices.
        """
        try:
            from pykalman import KalmanFilter
            from sklearn.linear_model import Ridge
        except ImportError:
            raise ImportError("Requires pykalman and scikit-learn. Run: pip install pykalman scikit-learn")

        print(f"Estimating system matrices (Latent dimensions: {latent_dim})...")
        
        # 1. Run standard EM to find the latent geometry (C, Q, R) and smoothed states
        flat_data = self.observation.reshape(-1, self.neuron_cnt)
        kf = KalmanFilter(n_dim_state=latent_dim, n_dim_obs=self.neuron_cnt)
        kf = kf.em(flat_data, n_iter=n_iter)
        
        # Extract the smoothed states for every trial
        smoothed_states = np.zeros((self.trial_cnt, self.timestep_cnt, latent_dim))
        for i in range(self.trial_cnt):
            smoothed_states[i], _ = kf.smooth(self.observation[i])

        A_est = kf.transition_matrices
        B_est = None

        # 2. If a control function is provided, use regression to find A and B simultaneously
        if control_function is not None:
            X_curr_list, X_next_list, U_list = [], [], []
            
            # Build the state and input histories
            for trial in range(self.trial_cnt):
                for t in range(self.timestep_cnt - 1):
                    # Evaluate the control function at this timestep
                    u_t = np.atleast_1d(control_function(t, self.observation[trial, :t+1]))
                    
                    X_curr_list.append(smoothed_states[trial, t])
                    X_next_list.append(smoothed_states[trial, t+1])
                    U_list.append(u_t)

            X_curr = np.array(X_curr_list)
            X_next = np.array(X_next_list)
            U = np.array(U_list)

            # Concatenate X_t and U_t into a single feature matrix
            # Equation: X_{t+1} = [A, B] * [X_t ; U_t]
            features = np.hstack((X_curr, U))
            
            # Fit a regularized regression to find the combined [A, B] matrix
            model = Ridge(alpha=1.0, fit_intercept=False)
            model.fit(features, X_next)
            
            # Split the learned coefficients back into A and B
            A_est = model.coef_[:, :latent_dim]
            B_est = model.coef_[:, latent_dim:]
            print("Successfully regressed B matrix using provided control function!")

        matrices = {
            "A": A_est,
            "C": kf.observation_matrices,
            "B": B_est,
            "Q": kf.transition_covariance,
            "R": kf.observation_covariance
        }
        
        print("Estimation complete.")
        print("-" * 20)
        print("A Matrix (Dynamics):\n", np.round(matrices["A"], 4))
        print("\nC Matrix (Observation):\n", np.round(matrices["C"], 4))
        if B_est is not None:
            print("\nB Matrix (Control):\n", np.round(matrices["B"], 4))
        else:
            print("\nB Matrix (Control): None (No control function provided).")
            
        print("\nQ Matrix (Process Noise Diagonal):\n", np.round(np.diag(matrices["Q"]), 4))
        print("\nR Matrix (Observation Noise Diagonal):\n", np.round(np.diag(matrices["R"]), 4))
        
        return matrices
    

       ###------------------------------------------------------------------------------------------------

    #Vector autoregression estimation method
    

    def estimate_var_matrix(self, lag: int = 1) -> np.ndarray:
        """
        Estimates the Vector Autoregressive (VAR) transition matrix W for a given time lag.
        
        Solves for W in: X(t + lag) = W * X(t) + noise
        Formula used: W = C_lag * inv(C_0)
        
        Parameters
        ----------
        lag : int
            The number of steps ahead the transition rules map.
            
        Returns
        -------
        np.ndarray
            The estimated (Neurons x Neurons) structural transition weight matrix.
        """
        # 1. Fetch shifted time slices
        x_t_flat, x_lag_flat = self._slice_and_flatten(lag)
        
        # 2. Mean-center signals to strip global population offset/biases
        x_t_centered = x_t_flat - np.mean(x_t_flat, axis=0)
        x_lag_centered = x_lag_flat - np.mean(x_lag_flat, axis=0)
        num_samples = x_t_flat.shape[0]
        
        # 3. Calculate internal contemporaneous covariance (C_0) and lag covariance (C_lag)
        c_0 = np.dot(x_t_centered.T, x_t_centered) / num_samples
        c_lag = np.dot(x_t_centered.T, x_lag_centered) / num_samples
        
        # Add a tiny ridge regression penalty to the diagonal to ensure stability during inversion
        c_0 += np.eye(self.neuron_cnt) * 1e-6
        
        # 4. Clear spatial confounders out: W^T = inv(C_0) * C_lag -> W = C_lag^T * inv(C_0)^T
        # Transpose adjustment matches row/column orientation standard to dynamical modeling
        transition_matrix = np.linalg.solve(c_0, c_lag).T
        
        return transition_matrix
    