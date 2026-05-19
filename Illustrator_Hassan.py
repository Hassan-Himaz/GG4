import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
import scipy.linalg as la

class Illustrator:
    """
    A class to visualize neural activity data.
    Please finish the implementation and annotations of this class.
    """
    __slot__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt']

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator with the given observation data.
        Accepts a 3D numpy array of shape (Trials, Timepoints, Neurons) and stores it for visualization.
        Parameters:
            observation (np.ndarray): A 3D array containing neural activity data with dimensions (Trials, Timepoints, Neurons).
        """
        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape

    def _slice_and_flatten(self, lag: int) -> Tuple[np.ndarray, np.ndarray]:
        """Private helper to split and flatten time series into t and t+lag segments."""
        if not 1 <= lag < self.timestep_cnt:
            raise ValueError(f"Lag must be between 1 and {self.timestep_cnt - 1}.")
            
        x_t = self.observation[:, :-lag, :]
        x_t_plus_lag = self.observation[:, lag:, :]
        
        return x_t.reshape(-1, self.neuron_cnt), x_t_plus_lag.reshape(-1, self.neuron_cnt)

    def _compute_raw_expectation(self, x_t: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """Private helper to compute the raw uncentered expected value matrix E[X_t^T * X_target]."""
        return np.dot(x_t.T, x_target) / x_t.shape[0]

    def _compute_normalized_correlation(self, x_t_flat: np.ndarray, x_lag_flat: np.ndarray) -> np.ndarray:
        """Private helper to compute Pearson correlation coefficient matrices."""
        x_t_centered = x_t_flat - np.mean(x_t_flat, axis=0)
        x_lag_centered = x_lag_flat - np.mean(x_lag_flat, axis=0)
        
        covariance_matrix = np.dot(x_t_centered.T, x_lag_centered) / x_t_flat.shape[0]
        
        std_t = np.std(x_t_flat, axis=0)[:, np.newaxis]
        std_lag = np.std(x_lag_flat, axis=0)[np.newaxis, :]
        
        std_matrix = np.dot(std_t, std_lag)
        std_matrix[std_matrix == 0] = 1.0  # Protect against flatlined signals
        
        return covariance_matrix / std_matrix

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

    def plot_autocorrelation(self, lag: int = 1, normalize: bool = True):
        """
        Computes and plots the temporal lag-k relationship matrix across all trials.
        """
        x_t_flat, x_lag_flat = self._slice_and_flatten(lag)
        
        if normalize:
            lag_matrix = self._compute_normalized_correlation(x_t_flat, x_lag_flat)
            cbar_label = 'Correlation Coefficient [-1, 1]'
            title_str = f'Normalized Temporal Lag-{lag} Correlation Matrix\n(Pearson $r$)'
            vmin, vmax = -1.0, 1.0
        else:
            lag_matrix = self._compute_raw_expectation(x_t_flat, x_lag_flat)
            vmax = np.max(np.abs(lag_matrix))
            vmax = vmax if vmax > 0 else 1.0
            vmin = -vmax
            cbar_label = f'Expected Value $E[X_i(t) \cdot X_j(t+{lag})]$'
            title_str = f'Raw Temporal Lag-{lag} Correlation Matrix'

        # Plotting Setup
        plt.figure(figsize=(8, 6))
        heatmap = plt.imshow(lag_matrix, cmap='bwr', vmin=vmin, vmax=vmax)
        plt.colorbar(heatmap, label=cbar_label)
        plt.title(title_str)
        plt.xlabel(f'Neuron Index ($t+{lag}$)')
        plt.ylabel('Neuron Index ($t$)')
        
        plt.figtext(0.5, -0.02, 
                    "Note: The diagonal represents true temporal autocorrelation for each neuron.", 
                    ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
        plt.tight_layout()
        plt.show()

    def plot_neuron(self, neuron_id: int = 0, function: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        """
        Plot one selected neuron across all trials.
        """
        if not 0 <= neuron_id < self.neuron_cnt:
            raise ValueError(f"neuron_id must be between 0 and {self.neuron_cnt - 1}.")

        time = np.arange(self.timestep_cnt)

        plt.figure(figsize=(10, 5))
        for trial_id in range(self.trial_cnt):
            values = self.observation[trial_id, :, neuron_id]
            transformed_values = function(values)

            plt.plot(
                time,
                transformed_values,
                label=f"Trial {trial_id}",
                alpha=0.6 if self.trial_cnt > 10 else 0.8,
            )

        plt.xlabel("Time step")
        plt.ylabel("Observed activity")
        plt.title(f"Neuron {neuron_id} Profile Across All Trials")
        if self.trial_cnt <= 15:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    def plot_neuron(self, neuron_id: int = 0, function = lambda x: x):
        """
        Plot one selected neuron across all trials.

        Parameters
        ----------
        neuron_id : int
            Index of the neuron/signal to visualize.
        """
        if not 0 <= neuron_id < self.neuron_cnt:
            raise ValueError(f"neuron_id must be between 0 and {self.neuron_cnt - 1}.")

        time = np.arange(self.timestep_cnt)

        plt.figure(figsize=(10, 5))
        for trial_id in range(self.trial_cnt):
            values = self.observation[trial_id, :, neuron_id]
            transformed_values = function(values)

            plt.plot(
                time,
                transformed_values,
                label=f"trial {trial_id}",
                alpha=0.8,
            )

        plt.xlabel("Time step")
        plt.ylabel("Observed activity")
        plt.title(f"Neuron {neuron_id} across all trials")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
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
    
    def plot_matrix(self, matrix: np.ndarray, title: str = "Matrix Plot", 
                    color_coded: bool = True, show_numbers: bool = False, cmap: str = 'bwr'):
        """
        Displays a 2D matrix layout cleanly with synchronized grid markers.
        
        Parameters
        ----------
        matrix : np.ndarray
            The 2D matrix array to visualize.
        title : str
            The title header appended to the plot window.
        color_coded : bool
            If True, colors the pixels using a color map. If False, prints a grayscale layout.
        show_numbers : bool
            If True, overlays the actual numeric values inside each matrix cell.
        cmap : str
            Matplotlib colormap profile string (e.g., 'bwr', 'coolwarm', 'viridis').
        """
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        if color_coded:
            # Anchor maximum color ranges symmetrically around 0 for diverging maps
            vmax = np.max(np.abs(matrix))
            vmax = vmax if vmax > 0 else 1.0
            vmin = -vmax if cmap in ['bwr', 'seismic', 'coolwarm'] else np.min(matrix)
            
            heatmap = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            plt.colorbar(heatmap, label='Coefficient Intensity Value')
        else:
            heatmap = ax.imshow(matrix, cmap='gray', aspect='auto')
            plt.colorbar(heatmap, label='Value Scale')

        # Map dynamic tick marks for rows and columns
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        
        # Overlay the actual numbers if toggled on
        if show_numbers:
            # Get the current colormap to evaluate background brightness
            current_cmap = plt.get_cmap(cmap if color_coded else 'gray')
            norm = heatmap.norm
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    
                    # Determine cell background color to adjust text contrast dynamically
                    cell_color = current_cmap(norm(val))
                    # Compute relative luminance (standard formula for text legibility)
                    luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                    text_color = "black" if luminance > 0.5 else "white"
                    
                    # Place text cleanly centered in the cell (formatted to 2 decimal places)
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                            color=text_color, fontweight='bold', fontsize=9)

        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xlabel("Columns (Destination / Output Index)")
        ax.set_ylabel("Rows (Source / Input Index)")
        
        plt.grid(False) # Prevent gridlines from crossing inside pixels
        plt.tight_layout()
        plt.show()
    def plot_scree(self, horizon: int = 4):
        """
        Computes and plots a Scree Plot of the singular values from the 
        SSID projection matrix. This visualizes the 'energy' of each dimension
        to help determine the optimal hidden state_dim (the "elbow" method).
        
        Parameters
        ----------
        horizon : int
            The number of time steps used for the past/future data blocks.
            Should match the horizon you intend to use in estimate_ssid_matrices.
        """
        # 1. Flatten and center data
        total_timepoints = self.trial_cnt * self.timestep_cnt
        y_flat = self.observation.reshape(total_timepoints, self.neuron_cnt)
        y_centered = y_flat - np.mean(y_flat, axis=0)
        
        N = total_timepoints - 2 * horizon + 1
        if N <= 0:
            raise ValueError("Observation timeline is too short for this horizon.")
            
        # 2. Reconstruct the block Hankel rows
        H = np.zeros((2 * horizon * self.neuron_cnt, N))
        for i in range(2 * horizon):
            H[i * self.neuron_cnt : (i + 1) * self.neuron_cnt, :] = y_centered[i : i + N, :].T
            
        Y_p = H[: horizon * self.neuron_cnt, :]
        Y_f = H[horizon * self.grid_cols if hasattr(self, 'grid_cols') else horizon * self.neuron_cnt :, :]

        # 3. Geometric Projection
        R_ff = np.dot(Y_f, Y_p.T)
        R_pp = np.dot(Y_p, Y_p.T) + np.eye(Y_p.shape[0]) * 1e-6
        projection = np.dot(R_ff, la.solve(R_pp, Y_p))

        # 4. Extract Singular Values (No truncation here, we want to see all of them)
        _, Sigma, _ = la.svd(projection, full_matrices=False)
        
        # Calculate variance metrics for plotting
        variance_explained = (Sigma**2) / np.sum(Sigma**2) * 100
        cumulative_variance = np.cumsum(variance_explained)
        num_components = len(Sigma)
        x_ticks = np.arange(1, num_components + 1)

        # 5. Plotting a dual-axis Scree/Manifold chart
        fig, ax1 = plt.subplots(figsize=(9, 5))

        # Left Axis: Individual Singular Values
        color = 'tab:blue'
        ax1.set_xlabel('Component / Subspace Dimension Index', fontweight='bold')
        ax1.set_ylabel('Singular Value Magnitude', color=color, fontweight='bold')
        line1 = ax1.plot(x_ticks, Sigma, 'o-', color=color, linewidth=2, label='Singular Value')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x_ticks)
        ax1.grid(True, alpha=0.3)

        # Right Axis: Cumulative Explained Variance
        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('Cumulative Variance Explained (%)', color=color, fontweight='bold')
        line2 = ax2.plot(x_ticks, cumulative_variance, 's--', color=color, alpha=0.7, label='Cumulative Variance')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 105)

        # Dynamic annotations to guide selection
        plt.title('SSID Subspace Scree Plot\n(Look for the "Elbow" where Singular Values flatten out)', fontsize=12, pad=15)
        
        # Add a unified legend for both axes lines
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        plt.show()

    # =====================================================================
    # INPUT GENERATION UTILITIES (Static Methods)
    # =====================================================================
    
    @staticmethod
    def generate_impulse(length: int, trigger_time: int, input_dim: int = 1) -> np.ndarray:
        """Generates a unit impulse (a single spike) at a specific time."""
        u = np.zeros((length, input_dim))
        if 0 <= trigger_time < length:
            u[trigger_time, :] = 1.0
        return u

    @staticmethod
    def generate_pulse(length: int, start_time: int, end_time: int, input_dim: int = 1) -> np.ndarray:
        """Generates a sustained square pulse between a start and end time."""
        u = np.zeros((length, input_dim))
        start = max(0, start_time)
        end = min(length, end_time)
        u[start:end, :] = 1.0
        return u

    # =====================================================================
    # THE FITTING ENGINE
    # =====================================================================

    def fit_abc_matrices(self, U: np.ndarray, state_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits the A, B, and C matrices for a linear state-space system driven by input U.
        Extracts the observation data directly from self.observation.
        
        Parameters:
            U (np.ndarray): The known input sequence applied to the system. 
                            Shape (Trials, Timesteps, Input_Dims)
            state_dim (int): The number of hidden states (n).
            
        Returns:
            A (np.ndarray): State transition matrix (state_dim x state_dim)
            B (np.ndarray): Input control matrix (state_dim x Input_Dims)
            C (np.ndarray): Observation matrix (Neurons x state_dim)
            seed (np.ndarray): The initial state vector X(0) to replicate the data.
        """
        # Validate input dimensions match the stored observations
        if U.shape[0] != self.trial_cnt or U.shape[1] != self.timestep_cnt:
            raise ValueError(f"Input U shape {U.shape[:2]} does not match observation "
                             f"trials/timesteps {(self.trial_cnt, self.timestep_cnt)}.")
            
        input_dim = U.shape[2]
        
        # 1. Isolate the State Subspace (X) and Observation Mapping (C) via SVD
        # We flatten across trials and time to capture spatial variance
        Y_flat = self.observation.reshape(-1, self.neuron_cnt)
        
        # Perform SVD: Y = U_svd * Sigma * V^T
        U_svd, Sigma, Vt = np.linalg.svd(Y_flat, full_matrices=False)
        
        # Truncate to the requested state dimension
        C = Vt[:state_dim, :].T  # Shape: (Neurons, state_dim)
        
        # The hidden state trajectory is the remaining SVD components scaled by variance
        X_flat = U_svd[:, :state_dim] @ np.diag(Sigma[:state_dim])
        
        # Reshape X back to 3D so we can safely extract t and t+1 WITHOUT crossing trial boundaries
        X = X_flat.reshape(self.trial_cnt, self.timestep_cnt, state_dim)
        
        # 2. Prepare time-shifted arrays for Least Squares Regression
        # We want to solve: X(t+1) = A * X(t) + B * U(t)
        X_t = X[:, :-1, :].reshape(-1, state_dim)
        X_next = X[:, 1:, :].reshape(-1, state_dim)
        U_t = U[:, :-1, :].reshape(-1, input_dim)
        
        # Concatenate X_t and U_t horizontally to solve for A and B simultaneously
        # Equation becomes: X_next = [X_t, U_t] * [A^T, B^T]^T
        Z = np.hstack((X_t, U_t))
        
        # Solve Ordinary Least Squares (OLS)
        # W contains both A^T and B^T stacked vertically
        W, _, _, _ = np.linalg.lstsq(Z, X_next, rcond=None)
        
        # Extract and transpose back to standard control orientation
        A = W[:state_dim, :].T  # Shape: (state_dim, state_dim)
        B = W[state_dim:, :].T  # Shape: (state_dim, Input_Dim)
        
        # 3. Extract the Seed (Initial State)
        # Average the starting state X(0) across all trials to get a clean replication seed
        seed = np.mean(X[:, 0, :], axis=0)
        
        return A, B, C, seed
    def fit_full_system(self, U: np.ndarray, state_dim: int) -> tuple:
        """
        Fits the complete linear state-space system (A, B, C) and 
        estimates the noise covariance matrices (Q, R).
        
        Returns:
            A, B, C, Q, R, seed
        """
        # Validate dimensions
        if U.shape[0] != self.trial_cnt or U.shape[1] != self.timestep_cnt:
            raise ValueError(f"Input U shape {U.shape[:2]} does not match observation "
                             f"trials/timesteps {(self.trial_cnt, self.timestep_cnt)}.")
            
        input_dim = U.shape[2]
        
        # 1. Isolate the State Subspace (X) and Observation Mapping (C) via SVD
        Y_flat = self.observation.reshape(-1, self.neuron_cnt)
        U_svd, Sigma, Vt = np.linalg.svd(Y_flat, full_matrices=False)
        
        C = Vt[:state_dim, :].T  # Shape: (Neurons, state_dim)
        X_flat = U_svd[:, :state_dim] @ np.diag(Sigma[:state_dim])
        X = X_flat.reshape(self.trial_cnt, self.timestep_cnt, state_dim)
        
        # 2. Least Squares Regression for A and B
        X_t = X[:, :-1, :].reshape(-1, state_dim)
        X_next = X[:, 1:, :].reshape(-1, state_dim)
        U_t = U[:, :-1, :].reshape(-1, input_dim)
        
        Z = np.hstack((X_t, U_t))
        W, _, _, _ = np.linalg.lstsq(Z, X_next, rcond=None)
        
        A = W[:state_dim, :].T
        B = W[state_dim:, :].T
        
        # ==========================================================
        # 3. NOISE COVARIANCE ESTIMATION (Q and R)
        # ==========================================================
        
        # Calculate Observation Residuals v(t) for R
        # Y_pred = X * C^T. We use X_flat to compute over all time/trials instantly.
        Y_pred = X_flat @ C.T
        V_residuals = Y_flat - Y_pred
        
        # R is the covariance of the observation errors (Neurons x Neurons)
        # rowvar=False because our features (neurons) are in columns
        R = np.cov(V_residuals, rowvar=False)
        
        # Calculate Process Residuals w(t) for Q
        # X_next_pred = A*X(t) + B*U(t). Note: Using transposes for aligned batch matrix math.
        X_next_pred = (X_t @ A.T) + (U_t @ B.T)
        W_residuals = X_next - X_next_pred
        
        # Q is the covariance of the process errors (State_Dim x State_Dim)
        Q = np.cov(W_residuals, rowvar=False)

        # 4. Extract Seed
        seed = np.mean(X[:, 0, :], axis=0)
        
        return A, B, C, Q, R, seed