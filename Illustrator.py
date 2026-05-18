from multiprocessing import Value

from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



class Illustrator:
    """
    A class for basic analysis and visualization of neural-style observation data.

    The expected input is a 3D NumPy array with shape:
        (Trials, Timepoints, Neurons)

    Example:
        data.shape = (5, 60, 16)

    This class is designed for Week 1 system exploration.
    It helps us answer questions such as:
        - How do observed signals change over time?
        - Are some neurons/signals more active than others?
        - Are different trials similar?
        - Which signals seem informative?
    """
    
    #slots conflict with our properties so will omit for now
    # __slots__ = ["observation", "trial_cnt", "timestep_cnt", "neuron_cnt"]

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator.

        Parameters
        ----------
        observation : np.ndarray
            A 3D array of shape (Trials, Timepoints, Neurons).
        """
        observation = np.asarray(observation, dtype=float)

        if observation.ndim != 3:
            raise ValueError(
                f"observation must be a 3D array with shape "
                f"(Trials, Timepoints, Neurons), got shape {observation.shape}."
            )

        if not isinstance(observation, np.ndarray):
            raise ValueError('Ensure observation passed as a np array')

        if len(observation.shape)!=3:
            raise ValueError('Incorrect format of the observation. Should be a 3-D array')
        self._observation = observation # keep as protected for keeping data set
        self._trial_cnt, self._timestep_cnt, self._neuron_cnt = observation.shape # keep as protected for keeping data safe


    """
    Prefer working with propeties so that the data is safe.
    Particularly useful in changing the observations
    """
    @property
    def neuron_cnt(self):
        return self._neuron_cnt
    
    @property
    def trial_cnt(self):
        return self._trial_cnt
    
    @property
    def timestep_cnt(self):
        return self._timestep_cnt

    @property
    def observation(self):
        return self._observation
    
    @observation.setter
    def observation(self, new_observation: np.ndarray)->None:
        self._observation = new_observation
        self._trial_cnt, self._timestep_cnt, self._neuron_cnt = new_observation.shape
    
    def get_trial_averaged_mean(self, chosen_neurons: list[int]|np.ndarray|int|None=None)->np.ndarray:
        """
        Use this function to obtain the trial averaged values for the desired neurons
        Eg: for each time step accross 5 trials average the values for each neuron
        If left None, returns time average for all the neurons accross trials
        If list: ensure the list is 0 indexed count of neuron eg: [0,5] returns the 1st and 6th neurons time averaged activity
        The same logic applies for int
        """
        if isinstance(chosen_neurons, list) or isinstance(chosen_neurons, np.ndarray):
            if max(chosen_neurons)>self.neuron_cnt:
                raise ValueError('One of the chosen neurons not in the dataset. Check indexing')
            filtered_neurons = self.observation[:,:,chosen_neurons] # get neural data for the filtered neurons
            return np.mean(filtered_neurons, axis=0) 
        elif isinstance(chosen_neurons, int):
            if chosen_neurons > self.neuron_cnt:
                raise ValueError('The chosen neuron is not in the dataset. Check indexing')
            filtered_neurons = self.observation[:,:,chosen_neurons] # get neural data for the filtered neurons
            return np.atleast_2d(np.mean(filtered_neurons, axis=0)).T # Use atleast 2D such that structure preserved (timesteps, recording count)
        
        elif chosen_neurons is None:
            return np.mean(self.observation[:,:,:],axis=0)
        else:
            raise ValueError('Invalid data type') 

    def get_inter_neuron_covariance_matrix(
            self,
            trial : int| None = None,
            )->np.ndarray:
        """Use this to calculate the covariance matrix accross the neurons
        A_{ij} = E[X_iX_j] - E[X_i]E[X_j] for X_i, X_j neurons
        This can be studied how the overall activity levels influence from one neuron to another

        parameters
        ----------
        trial: int | None
            If specified, calculates the covariance matrix for that specific trial.
            If None, calculates the covariance matrix across all trials and timepoints.

        """
        if trial is not None:
            if not 0 <= trial < self.trial_cnt:
                raise ValueError(f"trial must be between 0 and {self.trial_cnt - 1}.")
            return np.cov(self.observation[trial], rowvar=False)
        
        reshaped_observations = self.observation.reshape(self.trial_cnt*self.timestep_cnt, self.neuron_cnt)
        return np.cov(reshaped_observations,rowvar=False)

    def get_autocovariance_matrix(self, neuron_index: int)->np.ndarray:
        """Use this to get the autocovariance matrix of specific neural process
        C_{t1,t2} = E[X_t1 X_t2] - E[X_t1]E[X_t2]
        This can help understand the temporal variance
        """
        individual_data = self.observation[:,:,neuron_index]
        return np.cov(individual_data, rowvar=False)
    
    def get_trial_covariance_matrix(self):
        """Use this to udnerstand the variance between trials
        This can help validate if the trials are independent or not and form sthe base of the statistical analysis.
        """
        reshaped = self.observation.reshape(self.trial_cnt, self.timestep_cnt*self.neuron_cnt)
        return np.cov(reshaped)
    


    def compare_cov_matrices(self, cov1: np.ndarray, cov2: np.ndarray)->float:
        """
        Use this to compare two covariance matrices

        This will return correlation of flattened upper triangle of the covariance matrices (excluding the diagonal).

        parameters
        ------
        cov1: np.ndarray        
        cov2: np.ndarray

        The covariance matrices to compare. Must have the same shape.
    
        """
        if cov1.shape != cov2.shape:
            raise ValueError("Covariance matrices must have the same shape for comparison.")
        
        # Flatten the upper triangle of the covariance matrices (excluding the diagonal)
        triu_indices = np.triu_indices_from(cov1, k=1)
        cov1_flat = cov1[triu_indices]
        cov2_flat = cov2[triu_indices]

        # Compute the correlation between the flattened covariance values
        correlation = np.corrcoef(cov1_flat, cov2_flat)[0, 1]
        
        return correlation
    

    def get_full_cross_trial_cross_neuron_covariance_tensor(self) -> np.ndarray:
        '''
        Use this to get the full cross trial cross neuron covariance tensor
        C[i,j,a,b] = Cov(X_{trial i, neuron a}, X_{trial j, neuron b})

        for trial i and j and neurons a and b, this gives the covariance between the activity of neuron a in trial i and neuron b in trial j

        this is the most in-depth way to understand the covariance structure of the data, but also the most computationally expensive.

        
        '''

        data = self.observation
        time_points = data.shape[1]
        
        # Center each (trial, neuron) time series
        data_centered = data - data.mean(axis=1, keepdims=True)

        #can use einsum to compute what would 4 nested for loops do in a more efficient way

        # Contract over time: result[i,j,a,b] = sum_t Xc[i,t,a] * Xc[j,t,b]
        cov_tensor = np.einsum('ita,jtb->ijab', data_centered, data_centered) / (time_points - 1)

        return cov_tensor





    def get_dissimilarity_matrix(self) -> np.ndarray:
        '''
        Use this to find the dissimilarity between trials

        Should be able to detect potential mixed modes of trial behavior which may get washed out through the trial covariance matrix. 
        (like cross trial cross neuron covariance tensor but more general)

        we find the 16x16 covariance matrix for each trial then compare the similarity of these
        covariance matrices across each possible pair of tr

        
        '''
            # Compute each trial's covariance matrix once
        per_trial_covs = [
            self.get_inter_neuron_covariance_matrix(trial=t) for t in range(self.trial_cnt)
        ]
        
        dissimiarity_matrix = np.zeros((self.trial_cnt, self.trial_cnt))
        for i in range(self.trial_cnt):
            for j in range(i + 1, self.trial_cnt):
                d = 1 - self.compare_cov_matrices(per_trial_covs[i], per_trial_covs[j])
                dissimiarity_matrix[i, j] = d
                dissimiarity_matrix[j, i] = d  # exploit symmetry
        
        return dissimiarity_matrix

    def plot_matrix(self, matrix: np.ndarray):
        plt.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.title('Matrix Heatmap')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()
                                        


    def summary(self):
        """
        Print basic information and summary statistics of the dataset.
        """
        print("Dataset summary")
        print("----------------")
        print(f"Number of trials: {self.trial_cnt}")
        print(f"Number of timepoints per trial: {self.timestep_cnt}")
        print(f"Number of neurons/signals: {self.neuron_cnt}")
        print(f"Overall mean: {np.mean(self.observation):.4f}")
        print(f"Overall standard deviation: {np.std(self.observation):.4f}")
        print(f"Minimum value: {np.min(self.observation):.4f}")
        print(f"Maximum value: {np.max(self.observation):.4f}")

    def plot_trial(self, trial_id: int = 0):
        """
        Plot all neuron signals over time for one selected trial.

        Parameters
        ----------
        trial_id : int
            Index of the trial to visualize.
        """
        if not 0 <= trial_id < self.trial_cnt:
            raise ValueError(f"trial_id must be between 0 and {self.trial_cnt - 1}.")

        trial_data = self.observation[trial_id]
        time = np.arange(self.timestep_cnt)

        plt.figure(figsize=(10, 5))
        for neuron_id in range(self.neuron_cnt):
            plt.plot(time, trial_data[:, neuron_id], alpha=0.75)

        plt.xlabel("Time step")
        plt.ylabel("Observed activity")
        plt.title(f"All neuron signals in trial {trial_id}")
        plt.grid(alpha=0.3)
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


    def plot_neurons_single_trial(
            self,
            neuron_ids: list[int]|np.ndarray|int|None=None, 
            trial_id: int = 0, 
            function = lambda x: x):
        '''
        plot one or more neurons for a single trial
        
        parameters
        ----------
        neuron_ids: list[int]|np.ndarray|int|None
            The neuron(s) to plot. Can be a single int, a list of ints, or None for all neurons.

        trial_id: int
            The trial to plot. Must be between 0 and trial_cnt - 1.
        
        
        '''
        if not 0 <= trial_id < self.trial_cnt:
            raise ValueError(f"trial_id must be between 0 and {self.trial_cnt - 1}.")
        
        if isinstance(neuron_ids, list) or isinstance(neuron_ids, np.ndarray):
            if max(neuron_ids)> self.neuron_cnt:
                raise ValueError('One of the chosen neurons not in the dataset. Check indexing')
            
            filtered_neurons = self.observation[trial_id][:,neuron_ids] # get neural data for the desired neurons and trials
            time = np.arange(self.timestep_cnt)
            plt.figure(figsize=(10, 5))
            for i, neuron_id in enumerate(neuron_ids):
                
                values = filtered_neurons[:, i]
                transformed_values = function(values) # apply the transformation function to the values | default is identity function
                plt.plot(
                    time,
                    transformed_values,
                    label=f"neuron {neuron_id}",
                    alpha=0.8,
                )
            plt.xlabel("Time step")
            plt.ylabel("Observed activity")
            plt.title(f"Neurons {neuron_ids} in trial {trial_id}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()


    

    def plot_trial_average(self):
        """
        Plot the average activity over trials for each neuron.

        This helps us see the typical time pattern of each observed signal.
        """
        mean_over_trials = np.mean(self.observation, axis=0)
        time = np.arange(self.timestep_cnt)

        plt.figure(figsize=(10, 5))
        for neuron_id in range(self.neuron_cnt):
            plt.plot(time, mean_over_trials[:, neuron_id], alpha=0.75)

        plt.xlabel("Time step")
        plt.ylabel("Average observed activity")
        plt.title("Average activity over trials for each neuron")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_population_average(self):
        """
        Plot the average activity across all neurons and trials over time.

        This gives a simple population-level view of the dataset.
        """
        population_average = np.mean(self.observation, axis=(0, 2))
        time = np.arange(self.timestep_cnt)

        plt.figure(figsize=(8, 4))
        plt.plot(time, population_average, linewidth=2)
        plt.xlabel("Time step")
        plt.ylabel("Average activity")
        plt.title("Population average activity over time")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_heatmap(self, trial_id: int = 0):
        """
        Plot a heatmap of one trial.

        Rows represent timepoints.
        Columns represent neurons/signals.

        Parameters
        ----------
        trial_id : int
            Index of the trial to visualize.
        """
        if not 0 <= trial_id < self.trial_cnt:
            raise ValueError(f"trial_id must be between 0 and {self.trial_cnt - 1}.")

        plt.figure(figsize=(9, 5))
        plt.imshow(
            self.observation[trial_id],
            aspect="auto",
            origin="lower",
        )
        plt.colorbar(label="Observed activity")
        plt.xlabel("Neuron / signal index")
        plt.ylabel("Time step")
        plt.title(f"Heatmap of trial {trial_id}")
        plt.show()

    def plot_neuron_variance(self):
        """
        Plot the variance of each neuron/signal across trials and time.

        A higher variance may suggest that a neuron/signal carries more visible dynamics.
        """
        neuron_variance = np.var(self.observation, axis=(0, 1))

        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(self.neuron_cnt), neuron_variance)
        plt.xlabel("Neuron / signal index")
        plt.ylabel("Variance")
        plt.title("Variance of each neuron/signal")
        plt.grid(axis="y", alpha=0.3)
        plt.show()

    def neuron_statistics(self):
        """
        Compute useful summary statistics for each neuron/signal.

        all stats are averaged over all trials

        Returns
        -------
        stats : dict
            Dictionary containing mean, std, variance, peak value, peak time and peak slope
            for each neuron.
        """
        mean_per_neuron = np.mean(self.observation, axis=(0, 1))
        std_per_neuron = np.std(self.observation, axis=(0, 1))
        var_per_neuron = np.var(self.observation, axis=(0, 1))

        # Average over trials first, then find peak over time
        mean_timecourse = np.mean(self.observation, axis=0)  # shape: (time, neuron)
        peak_value = np.max(mean_timecourse, axis=0)
        peak_time = np.argmax(mean_timecourse, axis=0)

        # would maybe want to change for general savgol filter use 
        #but would be less clean stats method, will leave for now

        peak_slope = np.max(np.abs(savgol_filter(
            mean_timecourse,
            window_length=11,
            polyorder=3,
            deriv=1,
            delta = 1.0,
            axis =0,

        )), axis=0)

        stats = {
            "mean": mean_per_neuron,
            "std": std_per_neuron,
            "variance": var_per_neuron,
            "peak_value": peak_value,
            "peak_time": peak_time,
            "peak_slope": peak_slope,
        }

        print("Neuron statistics")
        print("-----------------")
        for neuron_id in range(self.neuron_cnt):
            print(
                f"Neuron {neuron_id:2d}: "
                f"mean={mean_per_neuron[neuron_id]:.3f}, "
                f"std={std_per_neuron[neuron_id]:.3f}, "
                f"var={var_per_neuron[neuron_id]:.3f}, "
                f"peak={peak_value[neuron_id]:.3f}, "
                f"peak_time={peak_time[neuron_id]},"
                f"peak_slope={peak_slope[neuron_id]:.3f}"
            )

        return stats
    
    def plot_neuron_mean_activity(self):
        """
        Plot the mean activity of each neuron across all trials and timepoints.
        """
        mean_per_neuron = np.mean(self.observation, axis=(0, 1))

        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(self.neuron_cnt), mean_per_neuron)
        plt.xlabel("Neuron / signal index")
        plt.ylabel("Mean activity")
        plt.title("Mean activity of each neuron/signal")
        plt.grid(axis="y", alpha=0.3)
        plt.show()

    def plot_neuron_with_trial_std(self, neuron_id: int = 0):
        """
        Plot one neuron averaged across trials, with trial-to-trial standard deviation.

        This is useful for checking whether repeated trials are identical or noisy.
        """
        if not 0 <= neuron_id < self.neuron_cnt:
            raise ValueError(f"neuron_id must be between 0 and {self.neuron_cnt - 1}.")

        mean_signal = np.mean(self.observation[:, :, neuron_id], axis=0)
        std_signal = np.std(self.observation[:, :, neuron_id], axis=0)
        time = np.arange(self.timestep_cnt)

        plt.figure(figsize=(10, 5))
        plt.plot(time, mean_signal, label=f"neuron {neuron_id} mean", linewidth=2)
        plt.fill_between(
            time,
            mean_signal - std_signal,
            mean_signal + std_signal,
            alpha=0.25,
            label="±1 trial std",
        )

        plt.xlabel("Time step")
        plt.ylabel("Observed activity")
        plt.title(f"Neuron {neuron_id}: mean ± trial std")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        print(f"Mean trial std for neuron {neuron_id}: {np.mean(std_signal):.6f}")
        print(f"Max trial std for neuron {neuron_id}: {np.max(std_signal):.6f}")

    def trial_variability_report(self):
        """
        Report trial-to-trial variability for each neuron.

        Low values mean trials are very similar.
        Values close to zero mean repeated trials are almost identical.
        """
        trial_std = np.std(self.observation, axis=0)  # shape: (time, neuron)
        mean_trial_std_per_neuron = np.mean(trial_std, axis=0)
        max_trial_std_per_neuron = np.max(trial_std, axis=0)

        print("Trial-to-trial variability report")
        print("---------------------------------")
        print(f"Overall mean trial std: {np.mean(trial_std):.6f}")
        print(f"Overall max trial std: {np.max(trial_std):.6f}")

        for neuron_id in range(self.neuron_cnt):
            print(
                f"Neuron {neuron_id:2d}: "
                f"mean trial std={mean_trial_std_per_neuron[neuron_id]:.6f}, "
                f"max trial std={max_trial_std_per_neuron[neuron_id]:.6f}"
            )

        return mean_trial_std_per_neuron, max_trial_std_per_neuron
    
    def plot_correlation_matrix(self):
        """
        Plot correlation matrix between neurons/signals.

        Strong correlations suggest that multiple observed channels may be driven by
        shared latent dynamics.
        """
        # Combine trials and time into one sample dimension
        flattened = self.observation.reshape(-1, self.neuron_cnt)

        corr = np.corrcoef(flattened, rowvar=False)

        plt.figure(figsize=(7, 6))
        plt.imshow(corr, vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xlabel("Neuron / signal index")
        plt.ylabel("Neuron / signal index")
        plt.title("Correlation matrix between neurons/signals")
        plt.show()

        return corr
    
    def pca_energy(self):
        """
        Perform a simple PCA/SVD energy analysis on the observation data.

        This helps check whether high-dimensional observations may be explained by
        a lower-dimensional latent structure.
        """
        Y = self.observation.reshape(-1, self.neuron_cnt)

        # Center data
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

        # SVD
        U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)

        explained_variance = S**2 / np.sum(S**2)
        cumulative_variance = np.cumsum(explained_variance)

        print("PCA/SVD energy")
        print("--------------")
        for i in range(len(explained_variance)):
            print(
                f"Component {i + 1:2d}: "
                f"explained={explained_variance[i]:.4f}, "
                f"cumulative={cumulative_variance[i]:.4f}"
            )

        plt.figure(figsize=(8, 4))
        plt.plot(
            np.arange(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o",
        )
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA/SVD cumulative explained variance")
        plt.grid(alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()

        return explained_variance, cumulative_variance

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