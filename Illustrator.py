import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.signal import savgol_filter


class Illustrator:
    """
    A class for basic analysis and visualization of neural-style observation data.

    The expected input is a 3D NumPy array with shape:
        (Trials, Timepoints, Neurons)

    Example:
        data.shape = (5, 60, 16)
    """

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

    #Use of properties applies encapsulation

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



    def spam_everything(self):
        """
        A simple function to call all the plotting and statistics methods in one go.

        This is useful for quickly getting a comprehensive overview of the dataset.

        useful for demonstrating all the abilities of the illustrator class
        """
        self.summary()
        self.neuron_statistics()
        self.trial_variability_report()
        self.plot_general()
        self.plot_trial_average()
        self.plot_population_average()
        self.plot_heatmap()
        self.plot_neuron_variance()
        self.plot_neuron_mean_activity()
        self.plot_neuron_with_trial_std(neuron_id=0)
        self.plot_correlation_matrix()
        self.pca_energy()
        self.plot_PSD()



    ################################################################################################
    ################################################################################################

    
    #       Getting statistics methods


    ################################################################################################
    ################################################################################################



    
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
        """
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
    

    def get_PSD(
            self,
            neuron_list: np.ndarray,
            time_list: np.ndarray,
            trial_list: np.ndarray,
    ):
        """Use this to get the power spectral density of the neural processes
        This can help understand the frequency content of the neural signals and identify any rhythmic patterns or oscillations.
        """
        from scipy.signal import welch

        if neuron_list is None:
            neuron_list = np.arange(self.neuron_cnt)
        if time_list is None:
            time_list = np.arange(self.timestep_cnt)
        if trial_list is None:
            trial_list = np.arange(self.trial_cnt)

        #of the trials selected we are going to concantenate the data

        psd_results = {}
        for neuron in neuron_list:
            psd_results[neuron] = []

            neuron_data = self.observation[trial_list][:, time_list, neuron].reshape(-1)  # Concatenate across trials and time
            freqs, psd = welch(neuron_data, fs=1.0 / (time_list[1] - time_list[0]), nperseg=min(256, len(neuron_data)))
            psd_results[neuron].append((freqs, psd))
             
        return psd_results

    def empirical_observability_gramian(self):

        pass

        
  




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

    def neuron_statistics(self):
        """
        Compute useful summary statistics for each neuron/signal.

        Returns
        -------
        stats : dict
            Dictionary containing mean, std, variance, peak value, and peak time
            for each neuron.
        """
        mean_per_neuron = np.mean(self.observation, axis=(0, 1))
        std_per_neuron = np.std(self.observation, axis=(0, 1))
        var_per_neuron = np.var(self.observation, axis=(0, 1))

        # Average over trials first, then find peak over time
        mean_timecourse = np.mean(self.observation, axis=0)  # shape: (time, neuron)
        peak_value = np.max(mean_timecourse, axis=0)
        peak_time = np.argmax(mean_timecourse, axis=0)

        # Peak slope: smooth + differentiate, then take max magnitude per neuron
        slopes = savgol_filter(
            mean_timecourse,
            window_length=5,
            polyorder=2,
            deriv=1,
            delta=1.0,
            axis=0,                      # IMPORTANT: differentiate along time
        )
        peak_slope = np.max(np.abs(slopes), axis=0)

        stats = {
            "mean": mean_per_neuron,
            "std": std_per_neuron,
            "variance": var_per_neuron,
            "peak_value": peak_value,
            "peak_time": peak_time,
            "peak slope": peak_slope,
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
                f"peak_time={peak_time[neuron_id]}"
                f"peak_slope = {peak_slope[neuron_id]}"
            )

        return stats

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







    ################################################################################################
    ################################################################################################

    
    #       Plotting methods


    ################################################################################################
    ################################################################################################




    

    #refactored the plotting trials/neurons into a more general scatter plot function

    def plot_general(
            self,
            neuron_list: np.ndarray|None = None,
            time_list: np.ndarray|None = None,
            trial_list: np.ndarray|None = None,

            ):

            default_trial_time_neuron_lists = self.default_trial_time_neuron_list()

            if neuron_list is None:
                neuron_list = default_trial_time_neuron_lists[2]
            if time_list is None:
                time_list = default_trial_time_neuron_lists[1]
            if trial_list is None:
                trial_list = default_trial_time_neuron_lists[0]
            #given the lists are not None
            if neuron_list is not None:
                if max(neuron_list) >= self.neuron_cnt or min(neuron_list) < 0:
                    raise ValueError("neuron_list contains invalid neuron indices.")
            if time_list is not None:
                if max(time_list) >= self.timestep_cnt or min(time_list) < 0:
                    raise ValueError("time_list contains invalid time indices.")
            if trial_list is not None:
                if max(trial_list) >= self.trial_cnt or min(trial_list) < 0:
                    raise ValueError("trial_list contains invalid trial indices.")

            plt.figure(figsize=(10, 6))
            if (neuron_list is not None) and (trial_list is not None) and (time_list is not None):
                for trial in trial_list:
                    for neuron in neuron_list:
                        plt.plot(time_list, self.observation[trial, time_list, neuron], label=f"Trial {trial}, Neuron {neuron}", alpha=0.6)
            else:
                raise ValueError("At least one of neuron_list, time_list, or trial_list must be provided.")

            plt.xlabel("Time step")
            plt.ylabel("Observed activity")
            plt.title("Scatter plot of selected neurons over time, accross trials {}".format(trial_list))
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


    def plot_PSD(
            self, 
            neuron_list: np.ndarray|None = None,
            time_list: np.ndarray|None = None,
            trial_list: np.ndarray|None = None
            ):
        '''
        plot PSD
        
        '''
        if (neuron_list is not None) and (trial_list is not None) and (time_list is not None):
            psd_results = self.get_PSD(neuron_list, time_list, trial_list)
        else:
            default_trial_time_neuron_lists = self.default_trial_time_neuron_list()
            psd_results = self.get_PSD(default_trial_time_neuron_lists[2], default_trial_time_neuron_lists[1], default_trial_time_neuron_lists[0]) 
        

        plt.figure(figsize=(10, 6))
        for neuron, psd_list in psd_results.items():
            for trial_idx, (freqs, psd) in enumerate(psd_list):
                plt.plot(freqs, psd, label=f"Neuron {neuron}, Trial {trial_idx}", alpha=0.6)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.title("Power Spectral Density of neurons {} accross concatenated trial data from trials {1}".format(neuron_list, trial_list))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    
    
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

   

 
    #--------------------------------------------------------------------
    #matrix visualisation

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
        
        return matrices
    


    


    #-------------------------------------------------------------------

    #hankel matrix method


    
    #--------------------------------------------------------------------


    #matrix visualisation

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

    # --------------------------------------------------------------------
    #scree plot
        
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





    

 ################################################################################################
################################################################################################


#      Helper functions


################################################################################################
################################################################################################


    def default_trial_time_neuron_list(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A helper function to get default lists of trial, time, and neuron indices for plotting.

        This can be used to quickly generate standard plots without needing to specify indices.
        """

        #will just plot all

        neuron_list = np.arange(self.neuron_cnt)  
        time_list = np.arange(self.timestep_cnt)  
        trial_list = np.arange(self.trial_cnt)

        return trial_list, time_list, neuron_list
