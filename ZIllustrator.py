from cProfile import label
from logging import raiseExceptions
import time
from typing import Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import adfuller
from scipy.signal import spectrogram, coherence, welch
from ZLDSParams import LDSParams
from ZSimulator import Simulator


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

        self._observation = observation # keep as protected for keeping data set
        self._trial_cnt, self._timestep_cnt, self._neuron_cnt = observation.shape # keep as protected for keeping data safe

    #Use of properties applies encapsulation

    @property
    def neuron_cnt(self)->int:
        return self._neuron_cnt
    
    @property
    def trial_cnt(self)->int:
        return self._trial_cnt
    
    @property
    def timestep_cnt(self)->int:
        return self._timestep_cnt

    @property
    def observation(self):
        return self._observation
    
    @observation.setter
    def observation(self, new_observation: np.ndarray)->None:
        if new_observation.ndim != 3:
            raise ValueError(
                f"observation must be a 3D array with shape "
                f"(Trials, Timepoints, Neurons), got shape {new_observation.shape}."
            )

        if not isinstance(new_observation, np.ndarray):
            raise ValueError('Ensure observation passed as a np array')

        self._observation = new_observation
        self._trial_cnt, self._timestep_cnt, self._neuron_cnt = new_observation.shape

    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Statistical Analysis ---------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###

    
    # Function to average accross all trials
    def get_trial_average(self, chosen_neurons: list[int]|np.ndarray|int|None=None, chosen_trials: list[int]|np.ndarray|None=None)->np.ndarray:
        """
        Use this function to average the neural accross trials

        Args:
            chosen_neurons (list[int] | np.ndarray | int | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: chosen Neuron index out of range
            ValueError: invalid datatype 

        Returns:
            np.ndarray: 2 dimensional array (time_steps, number of neurons selected)
        """
        if chosen_trials is None:
            chosen_trials = np.arange(0, self.trial_cnt)
        if isinstance(chosen_neurons, list) or isinstance(chosen_neurons, np.ndarray):
            if max(chosen_neurons)>=self.neuron_cnt:
                raise ValueError('One of the chosen neurons not in the dataset. Check indexing')
            filtered_neurons = self.observation[chosen_trials][:,:,chosen_neurons] # get neural data for the filtered neurons
            return np.mean(filtered_neurons, axis=0) 
        elif isinstance(chosen_neurons, int):
            if chosen_neurons >= self.neuron_cnt:
                raise ValueError('The chosen neuron is not in the dataset. Check indexing')
            filtered_neurons = self.observation[chosen_trials][:,:,chosen_neurons] # get neural data for the filtered neurons
            return np.atleast_2d(np.mean(filtered_neurons, axis=0)).T # Use atleast 2D such that structure preserved (timesteps, recording count)
        
        elif chosen_neurons is None:
            return np.mean(self.observation[chosen_trials,:,:],axis=0)
        else:
            raise ValueError('Invalid data type') 
    
    def get_mean_per_neuron(self)->np.ndarray:
        """
        Calculate the mean value for each neuron accross timesteps and trials

        Returns:
            np.ndarray: returns an array of length (neuron_cnt)
        """
        return np.mean(self.observation, axis=(0, 1))
    
    def get_std_per_neuron(self)->np.ndarray:
        """
        Calculate the std value for each neuron accross timesteps and trials

        Returns:
            np.ndarray: returns an array of length (neuron_cnt)
        """
        return np.std(self.observation, axis=(0,1))
    
    def get_variance_per_neuron(self)->np.ndarray:
        """
        Calculate the variance value for each neuron accross timesteps and trials

        Returns:
            np.ndarray: returns an array of length (neuron_cnt)
        """
        return np.var(self.observation, axis=(0,1))
    
    
    def get_signal_energy(self)->np.ndarray:
        """
        Calculate the signal energy (E[X^2]) value for each neuron accross timesteps and trials

        Returns:
            np.ndarray: returns an array of length (neuron_cnt)
        """
        return np.mean(np.square(self.observation), axis=(0,1))
    
    def get_trial_averaged_peak_value_time(self):
        """
        Calculate the peak value for each neuron accross trials

        Returns:
            np.ndarray: returns an array of length (neuron_cnt)
        """
        mean_timecourse = self.get_trial_average()
        return np.max(mean_timecourse, axis=0), np.argmax(mean_timecourse, axis=0)
    
    def get_response_slopes(self):
        """
        Calculate the time derivative for each neuron accross trials using SavGol filter

        Returns:
            np.ndarray: returns an array of shape (time_step, neuron_cnt)
        """
        mean_timecourse = self.get_trial_average()
        # Savgol filter cleans up noise, polyorder 2-> quadratic, deriv-> linear slope
        return np.asarray(savgol_filter(
                                        mean_timecourse,
                                        window_length=5,
                                        polyorder=2,
                                        deriv=1,
                                        delta=1.0,
                                        axis=0,                      # IMPORTANT: differentiate along time
                                    ))

    def get_trial_variance_per_neuron(self)->np.ndarray:
        """
        Calculate the variance value for each neuron accross trials

        Returns:
            np.ndarray: returns an array of shape (time_steps, neuron_cnt)
        """
        return np.var(self.observation, axis=0)
    
    def get_neuron_statistics(self)->dict[str, np.ndarray]:
        """
        Compute useful summary statistics for each neuron/signal.
        Assumes stationaryness of signal => Constant Mean (this may not be true if there is drift)
        Returns
        -------
        stats : dict
            Dictionary containing mean, std, variance, peak value, and peak time
            for each neuron.
        """
        mean_per_neuron = self.get_mean_per_neuron()
        std_per_neuron = self.get_std_per_neuron()
        var_per_neuron = self.get_variance_per_neuron()
        signal_energy_per_neuron = self.get_signal_energy() 
        

        # Average over trials first, then find peak over time
        peak_value, peak_time = self.get_trial_averaged_peak_value_time()

        # Peak slope: smooth + differentiate, then take max magnitude per neuron
        slopes = self.get_response_slopes()
        peak_slope = np.max(np.abs(slopes), axis=0)

        stats = {
            "mean": mean_per_neuron,
            'std': std_per_neuron,
            'signal energy': signal_energy_per_neuron, 
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
                f"energy={signal_energy_per_neuron[neuron_id]: .3f}, "
                f"var={var_per_neuron[neuron_id]:.3f}, "
                f"peak={peak_value[neuron_id]:.3f}, "
                f"peak_time={peak_time[neuron_id]: .3f}, "
                f"peak_slope={peak_slope[neuron_id]:.3f}"
            )

        return stats
    
    def get_PSD(self, neuron_list: list[int]|np.ndarray|None=None, time_list: list[int]|np.ndarray|None=None,
                trial_list: list[int]|np.ndarray|None=None, dt: float=1.0, nperseg: int|None=None, average_across_trials: bool=False)->dict[np.int64, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the Power Spectrum density for chosen neurons over a custom time region and custom trials.
        Can use Evoked PSD (average nerual response accross specified trials ) or Induced (average the PSD accross trials)

        Args:
            neuron_list (list[int] | np.ndarray | None, optional): 
            list of neurons for whom psd to be calculated.
            Defaults to none implies for all neurons

            time_list (list[int] | np.ndarray | None, optional):
            timesteps indices at which psd to be calculated
            Defaults to none implies for all timesteps
            
            trial_list (list[int] | np.ndarray | None, optional): 
            trials considered for recording psd
            Defaults to None implies for all trials

            dt (float): sampling time defaults to 1 

            average_across_trials (bool): if true, first sample accross trials and then calculate psd
            Defaults to False

        Returns:
            dict[np.int64, Tuple[np.ndarray, np.ndarray]]: {neuron_idx: (freqs, psd: np.float)}
        """

        if neuron_list is None:
            neuron_list = np.arange(self.neuron_cnt)
        if time_list is None:
            time_list = np.arange(self.timestep_cnt)
        if trial_list is None:
            trial_list = np.arange(self.trial_cnt)

        if isinstance(neuron_list, int):
            neuron_list = [neuron_list]
        
        if isinstance(trial_list, int):
            trial_list = [trial_list]
        
        if nperseg is None:
            nperseg = min(256, max(8,len(time_list)//8)) 
        
        fs = 1/dt # Sampling psd
        psd_results = {}


        if average_across_trials:

            x_all = self.get_trial_average(neuron_list, trial_list)  # (time, neurons)
            x_all = x_all[time_list, :]

            for i, neuron in enumerate(neuron_list):
                signal = x_all[:, i]

                freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

                psd_results[neuron] = (freqs, psd)
        
        else:

            for neuron in neuron_list:

                trial_psds = []

                for trial in trial_list:

                    x = self.observation[trial][time_list, neuron]

                    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

                    trial_psds.append(psd)

                mean_psd = np.mean(trial_psds, axis=0)

                psd_results[neuron] = (freqs, mean_psd)

        return psd_results
             
    def get_fft(self, neuron_list: list[int]|np.ndarray|None=None, time_list: list[int]|np.ndarray|None=None,
                trial_list: list[int]|np.ndarray|None=None, dt: float=1.0, average_across_trials: bool=False)->dict[np.int64, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the Fourier Transform for chosen neurons over a custom time region and custom trials.
    
        Args:
            neuron_list (list[int] | np.ndarray | None, optional): 
            list of neurons for whom ft to be calculated.
            Defaults to none implies for all neurons

            time_list (list[int] | np.ndarray | None, optional):
            timesteps indices at which ft to be calculated
            Defaults to none implies for all timesteps
            
            trial_list (list[int] | np.ndarray | None, optional): 
            trials considered for recording ft
            Defaults to None implies for all trials

            dt (float): sampling time defaults to 1 

            npersg (int): window length defaults to min(256, max(8,len(time_list)//8)) 

            average_across_trials (bool): if true, first sample accross trials and then calculate fft
            if false can destroy phase information
            A better implementation in the future would be distinguish between magnitude an
            
            Defaults to False

        Returns:
            dict[np.int64, Tuple[np.ndarray, np.ndarray]]: {neuron_idx: (freqs, fft: np.complex)}
        """

        if neuron_list is None:
            neuron_list = np.arange(self.neuron_cnt)
        if time_list is None:
            time_list = np.arange(self.timestep_cnt)
        if trial_list is None:
            trial_list = np.arange(self.trial_cnt)

        if isinstance(neuron_list, int):
            neuron_list = [neuron_list]
        
        if isinstance(trial_list, int):
            trial_list = [trial_list]
        
        
        
        n_time = len(time_list)

        fft_results= {}
        freqs = np.fft.fftfreq(n_time, d=dt)
        
        if average_across_trials:
        # average across trials first (evoked signal)
            x_all = self.get_trial_average(neuron_list, trial_list)  # (time, neurons)
            x_all = x_all[time_list, :]

            for i, neuron in enumerate(neuron_list):
                signal = x_all[:, i]
                spectrum = np.fft.fft(signal)
                fft_results[neuron] = (freqs, spectrum)

        else:
            # FFT per trial, then average complex spectrum (evoked-like averaging)
            for neuron in neuron_list:
                spectra = []

                for trial in trial_list:
                    signal = self.observation[trial][time_list, neuron]
                    spectra.append(np.fft.fft(signal))

                mean_spectrum = np.mean(spectra, axis=0)
                fft_results[neuron] = (freqs, mean_spectrum)

        return fft_results
    
        
         
    def get_spectrogram(self,  neuron_list: int|list[int] | np.ndarray | None = None, time_list: list[int] | np.ndarray | None = None,
                        trial_list: list[int] | np.ndarray | None = None, dt: float = 1.0, nperseg: int|None=None,
                        average_across_trials: bool = False) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the spectogram for chosen neurons over a custom time region and custom trials.
    
        Args:
            neuron_list (list[int] | np.ndarray | None, optional): 
            list of neurons for whom spectogram to be calculated.
            Defaults to none implies for all neurons

            time_list (list[int] | np.ndarray | None, optional):
            timesteps indices at which spectogram to be calculated
            Defaults to none implies for all timesteps
            
            trial_list (list[int] | np.ndarray | None, optional): 
            trials considered for recording spectogram
            Defaults to None implies for all trials

            dt (float): sampling time defaults to 1

            npersg (int): window length defaults to min(256, max(8,len(time_list)//8)) 

            average_across_trials (bool): if true, first sample accross trials and then calculate spectogram
            Defaults to False

        Returns:
            dict[np.int64, Tuple[np.ndarray, np.ndarray, np.ndarray]]: {neuron_idx: (freqs, time, spectogram)}
        """

        if neuron_list is None:
            neuron_list = np.arange(self.neuron_cnt)
        if time_list is None:
            time_list = np.arange(self.timestep_cnt)
        if trial_list is None:
            trial_list = np.arange(self.trial_cnt)
        
        if nperseg is None:
            nperseg = min(256, max(8,len(time_list)//8)) 

        if isinstance(neuron_list, (int, np.integer)):
            neuron_list = [int(neuron_list)]
        if isinstance(trial_list, (int, np.integer)):
            trial_list = [int(trial_list)]

        fs = 1.0 / dt
        res = {}
        if average_across_trials:
            # average across trials first (evoked signal)
            x_all = self.get_trial_average(neuron_list, trial_list)
            x_all = x_all[time_list, :]

            for i,neuron in enumerate(neuron_list):
                signal = x_all[:, i]

                f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)

                res[neuron] = (f, t, Sxx)
        else:

            for neuron in neuron_list:

                Sxx_trials = []
    
                for trial in trial_list:
                    signal = self.observation[trial][time_list, neuron]

                    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
                    Sxx_trials.append(Sxx)

                
                Sxx_mean = np.mean(Sxx_trials, axis=0)

                res[neuron] = (f, t, Sxx_mean)

        return res
        

    def _get_coherence(self,neuron1: int, neuron2: int, time_list: list[int] | np.ndarray | None = None,
                    trial_list: list[int] | np.ndarray | None = None, dt: float = 1.0,
                    average_across_trials: bool = False)->Tuple[np.ndarray, np.ndarray]:
        """
        Get the coherence between 2 neurons over a custom time region and custom trials.
        Work in progress. There might me a bug in this? 
        Args:
            neuron1 (int): neuron 1
            neuron2 (int): neuron 2
            
            time_list (list[int] | np.ndarray | None, optional):
            timesteps indices at which spectogram to be calculated
            Defaults to none implies for all timesteps
            
            trial_list (list[int] | np.ndarray | None, optional): 
            trials considered for recording spectogram
            Defaults to None implies for all trials

            dt (float): sampling time defaults to 1 

            average_across_trials (bool): if true, first sample accross trials and then calculate spectogram
            Defaults to False

        Returns:
            Tuple[np.ndarray, np.ndarray]: frequency, and coherence values
        """
        if time_list is None:
            time_list = np.arange(self.timestep_cnt)

        if trial_list is None:
            trial_list = np.arange(self.trial_cnt)

        fs = 1.0 / dt

        
        if average_across_trials:

            x1 = self.get_trial_average(neuron1)[time_list,0]
            x2 = self.get_trial_average(neuron2)[time_list,0]
            print(x1)
            print(x2)
            f, cxy = coherence(x1, x2, fs=fs)
            return f, cxy

        else:

            coh_trials = []

            for trial in trial_list:
                x1 = self.observation[trial][time_list, neuron1]
                x2 = self.observation[trial][time_list, neuron2]

                f, cxy = coherence(x1, x2, fs=fs)
                coh_trials.append(cxy)

            return f, np.mean(coh_trials, axis=0)
        

    def get_inter_neuron_covariance_matrix(self, trial_idx: int|None=None)->np.ndarray:
        """
        Obtain the covariance matrix
        C_{i,j} = E[X_i X_j] - E[X_i]E[X_j]
        Args:
            trial_idx (int | None, optional): specify trial for which calculated. 
            Defaults to None which returns covariance matrix accross all trials.

        Returns: 
            np.ndarray: shape (neuron_count, neuron_count) which is the covariance matrix
        """

        if trial_idx is not None:
            data = self.observation[trial_idx, :, :]
            return np.cov(data, rowvar=False)
        reshaped_observations = self.observation.reshape(self.trial_cnt*self.timestep_cnt, self.neuron_cnt)
        return np.cov(reshaped_observations,rowvar=False)
    
    def get_correlation_matrix(self)->np.ndarray:
        """
        Obtain the correlation matrix

        Returns:
            np.ndarray: (neural_cnt, neural_cnt) correlation matrix
        """
        flattened = self.observation.reshape(-1, self.neuron_cnt)
        corr = np.corrcoef(flattened, rowvar=False)
        return corr
    
    
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Dimensionality Reduction ------------------------------------ ###
    ### ---------------------------------------------------------------------------------------------------- ###

    def get_SVD(self):
        """
        Obtain the SVD of the dataset

        Returns:
           SVD result from numpy
           U, S, Vh
        """
        
        Y = self.observation.reshape(-1, self.neuron_cnt)

        # Center data
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

        # SVD
        SVD = np.linalg.svd(Y_centered, full_matrices=False)
        return SVD
    
    def plot_PCA(self, threshold: float = 0.9, n_show: int | None = None,
                trials_to_plot: int | list[int] | None = None):
        """
        Visualize PCA decomposition of the observation.

        Three figures:
        1. Scree + cumulative variance — pick state dimensionality
        2. Principal vector loadings — how each neuron contributes to each PC
        3. PC scores over time — trial-averaged trajectory of each PC

        Parameters
        ----------
        threshold : float
            Cumulative variance threshold passed through to get_PCA.
        n_show : int | None
            Number of components to display in plots 2 and 3.
            Defaults to `components_needed` from get_PCA.
        trials_to_plot : int | list | None
            Which individual trials to overlay on plot 3. Trial-average is always shown.
            Default: no individual trials.
        """
        explained, cumulative, k, vectors, scores = self.get_PCA(threshold=threshold)
        if n_show is None:
            n_show = k
        n_show = min(n_show, vectors.shape[1])

        # ---- 1. Scree + cumulative ----
        fig, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(1, len(explained) + 1)
        ax1.bar(x, explained * 100, alpha=0.7, label="explained")
        ax1.set_xlabel("component")
        ax1.set_ylabel("variance explained (%)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(x, cumulative * 100, "o-", color="tab:orange", label="cumulative")
        ax2.axhline(threshold * 100, linestyle="--", color="gray", alpha=0.5)
        ax2.axvline(k, linestyle="--", color="gray", alpha=0.5)
        ax2.set_ylabel("cumulative variance (%)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0, 105)
        ax1.set_title(f"PCA scree — {k} components for {threshold*100:.0f}% variance")
        fig.tight_layout()
        plt.show()

        # ---- 2. Principal vector loadings ----
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n_show)))
        im = ax.imshow(vectors[:, :n_show].T, aspect="auto", cmap="bwr",
                    vmin=-np.max(np.abs(vectors[:, :n_show])),
                    vmax= np.max(np.abs(vectors[:, :n_show])))
        plt.colorbar(im, ax=ax, label="loading")
        ax.set_xlabel("neuron")
        ax.set_ylabel("principal component")
        ax.set_yticks(np.arange(n_show))
        ax.set_yticklabels([f"PC{i+1}" for i in range(n_show)])
        ax.set_title("Principal vector loadings")
        fig.tight_layout()
        plt.show()

        # ---- 3. PC scores over time ----
        if isinstance(trials_to_plot, int):
            trials_to_plot = [trials_to_plot]

        t = np.arange(self.timestep_cnt)
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.8 * n_show),
                                sharex=True, squeeze=False)
        mean_scores = scores.mean(axis=0)   # (T, k)

        for i in range(n_show):
            ax = axes[i, 0]
            if trials_to_plot:
                for tr in trials_to_plot:
                    ax.plot(t, scores[tr, :, i], alpha=0.3, color="gray")
            ax.plot(t, mean_scores[:, i], linewidth=2, label=f"PC{i+1} mean")
            ax.set_ylabel(f"PC{i+1}")
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("time")
        fig.suptitle("PC scores over time (trial-averaged)")
        fig.tight_layout()
        plt.show()
    
    def get_PCA(self, SVD:Tuple[np.ndarray, np.ndarray, np.ndarray]|None=None, 
                threshold: float=0.9)->Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Obtain Principal Components

        Args:
            SVD (Tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional): 
            SVD =  (U, S, Vh)
            Defaults to None. If None, function calculates PCA
            threshold (float, optional): Threshold to . Defaults to 0.9.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
            explained, variance, cumulative variance, components needed, principal vectors, scores (activations of reduced eigen vectors)
        """
        if SVD is None: 
           SVD = self.get_SVD()

        U, S, Vh = SVD
        explained_variance = S**2 / np.sum(S**2)
        cumulative_variance = np.cumsum(explained_variance)
        components_needed = np.where(cumulative_variance>threshold)[0][0] + 1
        V = Vh.T
        principal_vectors = V[:,:components_needed]
        principal_scores = U[:,:components_needed] @ np.diag(S[:components_needed])
        scores = principal_scores.reshape(self.trial_cnt, self.timestep_cnt, components_needed)
        return explained_variance, cumulative_variance, components_needed, principal_vectors, scores
    


        


    
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Time Series Analysis ---------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###
    
    def test_stationaryness(self, time_list: list[int]|np.ndarray|None=None)->dict[int, float]:
        """
        ADF test is used to determine the presence of unit root in the series, and hence helps in understand if the series is stationary or not. T
        he null and alternate hypothesis of this test are:

        Null Hypothesis: The series has a unit root. (series non stationary)

        Alternate Hypothesis: The series has no unit root. (series stationary)

        If the null hypothesis in failed to be rejected, this test may provide evidence that the series is non-stationary.

        If pval < threshold => Reject Null Hypothesis, and conclude time series stationary
        (from wikipedia)

        Args:
            time_list (list[int] | np.ndarray | None, optional):
            List of time_indices to test for stationaryness 
            Defaults to None => see accross fullset.
        
        Returns:
            dict[int, float]: dictionary mapping neuron index to p val
        """
        if time_list is None:
            _, time_list, _ = self.default_trial_time_neuron_list()

        time_averaged = self.get_trial_average()[time_list, :]
        pvals = {}

        for neuron_idx in range(time_averaged.shape[1]):

            signal = time_averaged[:, neuron_idx]

            # Optional safety check
            if np.all(signal == signal[0]):
                pvals[neuron_idx] = np.nan
                continue

            adf_result = adfuller(signal)

            pvals[neuron_idx] = adf_result[1]

        return pvals

    def get_time_covariance_matrix(self, neuron_index: int|list|None=None, trial_index: list[int] | np.ndarray | None = None,
                                  time_list: list[int]|np.ndarray|None=None, normalized = False)->dict[int, np.ndarray]:
        """
        Work in progress and potentially has a bug
        Compute the time–time covariance (or correlation) structure of neural activity
        for selected neurons across trials.

        Args:
            neuron_index (int | list | None, optional): 
            Index or list of neuron indices to compute covariance for. 
            Defaults to None which returns for all neurons.

            trial_index (| list[int] | np.ndarray | None, optional):
            Indices of trials to include in the computation Need trials > 2 (need atleast 2 to calculate cov) else does outer product
            Defaults to None which implies to take accross all trials

            time_list (list[int] | np.ndarray | None, optional):
            Time indices to include in the computation.
            Defaults to None: which implies to take accross all time steps
            
            normalized (bool, optional): If True, returns the correlation matrix instead of covariance,. Defaults to False.

        Returns:
            dict[int, np.ndarray]: {neuron_idx: matrix of according shape dependend on time_list}
        """
        default_trials, default_times, _ = self.default_trial_time_neuron_list()

        if trial_index is None:
            trial_index = default_trials

        if time_list is None:
            time_list = default_times
        
        trial_index = np.asarray(trial_index)
        time_list = np.asarray(time_list)

        def compute_cov(neuron: int) -> np.ndarray:
            # Shape: (n_trials, n_times)
            
            data = self.observation[trial_index][:, time_list, neuron]

            if data.shape[0] < 2:
                print('Calculating Outer product')
                x = self.observation[trial_index, time_list, neuron]
                x = x - x.mean()

                return np.outer(x, x)

            cov = np.cov(data, rowvar=False)

            if normalized:
                std = np.sqrt(np.diag(cov))
                denom = np.outer(std, std)
                cov = cov / (denom + 1e-12)

            return cov

        # Single neuron
        if isinstance(neuron_index, int):
            return {neuron_index: compute_cov(neuron_index)}

        # Multiple neurons
        if neuron_index is None:
            neuron_indices = range(self.neuron_cnt)
        else:
            neuron_indices = np.asarray(neuron_index)

        return {int(neuron): compute_cov(int(neuron)) for neuron in neuron_indices}

        
    def get_autocorrelation_function(self, neuron_index: int|list|None=None, trial_index: int | list[int] | np.ndarray | None = None,
                                  time_list: list[int]|np.ndarray|None=None, average_trials: bool=True, normalized = False)->dict[int, np.ndarray]:
        """
        
        Work in process to implement. Might be buggy
        Compute the temporal autocorrelation function for selected neurons.
        Assume WSS

        The autocorrelation is computed WITHIN each trial over time,
        then optionally averaged across trials.

        For a signal x(t):

            R(tau) = E[x(t) x(t + tau)]

        Parameters
        ----------
        neuron_index : list[int] | np.ndarray | None
            For single neurons 
            Neuron(s) to compute autocorrelation for.
            None => all neurons.


        trial_index : int | list[int] | np.ndarray | None
            Trials to include.
            None => all trials.

        time_list : list[int] | np.ndarray | None
            Time indices to include.
            None => all timesteps.

        normalized : bool
            If True, normalize such that:
                R(0) = 1

        average_trials : bool
            If True:
                compute autocorrelation per trial,
                then average across trials.

            If False:
                return stacked trial autocorrelations.

        Returns
        -------
        dict[int, np.ndarray]
            Dictionary mapping neuron index -> autocorrelation array

            Shape if average_trials=True:
                (n_lags,)

            Shape if average_trials=False:
                (n_trials, n_lags)

        """
        
        default_trials, default_times, _ = self.default_trial_time_neuron_list()
        if trial_index is None:
            trial_index = default_trials

        if time_list is None:
            time_list = default_times

        if isinstance(trial_index, int):
            trial_index = [trial_index]
        else:
            trial_index = np.asarray(trial_index)
        time_list = np.asarray(time_list)


        def compute_autocorr(x) -> np.ndarray:
            """
            Compute autocorrelation of 1D signal.
            """
            
            # remove mean
            x = x - np.mean(x)

            # full autocorrelation
            acorr = np.correlate(x, x, mode="full")

            # keep nonnegative lags only
            acorr = acorr[acorr.size // 2:]

            # normalize
            if normalized and acorr[0] > 1e-12:
                acorr = acorr / acorr[0]

            return acorr
        
        
        
        results = {}
        # Multiple neurons
        if neuron_index is None:
            neuron_indices = range(self.neuron_cnt)
        elif isinstance(neuron_index, int):
            neuron_indices = [neuron_index]
        else:
            neuron_indices = np.asarray(neuron_index)

        for neuron in neuron_indices:

            trial_autocorrs = []

            for trial in trial_index:

                signal = self.observation[trial][time_list, neuron]

                acorr = compute_autocorr(signal)

                trial_autocorrs.append(acorr)

            trial_autocorrs = np.asarray(trial_autocorrs)

            if average_trials:
                results[int(neuron)] = np.mean(trial_autocorrs, axis=0)
            else:
                results[int(neuron)] = trial_autocorrs

        return results

    def _get_cross_autocorrelation_function(self, neuron1, neuron2, normalized=False):
        """ Work in process to implement"""
        activity1 = self.get_trial_average(neuron1)
        activity2 = self.get_trial_average(neuron2)
        ccf = np.correlate(activity1, activity2, mode='full')
        normalizer = np.max(ccf) if normalized else 1
        return ccf/normalizer
    
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Matrix Methods----------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###


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
    
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Trial Variance Analysis-------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###
    
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
            self.get_inter_neuron_covariance_matrix(trial_idx=t) for t in range(self.trial_cnt)
        ]
        
        dissimiarity_matrix = np.zeros((self.trial_cnt, self.trial_cnt))
        for i in range(self.trial_cnt):
            for j in range(i + 1, self.trial_cnt):
                d = 1 - self.compare_cov_matrices(per_trial_covs[i], per_trial_covs[j])
                dissimiarity_matrix[i, j] = d
                dissimiarity_matrix[j, i] = d  # exploit symmetry
        
        return dissimiarity_matrix
    
    def get_trial_covariance_matrix(self):
        """Use this to udnerstand the variance between trials
        This can help validate if the trials are independent or not and form sthe base of the statistical analysis.
        """
        reshaped = self.observation.reshape(self.trial_cnt, self.timestep_cnt*self.neuron_cnt)
        return np.cov(reshaped)
    
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


    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Plotting Functions  ----------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###

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

     
    
    def plot_neural_activity_histogram(self, neuron_idx, bins=10, two_dim=True):
        """
        Visualize distribution of neural activity.

        Parameters
        ----------
        neuron_idx : int
            Index of neuron.
        bins : int
            Number of histogram bins.
        two_dim : bool
            If True, shows time vs activity distribution across all trials.
            If False, shows global activity distribution.
        """

        values = self.observation[:, :, neuron_idx]

        if two_dim:
            # flatten trials into samples of (time, activity)
            t = np.tile(np.arange(values.shape[1]), values.shape[0])
            x = values.flatten()

            plt.figure(figsize=(8, 5))

            hist = plt.hist2d(t, x, bins=bins, cmap="viridis")

            # add colorbar
            plt.colorbar(hist[3], label="Count")

            plt.xlabel("Time")
            plt.ylabel("Activity")
            plt.title(f"Neuron {neuron_idx} activity over time")

            plt.tight_layout()
            plt.show()

        else:
            plt.figure(figsize=(6, 4))

            plt.hist(values.flatten(), bins=bins)

            plt.xlabel("Activity")
            plt.ylabel("Count")
            plt.title(f"Neuron {neuron_idx} activity distribution")

            plt.tight_layout()
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
    
    
    def plot_spectrum(self,neuron_list=None,time_list=None,trial_list=None,spectrum: str = "fft",**kwargs
    ):
        """
        Plot spectral representations of neural activity.

        Args
        ----------
        neuron_list : list[int] | np.ndarray | int | None
            Neurons to analyze. If None, uses all neurons.
        time_list : list[int] | np.ndarray | None
            Time indices to include. If None, uses all timesteps.
        trial_list : list[int] | np.ndarray | int | None
            Trials to include. If None, uses all trials.
        spectrum : str
            Type of spectral analysis. One of:
            - "fft"
            - "psd"
            - "spectrogram"
        **kwargs :
            Additional plotting / computation parameters:
            - nperseg(int): segment length (default: specified in computation)
            - dt (float): sampling interval (default = 1.0)
            - log_psd (bool): log-scale PSD plot
            - show (bool): whether to call plt.show() (default True)
            - figsize (tuple): matplotlib figure size
        """

        import matplotlib.pyplot as plt

        spectrum = spectrum.lower().strip()
        valid_spectrums = ["fft", "psd", "spectrogram", "coherence"]

        if spectrum not in valid_spectrums:
            raise ValueError(f"Unknown spectrum type '{spectrum}'. Choose from {valid_spectrums}.")

        # ---------------- defaults ----------------
        dt = kwargs.get("dt", 1.0)
        show = kwargs.get("show", True)
        log_psd = kwargs.get("log_psd", False)
        figsize = kwargs.get("figsize", None)
        nperseg = kwargs.get("nperseg", None)

        neuron_list = np.arange(self.neuron_cnt) if neuron_list is None else neuron_list
        time_list = np.arange(self.timestep_cnt) if time_list is None else time_list
        trial_list = np.arange(self.trial_cnt) if trial_list is None else trial_list

        if isinstance(neuron_list, (int, np.integer)):
            neuron_list = [int(neuron_list)]
        if isinstance(trial_list, (int, np.integer)):
            trial_list = [int(trial_list)]

        # ---------------- FFT ----------------
        if spectrum == "fft":
            results = self.get_fft(neuron_list, time_list, trial_list, dt=dt)


            for neuron, (freqs, spec) in results.items():
                freqs = np.fft.fftshift(freqs)
                spectrum = np.fft.fftshift(spec)

                plt.figure(figsize=figsize)
                plt.plot(freqs, np.abs(spectrum))
                plt.title(f"Neuron {neuron} FFT Magnitude")
                plt.xlabel("Frequency")
                plt.ylabel("|FFT|")
                if show:
                    plt.show()

        # ---------------- PSD ----------------
        elif spectrum == "psd":
            results = self.get_PSD(neuron_list, time_list, trial_list, dt=dt, nperseg=nperseg)

            for neuron, (freqs, psd) in results.items():
                plt.figure(figsize=figsize)

                if log_psd:
                    plt.semilogy(freqs, psd)
                else:
                    plt.plot(freqs, psd)

                plt.title(f"Neuron {neuron} PSD")
                plt.xlabel("Frequency")
                plt.ylabel("Power")
                if show:
                    plt.show()

        # ---------------- SPECTROGRAM ----------------
        elif spectrum == "spectrogram":
            results = self.get_spectrogram(neuron_list, time_list, trial_list, dt=dt, nperseg=nperseg)

            for neuron, (f, t, Sxx) in results.items():
                plt.figure(figsize=figsize)
                plt.pcolormesh(t, f, Sxx, shading="auto")
                plt.title(f"Neuron {neuron} Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.colorbar(label="Power")
                if show:
                    plt.show()

    
    
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
        matrix = np.asarray(matrix)

        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
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

    def plot_ll(self, llrs: np.ndarray) -> None:
        """Plot negative log-likelihood evolution across EM iterations."""
        steps = np.arange(len(llrs) - 1)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(steps, llrs[1:],
                color='#3266ad', linewidth=2,
                marker='o', markersize=4,
                markerfacecolor='white', markeredgewidth=1.5)

        ax.set_title('Log-likelihood convergence', fontsize=13, fontweight='medium', pad=12)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Negative log-likelihood', fontsize=11)

        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        plt.show()

    def plot_predicted_vs_actual(self,
                                 y_actual: np.ndarray,
                                 y_predicted: np.ndarray,
                                 R_true: np.ndarray | None = None,
                                 n_neurons_to_show: int = 6,
                                 ) -> None:
        '''
        Plot predicted vs actual observations on a held-out trial.

        Shows the n_neurons with the worst per-neuron RMSE, with overall metrics
        in the figure title.

        parameters
        ----------
        y_actual : (T, m)
            Held-out observations from the true system.
        y_predicted : (T, m)
            One-step-ahead predictions under the fitted model.
        R_true : (m, m) | None
            Optional true observation noise covariance, for the noise floor reference.
        n_neurons_to_show : int
            How many of the worst-fitting neurons to display.
        title : str
            Figure suptitle.
        '''
        assert y_actual.shape == y_predicted.shape, \
            f"shape mismatch: actual {y_actual.shape} vs predicted {y_predicted.shape}"

        T, m = y_actual.shape
        n_show = min(n_neurons_to_show, m)

        # Per-neuron RMSE → pick the worst-fitting ones
        rmse_per_neuron = np.sqrt(np.mean((y_actual - y_predicted) ** 2, axis=0))
        worst = np.argsort(rmse_per_neuron)[-n_show:][::-1]

        rmse_overall = np.sqrt(np.mean((y_actual - y_predicted) ** 2))

        fig, axes = plt.subplots(
            n_show, 1,
            figsize=(8, 1.6 * n_show),
            sharex=True,
        )
        if n_show == 1:
            axes = [axes]

        time = np.arange(T)
        for ax, idx in zip(axes, worst):
            ax.plot(time, y_actual[:, idx],    color="C0", lw=1.3, label="actual")
            ax.plot(time, y_predicted[:, idx], color="C3", lw=1.0, ls="--", label="predicted")
            ax.set_ylabel(f"y[{idx}]")
            ax.text(
                0.99, 0.95,
                f"RMSE = {rmse_per_neuron[idx]:.3f}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc="upper left", fontsize=9)
        axes[-1].set_xlabel("time step")

        # Header with overall metrics
        if R_true is not None:
            floor = np.sqrt(np.trace(R_true) / R_true.shape[0])
            ratio = rmse_overall / floor
            header = (f"Held-out one-step prediction \n"
                    f"overall RMSE = {rmse_overall:.3f},  "
                    f"noise floor = {floor:.3f},  "
                    f"ratio = {ratio:.2f}")
        else:
            header = f"{"Held-out one-step prediction"}\n overall RMSE = {rmse_overall:.3f}"
        fig.suptitle(header)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_horizons_vs_RMSE(em,
                              est_params_dynamax,
                              y_actual,
                              u_trial,
                              label:str,
                              horizons:np.ndarray = np.asarray([1,2,5,10,20,30]),
                              )->None:
        '''
        plots RMSE against reconstruction square error for varying prediction horizon
        
        
        '''
        rmses = []
        for k in horizons:
            y_pred, y_act = Simulator.multi_step_predict(em,est_params_dynamax,y_actual,u_trial, k)
            rmses.append(np.sqrt(np.mean((y_act - y_pred)**2)))

        plt.plot(horizons, rmses, 'o-',label = label)
        plt.xlabel('prediction horizon k')
        plt.ylabel('RMSE')
        plt.legend()
        plt.yscale('log')




    @staticmethod
    def plot_frequency_response(sim_sim, est_sim) -> None:
        '''
        Compare frequency responses of true vs estimated system.

        Plots magnitude (dB) for each input-output pair in a grid.
        '''
        if sim_sim.has_default_matrices or est_sim.has_default_matrices:
            raise ValueError(' at least one of the objects has default matices')
            return

        omega, H_sim = sim_sim.calculate_transfer_function(num_points=200)
        _,     H_est = est_sim.calculate_transfer_function(num_points=200)

        y_dim, u_dim = H_sim.shape[1], H_sim.shape[2]
        fig, axes = plt.subplots(y_dim, u_dim,
                                figsize=(3 * u_dim, 2 * y_dim),
                                sharex=True, squeeze=False)

        for i in range(y_dim):
            for j in range(u_dim):
                mag_sim = 20 * np.log10(np.abs(H_sim[:, i, j]) + 1e-12)
                mag_est = 20 * np.log10(np.abs(H_est[:, i, j]) + 1e-12)

                ax = axes[i, j]
                ax.plot(omega, mag_sim, "C0-",  lw=1.2, label="true")
                ax.plot(omega, mag_est, "C3--", lw=1.0, label="estimated")
                ax.grid(True, alpha=0.3)

                if i == 0:
                    ax.set_title(f"input {j}")
                if j == 0:
                    ax.set_ylabel(f"y[{i}] (dB)")
                if i == y_dim - 1:
                    ax.set_xlabel("ω (rad/sample)")

        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle("Frequency response: true vs estimated", y=1.01)
        plt.tight_layout()
        plt.show()


    def plot_eigenvalues(self,
                         eig_true:np.ndarray,
                         eig_est:np.ndarray,
                         )->None:
        

        '''
        use to compare the eigenvalues of the dynamics matric A

        parameters
        -----------
        eig_true:np.ndarray
            eigenvalues of true simulated system
        eig_est:np.ndarray
            estimated eignevalues from data
        '''
        fig, ax = plt.subplots(figsize=(5, 5))

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.6,
                label="unit circle")

        # Axes through origin
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)

        # Eigenvalues
        ax.scatter(eig_true.real, eig_true.imag,
                s=90, facecolors="none", edgecolors="C0", lw=2, label="true")
        ax.scatter(eig_est.real, eig_est.imag,
                s=90, marker="x", color="C3", lw=2, label="estimated")

        # Cosmetics
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")              # circle must be circular
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title("Argand diagram: estimated vs true eigenvalues")
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_all_ssm_matrices(self,simulated_params:LDSParams,estimated_params:LDSParams)->None:
        '''
        Side-by-side heatmap comparison of true vs estimated LGSSM matrices.

        Columns: true | estimated | difference (estimated - true).
        Rows: A, B, C, Q, R.

        Note: direct element-wise comparison is sensitive to the similarity-transform
        ambiguity of LGSSM identification. A non-zero difference does not imply a
        bad fit — compare eigenvalues of A and predictive likelihood for that.

        parameters
        ----------
        simulated_params : LDSParams
            Ground-truth parameters used to generate the data.
        estimated_params : LDSParams
            Parameters recovered by the identifier (EM, PEM, N4SID, ...).
        '''
        names = ["A", "B", "C", "Q", "R"]
        true_mats = [getattr(simulated_params, n) for n in names]
        est_mats  = [getattr(estimated_params, n) for n in names]

        n_rows = len(names)
        fig, axes = plt.subplots(
            n_rows, 3,
            figsize=(9, 2.2 * n_rows),
            constrained_layout=True,
        )

        col_titles = ["True", "Estimated", "Difference"]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=11)

        for i, (name, T, E) in enumerate(zip(names, true_mats, est_mats)):
            diff = E - T

            # Symmetric color limits so sign is readable.
            # Share scale between true & estimated; use diff's own scale for the diff column.
            if (T.size != 0) and (E.size != 0):
                vmax_te = max(np.abs(T).max(), np.abs(E).max())
                vmax_d  = np.abs(diff).max() if diff.size else 1.0
                vmax_d  = max(vmax_d, 1e-12)  # avoid zero range

                for ax, M, vmax in zip(axes[i], [T, E, diff], [vmax_te, vmax_te, vmax_d]):
                    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                    ax.set_xticks([]); ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                axes[i, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=15, va="center")
            else:
                for ax in axes[i]:
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.text(0.5, 0.5, f"{name}: empty",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=10, color="gray")
                axes[i, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=15, va="center")
                        

        # Numerical summary on the figure
        rho_true = float(np.max(np.abs(np.linalg.eigvals(simulated_params.A))))
        rho_est  = float(np.max(np.abs(np.linalg.eigvals(estimated_params.A))))
        fig.suptitle(
            f"LGSSM parameter comparison    |    "
            f"ρ(A_true) = {rho_true:.3f},  ρ(A_est) = {rho_est:.3f}",
            fontsize=11,
        )

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
        Y_f = H[horizon * self.neuron_cnt :, :]

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
        labels = list()
        for l in lines:
            labels.append(l.get_label())
            
        ax1.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        plt.show()

    
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Helper Functions--------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###
    
    
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
    
        



    def run_all(self, verbose: bool = True):
        """
        Run every analysis and visualization on the current observation.
        Useful for a quick full overview of the dataset.
        
        Parameters
        ----------
        verbose : bool
            If True, print section headers between calls.
        """
        def section(name):
            if verbose:
                print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        # --- Statistical summaries ---
        section("Neuron statistics")
        self.get_neuron_statistics()

        section("Trial variability report")
        self.trial_variability_report()

        section("Stationarity (ADF p-values)")
        pvals = self.test_stationaryness()
        for n, p in pvals.items():
            print(f"  neuron {n:2d}: p = {p:.4f}  {'(stationary)' if p < 0.05 else '(non-stationary)'}")

        # --- Plots: per-neuron / population summaries ---
        section("Population and trial-average plots")
        self.plot_population_average()
        self.plot_trial_average()
        self.plot_neuron_mean_activity()
        self.plot_neuron_variance()

        # --- Plots: single-neuron detail (first neuron only — full sweep would flood) ---
        section("Single-neuron detail (neuron 0)")
        self.plot_neuron_with_trial_std(neuron_id=0)
        self.plot_neural_activity_histogram(neuron_idx=0, two_dim=True)
        self.plot_neural_activity_histogram(neuron_idx=0, two_dim=False)

        # --- Plots: full-trial overview ---
        section("Trial heatmap (trial 0)")
        self.plot_heatmap(trial_id=0)

        section("All-trial all-neuron overlay")
        self.plot_general()

        # --- Spectral analyses ---
        section("Spectral analyses (neuron 0)")
        self.plot_spectrum(neuron_list=0, spectrum="fft")
        self.plot_spectrum(neuron_list=0, spectrum="psd")
        self.plot_spectrum(neuron_list=0, spectrum="spectrogram")

        # --- Covariance / correlation matrices ---
        section("Covariance and correlation matrices")
        cov = self.get_inter_neuron_covariance_matrix()
        corr = self.get_correlation_matrix()
        self.plot_matrix(cov,  title="Inter-neuron covariance",  show_numbers=False)
        self.plot_matrix(corr, title="Inter-neuron correlation", show_numbers=False)

        section("Trial covariance and dissimilarity matrices")
        self.plot_matrix(self.get_trial_covariance_matrix(),
                        title="Trial-trial covariance", show_numbers=False)
        self.plot_matrix(self.get_dissimilarity_matrix(),
                        title="Trial dissimilarity (1 - corr)", show_numbers=True, cmap="viridis")

        # --- Dimensionality reduction ---
        section("PCA / Scree")
        explained, cumulative, k, _, _ = self.get_PCA()
        print(f"  components for 90% variance: {k}")
        print(f"  cumulative variance (first 5): {cumulative[:5]}")
        try:
            self.plot_scree()
        except ValueError as e:
            print(f"  scree skipped: {e}")

        if verbose:
            print(f"\n{'=' * 60}\nrun_all complete\n{'=' * 60}")


    ####

        

            