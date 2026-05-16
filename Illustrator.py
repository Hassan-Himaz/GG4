import numpy as np
import matplotlib.pyplot as plt


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

    __slots__ = ["observation", "trial_cnt", "timestep_cnt", "neuron_cnt"]

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

        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape

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

    def plot_neuron(self, neuron_id: int = 0):
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
            plt.plot(
                time,
                self.observation[trial_id, :, neuron_id],
                label=f"trial {trial_id}",
                alpha=0.8,
            )

        plt.xlabel("Time step")
        plt.ylabel("Observed activity")
        plt.title(f"Neuron {neuron_id} across all trials")
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

        stats = {
            "mean": mean_per_neuron,
            "std": std_per_neuron,
            "variance": var_per_neuron,
            "peak_value": peak_value,
            "peak_time": peak_time,
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
    
    