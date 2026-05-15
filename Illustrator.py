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