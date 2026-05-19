import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy import signal


class Illustrator_two:
    """
    A class to visualize neural activity data of shape (Trials, Timepoints, Neurons).

    All plot methods return a (fig, ax) tuple so callers can save, further
    annotate, or embed figures as needed.  Every method is self-contained:
    pass in optional keyword arguments to override figure size, colour map, etc.

    Quick-start
    -----------
    >>> obs = np.random.randn(20, 100, 8)          # 20 trials, 100 steps, 8 neurons
    >>> ill = Illustrator(obs)
    >>> ill.plot_signals()
    >>> ill.plot_trial_average()
    >>> ill.compare_trials([0, 1, 2])
    >>> plt.show()
    """

    __slots__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt']

    def __init__(self, observation: np.ndarray):
        """
        Initialise the Illustrator with neural observation data.

        Parameters
        ----------
        observation : np.ndarray
            A 3D array of shape (Trials, Timepoints, Neurons) containing
            neural activity recordings.

        Raises
        ------
        ValueError
            If *observation* is not a 3-D array.
        """
        if observation.ndim != 3:
            raise ValueError(
                f"Expected a 3-D array (Trials, Timepoints, Neurons), "
                f"got shape {observation.shape}."
            )
        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _timesteps(self, dt: float = 1.0) -> np.ndarray:
        """Return a time axis of length *timestep_cnt* with step size *dt*."""
        return np.arange(self.timestep_cnt) * dt

    def _neuron_labels(self, neuron_ids=None):
        """Return a list of neuron label strings for the requested ids."""
        ids = range(self.neuron_cnt) if neuron_ids is None else neuron_ids
        return [f"Neuron {i}" for i in ids]

    def _trial_mean(self) -> np.ndarray:
        """Return the trial-averaged activity, shape (Timepoints, Neurons)."""
        return self.observation.mean(axis=0)

    def _trial_sem(self) -> np.ndarray:
        """Return the across-trial standard error, shape (Timepoints, Neurons)."""
        return self.observation.std(axis=0) / np.sqrt(self.trial_cnt)

    # ------------------------------------------------------------------
    # 1. Single-trial signal traces
    # ------------------------------------------------------------------

    def plot_signals(
        self,
        trial: int = 0,
        neuron_ids=None,
        dt: float = 1.0,
        figsize=None,
    ):
        """
        Plot raw activity traces for every neuron on a single trial.

        Each neuron is drawn on its own sub-axis, stacked vertically so
        temporal alignment is immediately obvious.

        Parameters
        ----------
        trial : int
            Which trial to display (default 0).
        neuron_ids : list[int] or None
            Subset of neuron indices to plot.  None plots all neurons.
        dt : float
            Sampling interval used to build the time axis (default 1.0).
        figsize : tuple or None
            Override the default figure size.

        Returns
        -------
        fig, axes : matplotlib Figure and array of Axes
        """
        ids = list(range(self.neuron_cnt)) if neuron_ids is None else list(neuron_ids)
        n = len(ids)
        t = self._timesteps(dt)
        fig, axes = plt.subplots(
            n, 1,
            figsize=figsize or (10, 1.8 * n),
            sharex=True,
        )
        if n == 1:
            axes = [axes]

        colours = plt.cm.tab10(np.linspace(0, 1, n))
        for ax, nid, colour in zip(axes, ids, colours):
            ax.plot(t, self.observation[trial, :, nid], color=colour, linewidth=1.2)
            ax.set_ylabel(f"N{nid}", rotation=0, labelpad=24, va='center')
            ax.spines[['top', 'right']].set_visible(False)

        axes[-1].set_xlabel("Time (steps)" if dt == 1.0 else "Time")
        fig.suptitle(f"Neural activity — trial {trial}", y=1.01)
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # 2. Trial-averaged traces with shaded SEM
    # ------------------------------------------------------------------

    def plot_trial_average(
        self,
        neuron_ids=None,
        dt: float = 1.0,
        figsize=None,
    ):
        """
        Plot the trial-averaged activity ± SEM for each neuron.

        The shaded band shows ±1 standard error across trials, giving a
        quick sense of trial-to-trial variability.

        Parameters
        ----------
        neuron_ids : list[int] or None
            Subset of neurons to display.
        dt : float
            Sampling interval for the time axis.
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, axes : matplotlib Figure and array of Axes
        """
        ids = list(range(self.neuron_cnt)) if neuron_ids is None else list(neuron_ids)
        n = len(ids)
        t = self._timesteps(dt)
        mean = self._trial_mean()
        sem = self._trial_sem()

        fig, axes = plt.subplots(n, 1, figsize=figsize or (10, 1.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        colours = plt.cm.tab10(np.linspace(0, 1, n))
        for ax, nid, colour in zip(axes, ids, colours):
            ax.plot(t, mean[:, nid], color=colour, linewidth=1.5, label="mean")
            ax.fill_between(
                t,
                mean[:, nid] - sem[:, nid],
                mean[:, nid] + sem[:, nid],
                alpha=0.25,
                color=colour,
                label="±SEM",
            )
            ax.set_ylabel(f"N{nid}", rotation=0, labelpad=24, va='center')
            ax.spines[['top', 'right']].set_visible(False)

        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time (steps)" if dt == 1.0 else "Time")
        fig.suptitle(f"Trial-averaged activity (n={self.trial_cnt})", y=1.01)
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # 3. Activity heatmap: neurons × time for one trial
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        trial: int = 0,
        dt: float = 1.0,
        cmap: str = "RdBu_r",
        figsize=None,
    ):
        """
        Display a neurons × time heatmap of activity for a single trial.

        Diverging colours (red = high, blue = low) make it easy to spot
        which neurons are strongly active at each moment.

        Parameters
        ----------
        trial : int
            Trial index to display.
        dt : float
            Sampling interval.
        cmap : str
            Matplotlib colourmap name.
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        data = self.observation[trial].T   # shape (Neurons, Timepoints)
        vmax = np.abs(data).max()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=figsize or (10, 0.5 * self.neuron_cnt + 1))
        im = ax.imshow(
            data, aspect='auto', cmap=cmap, norm=norm,
            extent=[0, self.timestep_cnt * dt, self.neuron_cnt - 0.5, -0.5],
        )
        ax.set_yticks(range(self.neuron_cnt))
        ax.set_yticklabels([f"N{i}" for i in range(self.neuron_cnt)], fontsize=8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Neuron")
        fig.colorbar(im, ax=ax, label="Activity")
        ax.set_title(f"Activity heatmap — trial {trial}")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 4. Compare selected trials for one neuron
    # ------------------------------------------------------------------

    def compare_trials(
        self,
        trial_ids,
        neuron: int = 0,
        dt: float = 1.0,
        figsize=None,
    ):
        """
        Overlay activity traces from multiple trials for a single neuron.

        Useful for assessing trial-to-trial variability and checking
        whether the system response is consistent across repetitions.

        Parameters
        ----------
        trial_ids : list[int]
            Trials to overlay.
        neuron : int
            Which neuron to inspect.
        dt : float
            Sampling interval.
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        t = self._timesteps(dt)
        fig, ax = plt.subplots(figsize=figsize or (10, 4))
        colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(trial_ids)))

        for tid, colour in zip(trial_ids, colours):
            ax.plot(t, self.observation[tid, :, neuron],
                    color=colour, linewidth=1, alpha=0.8, label=f"Trial {tid}")

        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        ax.set_title(f"Trial comparison — Neuron {neuron}")
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 5. Cross-neuron correlation matrix
    # ------------------------------------------------------------------

    def plot_correlation_matrix(
        self,
        trial: int = None,
        cmap: str = "coolwarm",
        figsize=None,
    ):
        """
        Plot the Pearson correlation matrix across neurons.

        If *trial* is None the trial-averaged activity is used; otherwise
        the correlation is computed on the specified trial alone.

        Strong off-diagonal entries reveal which neurons are co-active —
        key for understanding network coupling and identifying redundant
        channels.

        Parameters
        ----------
        trial : int or None
            Trial to use.  None uses the trial average.
        cmap : str
            Colourmap (should be diverging, centred at zero).
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        data = self._trial_mean() if trial is None else self.observation[trial]
        # data shape: (Timepoints, Neurons)
        corr = np.corrcoef(data.T)   # (Neurons, Neurons)

        fig, ax = plt.subplots(figsize=figsize or (6, 5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks(range(self.neuron_cnt))
        ax.set_yticks(range(self.neuron_cnt))
        ax.set_xticklabels([f"N{i}" for i in range(self.neuron_cnt)], fontsize=8)
        ax.set_yticklabels([f"N{i}" for i in range(self.neuron_cnt)], fontsize=8)
        fig.colorbar(im, ax=ax, label="Pearson r")
        title = "Neuron correlation (trial average)" if trial is None \
            else f"Neuron correlation — trial {trial}"
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 6. Power spectra across neurons
    # ------------------------------------------------------------------

    def plot_power_spectra(
        self,
        trial: int = 0,
        fs: float = 1.0,
        neuron_ids=None,
        figsize=None,
    ):
        """
        Plot the power spectral density (Welch's method) for each neuron.

        Reveals whether any neurons or input patterns carry oscillatory
        structure, and at what frequencies.

        Parameters
        ----------
        trial : int
            Trial to analyse.
        fs : float
            Sampling frequency in Hz (default 1.0 — i.e. normalised).
        neuron_ids : list[int] or None
            Subset of neurons.  None plots all.
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        ids = list(range(self.neuron_cnt)) if neuron_ids is None else list(neuron_ids)
        colours = plt.cm.tab10(np.linspace(0, 1, len(ids)))

        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        for nid, colour in zip(ids, colours):
            freqs, psd = signal.welch(self.observation[trial, :, nid], fs=fs)
            ax.semilogy(freqs, psd, color=colour, linewidth=1.2, label=f"N{nid}")

        ax.set_xlabel("Frequency" + (" (Hz)" if fs != 1.0 else " (normalised)"))
        ax.set_ylabel("PSD")
        ax.set_title(f"Power spectral density — trial {trial}")
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 7. Variance explained per neuron across trials
    # ------------------------------------------------------------------

    def plot_neuron_variability(self, figsize=None):
        """
        Bar chart of across-trial variance for each neuron.

        Higher variance suggests that a neuron is strongly driven by
        whatever varies across trials (e.g. different input patterns),
        making it a more informative readout channel.

        Parameters
        ----------
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        # Variance across trials then averaged over time  → (Neurons,)
        var_per_neuron = self.observation.var(axis=0).mean(axis=0)

        fig, ax = plt.subplots(figsize=figsize or (8, 3))
        colours = plt.cm.plasma(var_per_neuron / var_per_neuron.max())
        ax.bar(range(self.neuron_cnt), var_per_neuron, color=colours)
        ax.set_xticks(range(self.neuron_cnt))
        ax.set_xticklabels([f"N{i}" for i in range(self.neuron_cnt)])
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Mean trial variance")
        ax.set_title("Across-trial variability per neuron")
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 8. Phase-space (state-space) trajectory
    # ------------------------------------------------------------------

    def plot_state_trajectory(
        self,
        trial: int = 0,
        dim_x: int = 0,
        dim_y: int = 1,
        figsize=None,
    ):
        """
        Plot the 2-D trajectory of the neural state across time.

        The start is marked with a circle and the end with a star so you
        can track the direction of travel.  Useful for spotting fixed
        points, limit cycles, or transient trajectories.

        Parameters
        ----------
        trial : int
            Trial index to display.
        dim_x, dim_y : int
            Which neuron dimensions to use as the two axes.
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        x = self.observation[trial, :, dim_x]
        y = self.observation[trial, :, dim_y]
        t_norm = np.linspace(0, 1, self.timestep_cnt)

        fig, ax = plt.subplots(figsize=figsize or (5, 5))
        points = ax.scatter(x, y, c=t_norm, cmap='viridis', s=8, zorder=3)
        ax.plot(x, y, linewidth=0.6, alpha=0.4, color='gray')
        ax.scatter(*[x[0], y[0]], marker='o', s=80, color='green',
                   zorder=5, label='start')
        ax.scatter(*[x[-1], y[-1]], marker='*', s=120, color='red',
                   zorder=5, label='end')
        fig.colorbar(points, ax=ax, label="Normalised time")
        ax.set_xlabel(f"Neuron {dim_x}")
        ax.set_ylabel(f"Neuron {dim_y}")
        ax.set_title(f"State trajectory — trial {trial}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 9. Compare responses to different input conditions
    # ------------------------------------------------------------------

    def compare_conditions(
        self,
        condition_dict: dict,
        neuron: int = 0,
        dt: float = 1.0,
        figsize=None,
    ):
        """
        Overlay trial-averaged responses from different input conditions.

        Parameters
        ----------
        condition_dict : dict[str, np.ndarray]
            Mapping from condition label to an Illustrator-shaped array
            (Trials, Timepoints, Neurons).  The current instance is *not*
            automatically included — add it explicitly if desired.
        neuron : int
            Which neuron to compare across conditions.
        dt : float
            Sampling interval.
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes

        Example
        -------
        >>> ill.compare_conditions(
        ...     {"pulse": pulse_obs, "oscillatory": osc_obs, "random": rand_obs},
        ...     neuron=2,
        ... )
        """
        t = self._timesteps(dt)
        fig, ax = plt.subplots(figsize=figsize or (10, 4))
        colours = plt.cm.Set1(np.linspace(0, 0.9, len(condition_dict)))

        for (label, obs), colour in zip(condition_dict.items(), colours):
            mean = obs.mean(axis=0)[:, neuron]
            sem = obs.std(axis=0)[:, neuron] / np.sqrt(obs.shape[0])
            ax.plot(t, mean, color=colour, linewidth=1.5, label=label)
            ax.fill_between(t, mean - sem, mean + sem,
                            color=colour, alpha=0.15)

        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        ax.set_title(f"Condition comparison — Neuron {neuron}")
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 10. Summary dashboard
    # ------------------------------------------------------------------

    def summary_dashboard(
        self,
        trial: int = 0,
        dt: float = 1.0,
        figsize=None,
    ):
        """
        One-figure dashboard combining the four most informative views:
        signal traces, heatmap, correlation matrix, and neuron variability.

        Designed as a first look at any new dataset — call this before
        committing to deeper analysis.

        Parameters
        ----------
        trial : int
            Trial to feature in the trace and heatmap panels.
        dt : float
            Sampling interval.
        figsize : tuple or None
            Override figure size (default adapts to neuron count).

        Returns
        -------
        fig : matplotlib Figure
        """
        fig = plt.figure(
            figsize=figsize or (14, max(8, 0.5 * self.neuron_cnt + 5))
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        t = self._timesteps(dt)
        mean = self._trial_mean()
        sem = self._trial_sem()

        # --- panel A: trace of trial average for all neurons ----
        ax_traces = fig.add_subplot(gs[0, :2])
        colours = plt.cm.tab10(np.linspace(0, 1, self.neuron_cnt))
        for nid, colour in enumerate(colours):
            ax_traces.plot(t, mean[:, nid], color=colour,
                           linewidth=1.2, label=f"N{nid}")
            ax_traces.fill_between(
                t,
                mean[:, nid] - sem[:, nid],
                mean[:, nid] + sem[:, nid],
                alpha=0.15, color=colour,
            )
        ax_traces.set_xlabel("Time")
        ax_traces.set_ylabel("Activity")
        ax_traces.set_title("Trial-averaged traces ± SEM")
        ax_traces.legend(fontsize=7, ncol=2, loc='upper right')
        ax_traces.spines[['top', 'right']].set_visible(False)

        # --- panel B: correlation matrix -------------------------
        ax_corr = fig.add_subplot(gs[0, 2])
        corr = np.corrcoef(mean.T)
        im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        ax_corr.set_xticks(range(self.neuron_cnt))
        ax_corr.set_yticks(range(self.neuron_cnt))
        ax_corr.set_xticklabels([f"N{i}" for i in range(self.neuron_cnt)],
                                 fontsize=7)
        ax_corr.set_yticklabels([f"N{i}" for i in range(self.neuron_cnt)],
                                 fontsize=7)
        fig.colorbar(im, ax=ax_corr, shrink=0.8, label="r")
        ax_corr.set_title("Neuron correlation")

        # --- panel C: heatmap (single trial) ---------------------
        ax_heat = fig.add_subplot(gs[1, :2])
        data = self.observation[trial].T
        vmax = np.abs(data).max()
        ax_heat.imshow(
            data, aspect='auto', cmap='RdBu_r',
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
            extent=[0, self.timestep_cnt * dt, self.neuron_cnt - 0.5, -0.5],
        )
        ax_heat.set_yticks(range(self.neuron_cnt))
        ax_heat.set_yticklabels([f"N{i}" for i in range(self.neuron_cnt)],
                                 fontsize=7)
        ax_heat.set_xlabel("Time")
        ax_heat.set_title(f"Activity heatmap — trial {trial}")

        # --- panel D: across-trial variability -------------------
        ax_var = fig.add_subplot(gs[1, 2])
        var_per_neuron = self.observation.var(axis=0).mean(axis=0)
        bar_colours = plt.cm.plasma(var_per_neuron / var_per_neuron.max())
        ax_var.barh(range(self.neuron_cnt), var_per_neuron, color=bar_colours)
        ax_var.set_yticks(range(self.neuron_cnt))
        ax_var.set_yticklabels([f"N{i}" for i in range(self.neuron_cnt)],
                                fontsize=7)
        ax_var.set_xlabel("Mean trial variance")
        ax_var.set_title("Neuron variability")
        ax_var.spines[['top', 'right']].set_visible(False)

        fig.suptitle(
            f"Dataset summary  —  {self.trial_cnt} trials · "
            f"{self.timestep_cnt} timesteps · {self.neuron_cnt} neurons",
            fontsize=11,
        )
        return fig