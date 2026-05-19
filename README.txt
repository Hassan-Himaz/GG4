=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
THE PURPOSE OF THIS DOCUMENT IS TO EXPLAIN THE STRUCTURE OF OUR CODEBASE
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
THE ILLUSTRATOR CLASS
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
The Illustrator class is meant to find a myriad of statistics on a given
dataset and contains all plotting functionalities.

Input data is a 3D NumPy array of shape:
    (Trials, Timepoints, Neurons)

Example: data.shape = (5, 60, 16)


ILLUSTRATOR CLASS ATTRIBUTES
 - observation        -> numpy array of seen data (trials, timepoints, neurons)
                         Exposed via @property; setter updates the cnt values.
 - trial_cnt          -> number of trials                (read-only property)
 - timestep_cnt       -> number of timepoints per trial  (read-only property)
 - neuron_cnt         -> number of neurons               (read-only property)

The underlying storage uses protected attributes (_observation, _trial_cnt, etc.)
with properties for controlled access — i.e., encapsulation.


ILLUSTRATOR METHODS

--- Statistics methods ---
 - get_trial_averaged_mean(chosen_neurons=None)
       Returns the trial-averaged time course for the chosen neurons.
       Accepts int, list, np.ndarray, or None (defaults to all neurons).
       Output shape: (timesteps, neurons_selected).
 
 - get_covariance_matrix()
       Returns the (neurons x neurons) covariance matrix, pooling trials
       and timepoints together. Captures co-activation across neurons.
 
 - get_autocovariance_matrix(neuron_index)
       Returns the (timesteps x timesteps) autocovariance matrix for one
       neuron, treating each trial as one sample. Captures temporal
       structure of a single neural process.
 
 - get_trial_covariance_matrix()
       Returns the (trials x trials) covariance matrix, treating each trial
       as a flattened vector. Useful for checking trial independence.

--- Summary report methods ---
 - summary()
       Prints dataset shape and overall mean / std / min / max.
 
 - neuron_statistics()
       Prints and returns a dict with per-neuron mean, std, variance,
       peak value (across trial-averaged time course), and peak time.
 
 - trial_variability_report()
       Prints and returns mean and max trial-to-trial standard deviation
       for each neuron. Low values mean trials are nearly identical.

--- Plotting methods ---
 - plot_scatter(neuron_list=None, time_list=None, trial_list=None)
       Generalised scatter plot of selected neurons, timesteps, and trials.
       Defaults to plotting everything.
 
 - plot_trial_average()
       Plots the trial-averaged time course for every neuron.
 
 - plot_population_average()
       Plots the average activity across all neurons and trials.
 
 - plot_heatmap(trial_id=0)
       Heatmap of one trial: rows = time, columns = neurons.
 
 - plot_neuron_variance()
       Bar chart of variance per neuron across trials and time.
 
 - plot_neuron_mean_activity()
       Bar chart of mean activity per neuron.
 
 - plot_neuron_with_trial_std(neuron_id=0)
       Trial-averaged trace of one neuron with ±1 std trial-variability band.
 
 - plot_correlation_matrix()
       Heatmap of the (neurons x neurons) correlation matrix.
 
 - pca_energy()
       Prints and plots cumulative variance explained by PCA components.
       Useful for guessing latent state dimensionality.
 
 - plot_matrix(matrix, title, color_coded, show_numbers, cmap)
       General-purpose heatmap visualisation for any 2D matrix.
       Supports overlaying numeric values with automatic text contrast.
 
 - plot_scree(horizon=4)
       Subspace-identification scree plot built from a block-Hankel
       projection. Shows individual singular values and cumulative variance
       on dual axes — used to pick latent dim via the "elbow" method.



=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
THE SIMULATOR CLASS
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
The Simulator class holds an LDS parameter set (A, B, C, Q, R) and:
  1 - generates synthetic data from those parameters, and
  2 - provides estimation methods to recover system matrices from data.

The state-space model is:
    x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
    y_t     = C x_t + v_t,            v_t ~ N(0, R)


SIMULATOR CLASS ATTRIBUTES
 - A             -> hidden state transition matrix          (d x d)
 - B             -> control / input matrix                  (d x m)
 - C             -> observation matrix                      (n x d)
 - Q             -> process noise covariance                (d x d)
 - R             -> observation noise covariance            (n x n)
 - x_dimensions  -> hidden state dimension d (from Q)
 - y_dimensions  -> observed dimension n (from R)


SIMULATOR METHODS

--- Data generation ---
 - generate_data(trials, lengths, seed, control_function)
       Generates a synthetic dataset using the stored A, B, C, Q, R.
       Returns array of shape (trials, lengths+1, n).
       control_function : callable f(time, data_history) → input vector u_t.
 
 - __generate_trial(length, seed, control_function)
       (Private) Generates one trial by rolling the LDS forward, adding
       process noise on the latent state and observation noise on y.

--- Estimation methods ---
 - estimate_ssid_matrices(state_dim, horizon=4)
       Subspace System Identification (N4SID-style).
       Builds a block-Hankel matrix, projects future-on-past, takes the
       SVD, and reads A and C from the truncated subspace.
       Returns (A, C) only; Q, R, B are not estimated here.
       horizon controls how far past/future the Hankel blocks extend.
 
 - estimate_system_matrices(latent_dim=2, n_iter=10, control_function=None)
       Full LDS parameter estimation via EM (pykalman) for (A, C, Q, R),
       optionally followed by Ridge regression to recover B if a control
       function is supplied.
       Returns a dict containing A, B, C, Q, R.
       Requires: pykalman, scikit-learn.
 
 - estimate_var_matrix(lag=1)
       Vector-autoregression estimate of the transition matrix at the
       observation level (skipping latent states):
           y_{t+lag} = W y_t + noise
       Returns W of shape (n x n). Useful as a quick baseline.


