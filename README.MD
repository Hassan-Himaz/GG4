# GG4 Codebase — Illustrator & Simulator

## Overview

Two core modules form the shared Week 1 foundation:

| File | Purpose |
|------|---------|
| `Illustrator_arnav.py` | Statistical analysis and visualisation of observation data |
| `Simulator.py` | LDS simulation with a live interactive parameter explorer |

Supporting files: `LDSParams.py` (parameter dataclass), `Controllers.py` (input function factory).

---

## Illustrator (`Illustrator_arnav.py`)

### What it does

`Illustrator` wraps a 3-D NumPy array of shape `(Trials, Timepoints, Neurons)` and exposes a suite of analysis and plotting methods. All methods work on any sub-selection of neurons, trials, and time windows — pass `None` to use all.

### Initialisation

```python
from Illustrator_arnav import Illustrator
import numpy as np

data = np.load("ExampleDataset.npy")   # shape: (Trials, Timepoints, Neurons)
ill = Illustrator(data)
```

The constructor validates that the array is 3-D and stores it as a protected attribute. The `observation` property has a setter that re-validates before replacing the data.

Key read-only properties: `ill.trial_cnt`, `ill.timestep_cnt`, `ill.neuron_cnt`.

---

### Statistical Analysis

| Method | Returns | Notes |
|--------|---------|-------|
| `get_trial_average(chosen_neurons, chosen_trials)` | `(T, N)` array | Trial-averaged signal; accepts int, list, or `None` |
| `get_mean_per_neuron()` | `(N,)` | Mean over all trials and time |
| `get_std_per_neuron()` | `(N,)` | Std over all trials and time |
| `get_variance_per_neuron()` | `(N,)` | Variance over all trials and time |
| `get_signal_energy()` | `(N,)` | E[X²] per neuron |
| `get_trial_averaged_peak_value_time()` | `(peak_values, peak_times)` | Peak of the trial-averaged trace |
| `get_response_slopes()` | `(T, N)` | Savitzky–Golay derivative of the trial-average |
| `get_trial_variance_per_neuron()` | `(T, N)` | Variance across trials at each time step |
| `get_neuron_statistics()` | `dict` | Prints and returns mean, std, energy, variance, peak, peak time, peak slope per neuron |

---

### Frequency-Domain Analysis

All frequency methods accept optional `neuron_list`, `time_list`, `trial_list`, `dt`, and `average_across_trials`.  
`average_across_trials=True` computes the *evoked* spectrum (average first, then transform); `False` averages individual-trial spectra.

| Method | Returns |
|--------|---------|
| `get_fft(...)` | `{neuron: (freqs, complex spectrum)}` |
| `get_PSD(...)` | `{neuron: (freqs, power)}` via Welch's method |
| `get_spectrogram(...)` | `{neuron: (freqs, times, Sxx)}` |

---

### Connectivity & Covariance

| Method | Returns | Notes |
|--------|---------|-------|
| `get_inter_neuron_covariance_matrix(trial_idx)` | `(N, N)` | Pass `trial_idx=None` for all trials |
| `get_correlation_matrix()` | `(N, N)` | Pearson correlation across all trials/times |
| `get_time_covariance_matrix(neuron_index, ...)` | `dict {neuron: (T, T)}` | Time–time structure per neuron |
| `get_autocorrelation_function(...)` | `dict {neuron: acf}` | Within-trial ACF, optionally averaged |
| `compare_cov_matrices(cov1, cov2)` | `float` | Correlation of upper-triangle elements |
| `get_dissimilarity_matrix()` | `(Trials, Trials)` | Pairwise trial dissimilarity via covariance comparison |
| `get_trial_covariance_matrix()` | `(Trials, Trials)` | Trial-level covariance (flattened signals) |
| `trial_variability_report()` | prints + arrays | Mean and max trial-to-trial std per neuron |

---

### Dimensionality Reduction

| Method | Returns |
|--------|---------|
| `get_SVD()` | `(U, S, Vh)` from centred data |
| `get_PCA(SVD, threshold)` | `(explained_var, cumulative_var, n_components, principal_vectors, scores)` |

---

### Stationarity

```python
pvals = ill.test_stationaryness()   # {neuron_idx: ADF p-value}
# p < 0.05  =>  reject non-stationarity null  =>  signal is likely stationary
```

Uses the Augmented Dickey–Fuller test on the trial-averaged signal per neuron.

---

### Plotting

| Method | What it shows |
|--------|---------------|
| `plot_general(neuron_list, time_list, trial_list)` | Raw traces for selected neurons/trials |
| `plot_trial_average()` | Trial-averaged time course, all neurons |
| `plot_population_average()` | Mean across all neurons and trials over time |
| `plot_heatmap(trial_id)` | Time × Neuron activation heatmap for one trial |
| `plot_neuron_with_trial_std(neuron_id)` | Mean ± 1 trial std for one neuron |
| `plot_neuron_variance()` | Bar chart of per-neuron variance |
| `plot_neuron_mean_activity()` | Bar chart of per-neuron mean activity |
| `plot_neural_activity_histogram(neuron_idx, bins, two_dim)` | 1-D or 2-D (time × activity) histogram |
| `plot_spectrum(..., spectrum)` | FFT / PSD / spectrogram — pass `spectrum="fft"`, `"psd"`, or `"spectrogram"` |
| `plot_matrix(matrix, title, ...)` | Colour-coded matrix display with optional numeric cell labels |
| `plot_scree(horizon)` | SSID subspace scree plot: singular values + cumulative variance |

---

## Simulator (`Simulator.py`)

### What it does

`Simulator` implements a **Linear Dynamical System (LDS)**:

```
x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
y_t     = C x_t + o_t,            o_t ~ N(0, R)
```

It wraps an `Illustrator` (for real-data comparison) and a `Controllers` (for input functions). The main feature is `explore()` — a live tkinter GUI for real-time parameter tweaking.

### Initialisation

```python
import numpy as np
from Illustrator_arnav import Illustrator
from Simulator import Simulator
from LDSParams import LDSParams
from Controllers import Controllers

data = np.load("ExampleDataset.npy")   # shape: (5, 60, 16)
ill  = Illustrator(data)
ctrl = Controllers()

# Example: 4 hidden states, 2 inputs, 16 observed neurons
params = LDSParams(
    A    = 0.9 * np.eye(4),
    B    = np.zeros((4, 2)),
    C    = np.random.randn(16, 4) * 0.1,
    Q    = 0.1 * np.eye(4),
    R    = 0.5 * np.eye(16),
    mu_0 = np.zeros(4),
    P_0  = np.eye(4),
)

sim = Simulator(params, ill, ctrl)
```

`LDSParams` can also be constructed from:
- a legacy tuple: `LDSParams.from_tuple((A, B, C, Q, R, mu_0, P_0))`
- fitted dynamax parameters: `LDSParams.from_dynamax(params)`

### Core methods

| Method | Description |
|--------|-------------|
| `generate_data(trials, lengths, seed, input_function)` | Batch-generate `(Trials, T, N)` array |
| `_generate_trial(length, seed, input_function, return_state)` | Single trial; `return_state=True` returns `(obs, states)` |
| `compare_plot(seed)` | Side-by-side real vs simulated trial plot |
| `finite_controllability_gramian(horizon)` | W_c over a finite horizon |
| `finite_observability_gramian(horizon)` | W_o over a finite horizon |
| `gramian_summary(horizon)` | Dict of both gramians + eigenvalues |
| `plot_gramians(horizon)` | Side-by-side heatmaps of W_c and W_o |
| `explore()` | **Live interactive GUI** (see below) |

`seed` is always a tuple `(mu_0, P_0)`.

---

### Controllers

`Controllers` is a factory for input callables with signature `f(time: int, data: list) -> np.ndarray`:

```python
ctrl = Controllers()
pulse = ctrl.make_pulse(t_on=5, t_off=20, amplitude=1.0, num_inputs=2)
ramp  = ctrl.make_ramp(t_on=0, t_off=30, slope=0.1,     num_inputs=2)
sine  = ctrl.make_sine(freq=0.05, amplitude=1.0,          num_inputs=2)
zero  = ctrl.make_zero(num_inputs=2)
```

---

### `explore()` — Interactive GUI

Call `sim.explore()` to open a multi-panel tkinter window:

```
Left panel     — editable matrix entry grids for A, B, C, Q, R, mu_0, P_0
Centre canvas  — live matplotlib plot: real (solid) vs simulated (dashed), coloured per neuron
Right panel    — eigenvalue / singular-value spectra for each matrix
Plot controls  — per-neuron visibility toggles for real and simulated traces independently
               — input and latent-state channel toggles
```

**Buttons:**

| Button | Action |
|--------|--------|
| Update (or press Return in any field) | Regenerate simulation with current parameters |
| Show inputs | Pop-up plot of u(t) for the active input type |
| Show states | Pop-up plot of latent states x(t) |
| Save trial to CSV | Write real + simulated data + parameters to `trial_dumps/` |
| Plot Gramians | Plot controllability and observability gramians |
| Open comparison window | Run any `Illustrator` method on both real and latest saved simulation |

**Input types (selectable in GUI):** pulse, ramp, sine, zero — each with configurable parameters (t_on, t_off, amplitude, etc.).

**Dimension spinboxes:** changing `x_dim` or `u_dim` automatically resizes A, B, C, Q, mu_0, P_0 while preserving existing values.

---

### CSV dumps

`Save trial to CSV` writes a timestamped file to `trial_dumps/`. Each file contains:
- comment-header lines with all matrix values and input configuration,
- columns: `t`, `real_0 … real_N`, `sim_0 … sim_N`, `input_0 … input_K`.

Load the latest dump programmatically:

```python
params_meta, df = Simulator.load_latest_dump("trial_dumps")
```

---

## Dependencies

```
numpy
scipy
matplotlib
statsmodels
```

Install with:

```bash
pip install -r requirements.txt
```
