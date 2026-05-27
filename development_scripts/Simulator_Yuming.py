import numpy as np
import matplotlib.pyplot as plt


class Simulator2:
    """
    Simulator for a linear Gaussian state-space model used in GG4 Week 1.

    The model is:

        x_{t+1} = A x_t + B u_t + w_t,      w_t ~ N(0, Q)
        y_t     = C x_t + o_t,              o_t ~ N(0, R)

    where:
        x_t : hidden state vector
        u_t : input vector
        y_t : observed output vector
        A   : state transition matrix
        B   : input matrix
        C   : observation matrix
        Q   : process noise covariance
        R   : observation noise covariance

    Main purposes of this class:
        1. Simulate observations under different input patterns. like:
                random inputs
                pulses
                oscillatory inputs
                activating only one input channel at a time
                activating different channels with different amplitudes
        2. Explore how A, B, and C affect system behaviour.
        3. Analyse controllability and observability.
        4. Compute controllability and observability Gramians.
        5. Provide simple plotting tools for Week 1 intuition-building.

    Expected parameter dictionary:

        parameters = {
            "A": A,
            "B": B,
            "C": C,
            "Q": Q,          optional
            "R": R,          optional
            "x0": x0,        optional
            "seed": 0        optional
        }

    If Q, R, or x0 are not provided, simple defaults are used.

    Simulator should include analysis of: 

        controllability_matrix()
        controllability_rank()
        is_controllable()

        observability_matrix()
        observability_rank()
        is_observable()

        finite_controllability_gramian
        finite_observability_gramian

        stability_report()
        system_report()

        plot_matrix()
        plot_simulation()
        compare_input_patterns()
    """

    def __init__(self, parameters):
        """
        Initialise the simulator from a parameter dictionary.

        Parameters
        ----------
        parameters : dict
            Dictionary containing at least A, B, and C.
        """

        self.parameters = parameters

        self.A = np.asarray(parameters["A"], dtype=float)
        self.B = np.asarray(parameters["B"], dtype=float)
        self.C = np.asarray(parameters["C"], dtype=float)

        # Dimensions
        self.state_dim = self.A.shape[0]
        self.input_dim = self.B.shape[1]
        self.output_dim = self.C.shape[0]

        # Basic shape checks
        self._check_shapes()

        # Optional parameters
        self.Q = np.asarray(
            parameters.get("Q", np.zeros((self.state_dim, self.state_dim))),
            dtype=float
        )

        self.R = np.asarray(
            parameters.get("R", np.zeros((self.output_dim, self.output_dim))),
            dtype=float
        )

        self.x0 = np.asarray(
            parameters.get("x0", np.zeros(self.state_dim)),
            dtype=float
        )

        self.seed = parameters.get("seed", None)
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------

    def _check_shapes(self):
        """
        Check that A, B, and C have compatible dimensions.
        """

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square matrix with shape (state_dim, state_dim).")

        if self.B.ndim != 2:
            raise ValueError("B must be a 2D matrix with shape (state_dim, input_dim).")

        if self.C.ndim != 2:
            raise ValueError("C must be a 2D matrix with shape (output_dim, state_dim).")

        if self.B.shape[0] != self.state_dim:
            raise ValueError("B must have the same number of rows as A.")

        if self.C.shape[1] != self.state_dim:
            raise ValueError("C must have the same number of columns as A has states.")

    def _check_noise_shapes(self):
        """
        Check Q, R, and x0 dimensions.
        """

        if self.Q.shape != (self.state_dim, self.state_dim):
            raise ValueError("Q must have shape (state_dim, state_dim).")

        if self.R.shape != (self.output_dim, self.output_dim):
            raise ValueError("R must have shape (output_dim, output_dim).")

        if self.x0.shape != (self.state_dim,):
            raise ValueError("x0 must have shape (state_dim,).")

    # ------------------------------------------------------------------
    # Input generation
    # ------------------------------------------------------------------

    def generate_input(
        self,
        T,
        pattern="random",
        amplitude=1.0,
        channel=None,
        frequency=0.1,
        pulse_time=None,
        pulse_width=1,
        amplitudes=None
    ):
        """
        Generate different input patterns for simulation.

        Parameters
        ----------
        T : int
            Number of time steps.

        pattern : str
            Type of input pattern. Options:
                "zero"          : all inputs are zero
                "random"        : random Gaussian input
                "pulse"         : short pulse input
                "sine"          : sinusoidal input
                "single_channel": activate only one input channel
                "step"          : constant input after a start time
                "custom_amp"    : different constant amplitudes for different channels

        amplitude : float
            General amplitude scale.

        channel : int or None
            Which input channel to activate.
            If None, all channels are activated.

        frequency : float
            Frequency for sinusoidal input, measured in cycles per time step.

        pulse_time : int or None
            Time index where pulse starts.
            If None, the pulse is placed in the middle.

        pulse_width : int
            Width of the pulse.

        amplitudes : array-like or None
            Used for pattern="custom_amp".
            Should have length input_dim.

        Returns
        -------
        U : ndarray
            Input array with shape (T, input_dim).
        """

        U = np.zeros((T, self.input_dim))

        if pattern == "zero":
            return U

        elif pattern == "random":
            U = amplitude * self.rng.normal(size=(T, self.input_dim))

        elif pattern == "pulse":
            if pulse_time is None:
                pulse_time = T // 2

            start = max(0, pulse_time)
            end = min(T, pulse_time + pulse_width)

            if channel is None:
                U[start:end, :] = amplitude
            else:
                U[start:end, channel] = amplitude

        elif pattern == "sine":
            t = np.arange(T)
            signal = amplitude * np.sin(2 * np.pi * frequency * t)

            if channel is None:
                U[:, :] = signal[:, None]
            else:
                U[:, channel] = signal

        elif pattern == "single_channel":
            if channel is None:
                channel = 0

            U[:, channel] = amplitude

        elif pattern == "step":
            if pulse_time is None:
                pulse_time = T // 2

            if channel is None:
                U[pulse_time:, :] = amplitude
            else:
                U[pulse_time:, channel] = amplitude

        elif pattern == "custom_amp":
            if amplitudes is None:
                raise ValueError("For pattern='custom_amp', please provide amplitudes.")

            amplitudes = np.asarray(amplitudes, dtype=float)

            if amplitudes.shape != (self.input_dim,):
                raise ValueError("amplitudes must have shape (input_dim,).")

            U[:, :] = amplitudes[None, :]

        else:
            raise ValueError(
                "Unknown input pattern. Choose from: "
                "'zero', 'random', 'pulse', 'sine', "
                "'single_channel', 'step', 'custom_amp'."
            )

        return U

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, T, U=None, x0=None, process_noise=True, observation_noise=True):
        """
        Simulate the linear Gaussian state-space model.

        Parameters
        ----------
        T : int
            Number of time steps.

        U : ndarray or None
            Input sequence with shape (T, input_dim).
            If None, zero input is used.

        x0 : ndarray or None
            Initial state. If None, self.x0 is used.

        process_noise : bool
            Whether to include process noise w_t.

        observation_noise : bool
            Whether to include observation noise o_t.


        Returns
        -------
        result : dict
            Dictionary containing:
                "X" : hidden states, shape (T+1, state_dim)
                "Y" : observations, shape (T, output_dim)
                "U" : inputs, shape (T, input_dim)
        """

        self._check_noise_shapes()

        if U is None:
            U = np.zeros((T, self.input_dim))
        else:
            U = np.asarray(U, dtype=float)

        if U.shape != (T, self.input_dim):
            raise ValueError("U must have shape (T, input_dim).")

        if x0 is None:
            x0 = self.x0
        else:
            x0 = np.asarray(x0, dtype=float)

        if x0.shape != (self.state_dim,):
            raise ValueError("x0 must have shape (state_dim,).")

        X = np.zeros((T + 1, self.state_dim))
        Y = np.zeros((T, self.output_dim))

        X[0] = x0

        for t in range(T):
            # Observation equation:
            # y_t = C x_t + o_t
            if observation_noise:
                o_t = self.rng.multivariate_normal(
                    mean=np.zeros(self.output_dim),
                    cov=self.R
                )
            else:
                o_t = np.zeros(self.output_dim)

            Y[t] = self.C @ X[t] + o_t

            # State transition equation:
            # x_{t+1} = A x_t + B u_t + w_t
            if process_noise:
                w_t = self.rng.multivariate_normal(
                    mean=np.zeros(self.state_dim),
                    cov=self.Q
                )
            else:
                w_t = np.zeros(self.state_dim)

            X[t + 1] = self.A @ X[t] + self.B @ U[t] + w_t

        return {
            "X": X,
            "Y": Y,
            "U": U
        }

    # ------------------------------------------------------------------
    # Controllability
    # ------------------------------------------------------------------

    def controllability_matrix(self):
        """
        Compute the controllability matrix:

            M_c = [B, AB, A^2B, ..., A^{n-1}B]

        where n is the state dimension.

        Returns
        -------
        Mc : ndarray
            Controllability matrix with shape (state_dim, state_dim * input_dim).
        """

        blocks = []
        A_power = np.eye(self.state_dim)

        for _ in range(self.state_dim):
            blocks.append(A_power @ self.B)
            A_power = A_power @ self.A

        Mc = np.hstack(blocks)
        return Mc

    def controllability_rank(self, tol=1e-10):
        """
        Compute the rank of the controllability matrix.

        If rank == state_dim, the system is controllable.
        """

        Mc = self.controllability_matrix()
        return np.linalg.matrix_rank(Mc, tol=tol)

    def is_controllable(self, tol=1e-10):
        """
        Return True if the system is fully controllable.
        """

        return self.controllability_rank(tol=tol) == self.state_dim

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def observability_matrix(self):
        """
        Compute the observability matrix:

            M_o = [C
                   CA
                   CA^2
                   ...
                   CA^{n-1}]

        where n is the state dimension.

        Returns
        -------
        Mo : ndarray
            Observability matrix with shape (state_dim * output_dim, state_dim).
        """

        blocks = []
        A_power = np.eye(self.state_dim)

        for _ in range(self.state_dim):
            blocks.append(self.C @ A_power)
            A_power = A_power @ self.A

        Mo = np.vstack(blocks)
        return Mo

    def observability_rank(self, tol=1e-10):
        """
        Compute the rank of the observability matrix.

        If rank == state_dim, the system is observable.
        """

        Mo = self.observability_matrix()
        return np.linalg.matrix_rank(Mo, tol=tol)

    def is_observable(self, tol=1e-10):
        """
        Return True if the system is fully observable.
        """

        return self.observability_rank(tol=tol) == self.state_dim

    # ------------------------------------------------------------------
    # Gramians
    # ------------------------------------------------------------------

    def finite_controllability_gramian(self, horizon=20):
        """
        Compute the finite-horizon controllability Gramian:

            W_c = sum_{k=0}^{horizon-1} A^k B B^T (A^k)^T

        Interpretation:
            Large eigenvalues mean those state directions are easy to excite by input.
            Small eigenvalues mean those directions are hard to control.

        Parameters
        ----------
        horizon : int
            Number of time steps used in the finite sum.

        Returns
        -------
        Wc : ndarray
            Finite-horizon controllability Gramian.
        """

        Wc = np.zeros((self.state_dim, self.state_dim))
        A_power = np.eye(self.state_dim)

        for _ in range(horizon):
            Wc += A_power @ self.B @ self.B.T @ A_power.T
            A_power = A_power @ self.A

        return Wc

    def finite_observability_gramian(self, horizon=20):
        """
        Compute the finite-horizon observability Gramian:

            W_o = sum_{k=0}^{horizon-1} (A^k)^T C^T C A^k

        Interpretation:
            Large eigenvalues mean those state directions are easy to observe.
            Small eigenvalues mean those directions are hidden from observations.

        Parameters
        ----------
        horizon : int
            Number of time steps used in the finite sum.

        Returns
        -------
        Wo : ndarray
            Finite-horizon observability Gramian.
        """

        Wo = np.zeros((self.state_dim, self.state_dim))
        A_power = np.eye(self.state_dim)

        for _ in range(horizon):
            Wo += A_power.T @ self.C.T @ self.C @ A_power
            A_power = A_power @ self.A

        return Wo

    def gramian_summary(self, horizon=20):
        """
        Return eigenvalue summaries for the finite-horizon Gramians.

        This is useful because Gramian eigenvalues tell us which hidden
        state directions are easy or hard to control/observe.
        """

        Wc = self.finite_controllability_gramian(horizon=horizon)
        Wo = self.finite_observability_gramian(horizon=horizon)

        eig_Wc = np.linalg.eigvalsh(Wc)
        eig_Wo = np.linalg.eigvalsh(Wo)

        return {
            "Wc": Wc,
            "Wo": Wo,
            "eig_Wc": eig_Wc,
            "eig_Wo": eig_Wo,
            "min_eig_Wc": np.min(eig_Wc),
            "max_eig_Wc": np.max(eig_Wc),
            "min_eig_Wo": np.min(eig_Wo),
            "max_eig_Wo": np.max(eig_Wo),
        }

    # ------------------------------------------------------------------
    # Stability and system report
    # ------------------------------------------------------------------

    def eigenvalues_A(self):
        """
        Return eigenvalues of A.
        """

        return np.linalg.eigvals(self.A)

    def spectral_radius(self):
        """
        Return the largest absolute eigenvalue of A.

        For a discrete-time system:
            spectral radius < 1  means stable without input/noise.
            spectral radius > 1  means unstable or growing dynamics.
        """

        eigvals = self.eigenvalues_A()
        return np.max(np.abs(eigvals))

    def is_stable(self):
        """
        Check discrete-time stability.

        For x_{t+1} = A x_t, the system is stable if all eigenvalues
        of A lie inside the unit circle.
        """

        return self.spectral_radius() < 1.0

    def system_report(self, horizon=20, print_report=True):
        """
        Produce a summary report of the system.

        This is designed to help with Week 1 discussion:
            - Is the system stable?
            - Is it controllable?
            - Is it observable?
            - Which directions are easy/hard to control or observe?
        """

        eigvals_A = self.eigenvalues_A()
        gram = self.gramian_summary(horizon=horizon)

        report = {
            "state_dim": self.state_dim,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "eigenvalues_A": eigvals_A,
            "spectral_radius": self.spectral_radius(),
            "is_stable": self.is_stable(),
            "controllability_rank": self.controllability_rank(),
            "is_controllable": self.is_controllable(),
            "observability_rank": self.observability_rank(),
            "is_observable": self.is_observable(),
            "controllability_gramian_eigenvalues": gram["eig_Wc"],
            "observability_gramian_eigenvalues": gram["eig_Wo"],
        }

        if print_report:
            print("=" * 60)
            print("SYSTEM REPORT")
            print("=" * 60)
            print(f"State dimension:       {self.state_dim}")
            print(f"Input dimension:       {self.input_dim}")
            print(f"Output dimension:      {self.output_dim}")
            print()
            print("Eigenvalues of A:")
            print(eigvals_A)
            print(f"Spectral radius:       {report['spectral_radius']:.4f}")
            print(f"Stable?                {report['is_stable']}")
            print()
            print(f"Controllability rank:  {report['controllability_rank']} / {self.state_dim}")
            print(f"Fully controllable?    {report['is_controllable']}")
            print()
            print(f"Observability rank:    {report['observability_rank']} / {self.state_dim}")
            print(f"Fully observable?      {report['is_observable']}")
            print()
            print("Controllability Gramian eigenvalues:")
            print(gram["eig_Wc"])
            print()
            print("Observability Gramian eigenvalues:")
            print(gram["eig_Wo"])
            print("=" * 60)

        return report

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------

    def plot_matrix(self, M, title="Matrix", xlabel="", ylabel=""):
        """
        Plot a matrix as a heatmap.

        Useful for visualising A, B, C, controllability matrix,
        observability matrix, and Gramians.
        """

        plt.figure(figsize=(6, 5))
        plt.imshow(M, aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def plot_system_matrices(self):
        """
        Plot A, B, and C matrices.
        """

        self.plot_matrix(self.A, title="A: State Transition Matrix")
        self.plot_matrix(self.B, title="B: Input Matrix")
        self.plot_matrix(self.C, title="C: Observation Matrix")

    def plot_gramians(self, horizon=20):
        """
        Plot finite-horizon controllability and observability Gramians.
        """

        Wc = self.finite_controllability_gramian(horizon=horizon)
        Wo = self.finite_observability_gramian(horizon=horizon)

        self.plot_matrix(
            Wc,
            title=f"Finite Controllability Gramian, horizon={horizon}"
        )

        self.plot_matrix(
            Wo,
            title=f"Finite Observability Gramian, horizon={horizon}"
        )

    def plot_simulation(self, result, title="Simulation Result"):
        """
        Plot input U, hidden state X, and observation Y.

        Parameters
        ----------
        result : dict
            Output from self.simulate().
        """

        X = result["X"]
        Y = result["Y"]
        U = result["U"]

        T = U.shape[0]
        t_u = np.arange(T)
        t_x = np.arange(T + 1)
        t_y = np.arange(T)

        plt.figure(figsize=(10, 4))
        plt.plot(t_u, U)
        plt.title(title + " - Input U")
        plt.xlabel("Time")
        plt.ylabel("Input value")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(t_x, X)
        plt.title(title + " - Hidden State X")
        plt.xlabel("Time")
        plt.ylabel("State value")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(t_y, Y)
        plt.title(title + " - Observation Y")
        plt.xlabel("Time")
        plt.ylabel("Observation value")
        plt.tight_layout()
        plt.show()

    def compare_input_patterns(
        self,
        T=100,
        patterns=None,
        amplitude=1.0,
        process_noise=False,
        observation_noise=False
    ):
        """
        Compare how different input patterns affect observations.

        This is directly useful for the Week 1 question:
            "effects of different input patterns and system dynamics"

        Parameters
        ----------
        T : int
            Number of time steps.

        patterns : list or None
            List of input patterns to compare.

        amplitude : float
            Input amplitude.

        process_noise : bool
            Whether to include process noise.

        observation_noise : bool
            Whether to include observation noise.

        Returns
        -------
        results : dict
            Dictionary mapping pattern names to simulation results.
        """

        if patterns is None:
            patterns = ["zero", "random", "pulse", "sine", "single_channel", "step"]

        results = {}

        for pattern in patterns:
            U = self.generate_input(
                T=T,
                pattern=pattern,
                amplitude=amplitude
            )

            result = self.simulate(
                T=T,
                U=U,
                process_noise=process_noise,
                observation_noise=observation_noise
            )

            results[pattern] = result

            self.plot_simulation(
                result,
                title=f"Input pattern: {pattern}"
            )

        return results

    # ------------------------------------------------------------------
    # Helper constructors
    # ------------------------------------------------------------------

    @staticmethod
    def make_random_stable_system(
        state_dim=4,
        input_dim=2,
        output_dim=3,
        noise_scale=0.01,
        seed=0
    ):
        """
        Create a random but stable system.

        This is useful for quick testing when no real A, B, C are provided.

        The matrix A is scaled so that its spectral radius is less than 1.
        """

        rng = np.random.default_rng(seed)

        A = rng.normal(size=(state_dim, state_dim))

        # Scale A to make the system stable.
        eigvals = np.linalg.eigvals(A)
        radius = np.max(np.abs(eigvals))

        if radius > 0:
            A = 0.8 * A / radius

        B = rng.normal(size=(state_dim, input_dim))
        C = rng.normal(size=(output_dim, state_dim))

        Q = noise_scale * np.eye(state_dim)
        R = noise_scale * np.eye(output_dim)
        x0 = np.zeros(state_dim)

        parameters = {
            "A": A,
            "B": B,
            "C": C,
            "Q": Q,
            "R": R,
            "x0": x0,
            "seed": seed
        }

        return Simulator2(parameters)