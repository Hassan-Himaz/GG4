import numpy as np
import matplotlib.pyplot as plt
from Dynamax_EM_fitting import Dynamax_EM_Fitting
from dynamax.linear_gaussian_ssm import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions, LinearGaussianSSM
from Subspace_and_EM import Subspace_and_EM
from LDSParams import LDSParams
from typing import Any, Callable, Tuple
import scipy.linalg as la
from scipy.linalg import solve_discrete_lyapunov
import jax.numpy as jnp



class Simulator():
    '''
    Class to input LGSSM parameters and generates data from the model
    
    '''
    def __init__(self, 
                 num_hidden_states:int,
                 num_inputs:int,
                 num_outputs:int = 16,
                 ) -> None:
        '''
        initialises with just parameters object

        parameters
        ------------

        num_hidden_states:int
            number of desired hidden states
        num_inputs:int
            number of desired inputs
        num_outputs:int
            number of desired outputs -> should match number of neurons so default 16

        typical usage
        -------------
        sim = Simulator(num_hidden_states=5,num_inputs=5)
        params = sim.generate_general_ssm_matrices()
        sim.set_params(params)
            
        '''
        # Initialise matrix attributes at first to nonsense np arrays

        self.has_default_matrices = True
        self.A = np.zeros(2)
        self.B = np.zeros(2) 
        self.C = np.zeros(2)
        self.Q = np.zeros(2) 
        self.R = np.zeros(2)
        self.mu_0 = np.zeros(2)
        self.P_0 = np.zeros(2)


        self.x_dim = num_hidden_states
        self.y_dim = num_outputs
        self.input_dim = num_inputs


    def set_params(self,
                    model_parameters:LDSParams,):
        '''
        use to set the parameters that the simulator uses

        check is done here to check that the matrices are stable

        parameters
        -----------
         model_parameters : LDSParams
            this object must be of the LDSParams dataclass 
            contains the LGSSM matrices

        
        '''
        
        self.params = model_parameters
        

        self.A    = self.params.A
        self.B    = self.params.B
        self.C    = self.params.C
        self.Q    = self.params.Q
        self.R    = self.params.R
        self.mu_0 = self.params.mu_0
        self.P_0  = self.params.P_0
        self.has_default_matrices = False


    def _generate_trial(self,
                        inputs:np.ndarray,
                        num_timesteps:int,
                        use_initialisation_prior:bool = True,
                        ) -> tuple[np.ndarray,np.ndarray]:
        '''
        Used to generate a single trial of neural data

        defaults to 101 timesteps

        Parameters
        ----------
        use_initialisation_prior:bool
            tells model to use the initial mean and covariance matrix to generate first data point

        num_timesteps:int
            number of desired timesteps per trial



        returns 
        ----------
        tuple of 2 numpy arrays
        data - y values
        states -x/hidden state values
         
        '''

        if self.has_default_matrices:
            raise ValueError('tried to generate data, when sim parameters where never updated')

        #intial latent state
        rng = np.random.default_rng()
        latent_state = rng.multivariate_normal(self.mu_0,self.P_0)
        observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dim),self.R)
        data, states = [observed_state], [latent_state]

        for time in range(num_timesteps-1):
            latent_state = np.matmul(self.A,latent_state) + np.matmul(self.B,inputs[time])+ rng.multivariate_normal(np.zeros(self.x_dim),self.Q)
            observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dim),self.R)
            data.append(observed_state)
            states.append(latent_state)
        data = np.asarray(data)
        states = np.asarray(states)
            
        return data, states 
    

    
    def generate_dataset(self,inputs:np.ndarray,num_trials:int,num_timesteps:int,use_initialisation_prior:bool = True) -> np.ndarray:
        '''
        used to generate dataset in similar format as the neural data we are given (trial, timesteps, neuron)
        defaults to 5 trials , 101 timesteps

        parameters
        ------------
        num_trials:int
            number of desired trials

        num_timesteps:int 
            number of desired timesteps

        use_initialisation_prior:bool
            use to switch on and off using mu_0, and p_0 for your initial state  -> currently no false option coded

        must match 
        
        
        '''
        dataset = np.empty([num_trials,num_timesteps,self.y_dim])
        for i in range(0,num_trials):
            trial_outputs, _ = self._generate_trial(inputs,num_timesteps, use_initialisation_prior)
            dataset[i,:,:] = trial_outputs
        
        return dataset


    def generate_and_show(self,inputs,num_trials:int = 5,num_timesteps:int = 101,use_initialisation_prior:bool = True):
        '''
        use to generate a dataset and plot it straight away

        
        parameters
        ----------
        num_trials:int
            desired number of trials

        num_timesteps:int
            number of desired timesteps per trial

        use_initialisation_prior:bool
            tells model to use the initial mean and covariance matrix to generate first data point

        '''
        dataset = self.generate_dataset(inputs,num_trials,num_timesteps,use_initialisation_prior)
        #reshape the data to plot all trials
        outputs = dataset.reshape(-1,dataset.shape[-1]) # dataset stacked
        time = np.arange(0,len(outputs),1)
        fig, ax = plt.subplots()
        ax.plot(time, outputs)
        ax.set_xlabel("time")
        ax.set_ylabel("neuron activation")
        plt.show()



    #--------------generate data_set using kalman filter

    def kalman_filter_predicted_observations(self,
                                             em:Dynamax_EM_Fitting,
                                             real_trial:np.ndarray,
                                             )->np.ndarray:
        

        '''
        Run the Kalman filter under the estimated model and return one-step-ahead
        predicted observations.

        Parameters
        ----------
        em : dynamax_EM_Fitting (just for em.model)
        est_params_dynamax : dynamax ParamsLGSSM (the fitted params, native form)
        y_trial : (T, emission_dim) — held-out observations
        u_trial : (T, input_dim) or None — held-out inputs

        Returns
        -------
        y_pred : (T, emission_dim) — one-step-ahead predictions
        '''
       
        trial_inputs = em.inputs
        y_jax = jnp.asarray(real_trial)
        u_jax = jnp.asarray(trial_inputs)

        posterior = em.model.filter(LDSParams.to_dynamax(self.params),y_jax,u_jax)
       
            
        x_pred = np.asarray(posterior.filtered_means)      # E[x_t | y_{1:t-1}]
        C_est  = np.asarray(LDSParams.to_dynamax(self.params).emissions.weights)
        print("x_pred shape:", x_pred.shape)               # should be (T, state_dim) = (61, 5)
        print("C_est shape:", C_est.shape)                 # should be (emission_dim, state_dim) = (16, 5)
        print("y_pred shape:", (x_pred @ C_est.T).shape)   # should be (61, 16)
        return x_pred @ C_est.T
    
    @staticmethod
    def multi_step_predict(em:Dynamax_EM_Fitting, 
                           est_params_dynamax, 
                           y_trial:np.ndarray, 
                           u_trial:np.ndarray, 
                           k: int): # steps
        '''
        k-step-ahead prediction:
        Filter up to time t, then propagate k steps forward using only inputs.
        Returns y_pred[t] = predicted y at time (t+k) given data up to t.
        '''
        posterior = em.model.filter(est_params_dynamax,
                                    jnp.asarray(y_trial),
                                    jnp.asarray(u_trial))
        x_filt = np.asarray(posterior.filtered_means)         # (T, n)
        A = np.asarray(est_params_dynamax.dynamics.weights)
        B = np.asarray(est_params_dynamax.dynamics.input_weights)
        C = np.asarray(est_params_dynamax.emissions.weights)
        u = np.asarray(u_trial)

        T = len(y_trial)
        y_pred = np.zeros((T - k, y_trial.shape[1]))
        for t in range(T - k):
            x = x_filt[t]
            for step in range(k):
                x = A @ x + B @ u[t + step]
            y_pred[t] = C @ x
        return y_pred, y_trial[k:]   # aligned
    

    # get RMSE on multistep predict
    def get_RMSE_array(self,
                   em: Dynamax_EM_Fitting|Subspace_and_EM,
                   est_params_dynamax: ParamsLGSSM,
                   y_trial: np.ndarray,
                   inputs: np.ndarray,
                   k: np.ndarray,
                   ) -> np.ndarray:

        rmses = np.empty(len(k), dtype=float)
        for idx, k_val in enumerate(k):
            y_pred, y_act = Simulator.multi_step_predict(em, est_params_dynamax,
                                                        y_trial, inputs, int(k_val))
            rmses[idx] = np.sqrt(np.mean((y_act - y_pred) ** 2))
        return rmses
    

    @staticmethod
    def get_frequency_response_data(sim_sim, est_sim, num_points: int = 200):
        '''
        Compute magnitude (dB) of true vs estimated frequency response
        for every input-output pair.

        Returns
        -------
        omega   : ndarray, shape (num_points,)
            Frequency axis (rad/sample).
        mag_sim : ndarray, shape (num_points, y_dim, u_dim)
            Magnitude in dB of the true system's transfer function.
        mag_est : ndarray, shape (num_points, y_dim, u_dim)
            Magnitude in dB of the estimated system's transfer function.
        '''
        if sim_sim.has_default_matrices or est_sim.has_default_matrices:
            raise ValueError('at least one of the objects has default matrices')

        omega, H_sim = sim_sim.calculate_transfer_function(num_points=num_points)
        _,     H_est = est_sim.calculate_transfer_function(num_points=num_points)

        mag_sim = 20 * np.log10(np.abs(H_sim) + 1e-12)
        mag_est = 20 * np.log10(np.abs(H_est) + 1e-12)

        return omega, mag_sim, mag_est

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
            Controllability matrix with shape (x_dim, x_dim * input_dim).
        """

        blocks = []
        A_power = np.eye(self.x_dim)

        for _ in range(self.x_dim):
            blocks.append(A_power @ self.B)
            A_power = A_power @ self.A

        Mc = np.hstack(blocks)
        return Mc

    def controllability_rank(self, tol=1e-10):
        """
        Compute the rank of the controllability matrix.

        If rank == x_dim, the system is controllable.
        """

        Mc = self.controllability_matrix()
        return np.linalg.matrix_rank(Mc, tol=tol)

    def is_controllable(self, tol=1e-10):
        """
        Return True if the system is fully controllable.
        """

        return self.controllability_rank(tol=tol) == self.x_dim

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
            Observability matrix with shape (x_dim * y_dim, x_dim).
        """

        blocks = []
        A_power = np.eye(self.x_dim)

        for _ in range(self.x_dim):
            blocks.append(self.C @ A_power)
            A_power = A_power @ self.A

        Mo = np.vstack(blocks)
        return Mo

    def observability_rank(self, tol=1e-10):
        """
        Compute the rank of the observability matrix.

        If rank == x_dim, the system is observable.
        """

        Mo = self.observability_matrix()
        return np.linalg.matrix_rank(Mo, tol=tol)

    def is_observable(self, tol=1e-10):
        """
        Return True if the system is fully observable.
        """

        return self.observability_rank(tol=tol) == self.x_dim

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

        Wc = np.zeros((self.x_dim, self.x_dim))
        A_power = np.eye(self.x_dim)

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

        Wo = np.zeros((self.x_dim, self.x_dim))
        A_power = np.eye(self.x_dim)

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
    
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------System Analysis --------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###

    def calculate_transfer_function(self, num_points: int = 200):
        '''
        Compute the discrete-time transfer function H(z) = C (zI - A)^{-1} B
        evaluated on the unit circle z = e^{jω} for ω ∈ [0, π].

        Returns
        -------
        omega : ndarray, shape (num_points,)
            Frequencies in [0, π].
        H : ndarray, shape (num_points, y_dim, input_dim)
            Complex transfer function values.
        '''
        omega = np.linspace(0, np.pi, num_points)
        z = np.exp(1j * omega)
        I = np.eye(self.x_dim)

        H = np.zeros((num_points, self.y_dim, self.input_dim), dtype=complex)
        for i, zi in enumerate(z):
            # solve (zI - A) X = B for X, then H = C @ X
            X = np.linalg.solve(zi * I - self.A, self.B)
            H[i] = self.C @ X
        return omega, H

    def calculate_eigen(self) -> Tuple[np.ndarray,np.ndarray]:
        """
        Calculate eigen values for stablity of system
        Calculate eigen vectors for mode of system

        return
        -------
        eigenvalues, eigenvectors


        """
        # Calculate the eigenvalues of the system matrix A
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        return eigenvalues, eigenvectors
    
    def get_controlablity_observablity(self):
        n = self.A.shape[0]
        
        Ctrb = self.B
        for i in range(1, n):
            Ctrb = np.hstack((Ctrb, np.linalg.matrix_power(self.A, i) @ self.B))

        Obs = self.C
        for i in range(1, n):
            Obs = np.vstack((Obs, self.C @ np.linalg.matrix_power(self.A, i)))
        
        return Ctrb, Obs

    def get_grammian(self):
        # Calculate the controllability Grammian
        Wc = solve_discrete_lyapunov(self.A, self.B @ self.B.T)

        # Calculate the observability Grammian
        Wo = solve_discrete_lyapunov(self.A.T, self.C.T @ self.C)

        return Wc, Wo
    
    def get_hankel_singular(self, Wc=None, Wo=None):
        """
        Use this to measure importance of latent states based on energies of system mode.
        Hankel singular values, provide a measure of energy for each state in a system. 
        They are the basis for balanced model reduction, in which high energy states are retained while low energy states are discarded.
        The reduced model retains the important features of the original model. 
        """
        if not Wc or not Wo:
            Wc, Wo = self.get_grammian()
        U, S, Vh = np.linalg.svd(Wc @ Wo)
        return S

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
            "state_dim": self.x_dim,
            "input_dim": self.input_dim,
            "output_dim": self.y_dim,
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
            print(f"State dimension:       {self.x_dim}")
            print(f"Input dimension:       {self.input_dim}")
            print(f"Output dimension:      {self.y_dim}")
            print()
            print("Eigenvalues of A:")
            print(eigvals_A)
            print(f"Spectral radius:       {report['spectral_radius']:.4f}")
            print(f"Stable?                {report['is_stable']}")
            print()
            print(f"Controllability rank:  {report['controllability_rank']} / {self.x_dim}")
            print(f"Fully controllable?    {report['is_controllable']}")
            print()
            print(f"Observability rank:    {report['observability_rank']} / {self.x_dim}")
            print(f"Fully observable?      {report['is_observable']}")
            print()
            print("Controllability Gramian eigenvalues:")
            print(gram["eig_Wc"])
            print()
            print("Observability Gramian eigenvalues:")
            print(gram["eig_Wo"])
            print("=" * 60)

        return report
    


    ##---------------------------------------------
    #--------inputs factory 

    #need to be applied across input dimensions
  
    def make_pulse(self,t_on: int, t_off: int, amplitude: float, num_inputs: int) -> Callable:
        '''
        use to return callable pulse function
        
        '''
        def pulse(time: int, data: list) -> np.ndarray:
            if t_on <= time < t_off:
                return np.full(num_inputs, amplitude)
            return np.zeros(num_inputs)
        return pulse

    def make_sine(self,freq: float, amplitude: float, num_inputs: int, dt: float = 1.0):
        '''
        use to make callable sine input
        '''
        def sine(time, data):
            return np.full(num_inputs, amplitude * np.sin(2 * np.pi * freq * time * dt))
        return sine

    def make_zero(self,num_inputs: int):
        '''
        use to make callable zeros input
        '''
        return lambda time, data: np.zeros(num_inputs)
    
    def make_ramp(self, t_on: int, t_off: int, slope: float, num_inputs: int) -> Callable:
        """Linear ramp from 0 at t_on, growing by `slope` per step, held flat after t_off."""
        def ramp(time: int, data: list) -> np.ndarray:
            if time < t_on:
                return np.zeros(num_inputs)
            elapsed = min(time, t_off) - t_on
            return np.full(num_inputs, slope * elapsed)
        return ramp
    
    #-------------------------------------------------------
    #-------------array inputs----------------------------
    #-----------------------------------------------------

    # think we also may want inputs that return numpy arrays
    def make_pulse_array(self,t_on: int, t_off: int, amplitude: float, num_inputs: int,total_signal_length:int) -> np.ndarray:
            
            empty_array = np.zeros((total_signal_length,num_inputs))
            t_on = max(0,t_on)
            t_off = min(total_signal_length,t_off)
            if t_on < t_off:
                empty_array[t_on:t_off,:] = amplitude
                inputs_array = empty_array
                return inputs_array
            else:
                return empty_array
            

    def make_ramp_array(self,
                        t_on: int,
                        t_off: int,
                        slope: float,
                        num_inputs: int,
                        total_signal_length: int) -> np.ndarray:
        '''
        use to make array of ramp inputs across all inputs

        ramp input drops back to 0 after t_off
        
        '''
        u = np.zeros((total_signal_length, num_inputs))
        t_on  = max(0, t_on)
        t_off = min(total_signal_length, t_off)
        if t_on < t_off:
            steps = np.arange(t_off - t_on)            # 0, 1, 2, ...
            u[t_on:t_off, :] = (slope * steps)[:, None]  # broadcast across inputs
        return u
    
    def make_pulse_array_per_channel(self, 
                                     t_ons:np.ndarray, 
                                     t_offs:np.ndarray, 
                                     amplitudes:np.ndarray, 
                                     total_signal_length:int, 
                                     num_inputs:int,
                                     ) ->np.ndarray:
        '''
        Use to make array of specified varied pulse inputs across channels 

        parameters
        ---------

        t_ons:np.ndarray
            array of on times for each input channel
        t_offs:np.ndarray
            array of off times for each input channel
        amplitudes:np.ndarray
            array of amplitudes for each input channel
        total_signal_length:np.ndarray, 
        num_inputs:int,

        '''
        u = np.zeros((total_signal_length, num_inputs))
        for c, (t_on, t_off, amp) in enumerate(zip(t_ons, t_offs, amplitudes)):
            u[int(t_on):int(t_off), c] = amp
        return u
    
    def make_prbs_array(self,
                    amplitude: float,
                    num_inputs: int,
                    total_signal_length: int,
                    bit_hold: int = 1,
                    n_bits: int  = 8,
                    seed: int = 0) -> np.ndarray:
        '''
        use to make array of independent PRBS (maximal-length LFSR) inputs,
        one per channel -> full-rank, persistently-exciting input for ID of B.

        each channel is a +/-amplitude m-sequence; channels use independent
        LFSR seeds so they are phase-shifted copies (cross-correlation ~ -1/period).

        bit_hold : timesteps each bit is held. >1 shifts excitation energy to
                lower frequencies -- use when the dominant A-modes are slow.
        n_bits   : LFSR register length; period = 2**n_bits - 1. if None, chosen
                so the period covers the window without repeating.
        '''
        # verified maximal-length taps for THIS right-shift Fibonacci update rule
        _PRBS_TAPS = {
            2: (2, 1), 3: (3, 1), 4: (4, 1), 5: (5, 1, 2, 3), 6: (6, 1),
            7: (7, 1), 8: (8, 1, 2, 3), 9: (9, 1, 2, 5), 10: (10, 1, 2, 5),
            11: (11, 1, 2, 4), 12: (12, 1, 3, 11),
        }
        n_flips = int(np.ceil(total_signal_length / bit_hold))
        if n_bits is None:
            n_bits = max(2, int(np.ceil(np.log2(n_flips + 1))))   # period >= n_flips
        if n_bits not in _PRBS_TAPS:
            raise ValueError(f"no tap set for n_bits={n_bits}; supported {sorted(_PRBS_TAPS)}")
        taps = _PRBS_TAPS[n_bits]
        period = (1 << n_bits) - 1
        if n_flips > period:
            import warnings
            warnings.warn(f"PRBS repeats: need {n_flips} bits but period is {period}; "
                        f"raise n_bits to keep full excitation order.")

        rng = np.random.default_rng(seed)
        u = np.zeros((total_signal_length, num_inputs))
        for ch in range(num_inputs):
            state = int(rng.integers(1, 1 << n_bits))     # nonzero seed per channel
            bits = np.empty(n_flips, dtype=np.int8)
            for i in range(n_flips):
                bits[i] = state & 1                        # output LSB
                fb = 0
                for t in taps:
                    fb ^= (state >> (t - 1)) & 1           # XOR tapped bits
                state = (state >> 1) | (fb << (n_bits - 1))
            seq = np.repeat(bits, bit_hold)[:total_signal_length]   # bit-hold
            u[:, ch] = amplitude * (2 * seq - 1)           # {0,1} -> {-a,+a}
        return u

       

    #-----------------------------------------------------
    # -----------generating ssm matrices
    ##---------------------------------------------------

    def get_ssm_2D(self) -> LDSParams:
        '''
        use to quickly return a set of 2 state lgssm matrices
        
        '''
         #2 hidden state 2 input state matrices
        A = np.array([[0.01,0 ],
                    [0, 0.01]])
        B = np.array([[0.2, 0],
                    [0,0.2 ]])
        C = np.array([[0.95, 0],
                    [0,0.95 ]])
        Q = np.array([[0.95, 0],
                    [0,0.95 ]])
        R = np.array([[0.95, 0],
                    [0,0.95 ]])
        mu_0 = np.array([0,0])
        P_0 = np.array([[0.95, 0],
                    [0,0.95 ]])


        params = (A,B,C,Q,R,mu_0,P_0)
        return LDSParams.from_tuple(params)
    
    def generate_general_ssm_matrices(self,) -> LDSParams:
        '''
        Use to generate general stable SSM matrices.


        returns
        -------
        LDSParams
            dataclass with random stable matrices (A, B, C, D, Q, R, mu_0, P_0)
        '''
        rng = np.random.default_rng()
        n, u, m = self.x_dim,self.input_dim, self.y_dim

        # A: random Gaussian rescaled to spectral radius 0.95 → stable, visible dynamics
        A = rng.standard_normal((n, n))
        A = A * (0.95 / np.max(np.abs(np.linalg.eigvals(A))))

        # B, C, D: random Gaussian, full rank almost surely
        B = rng.standard_normal((n, u))
        C = rng.standard_normal((m, n))

        # Q, R: PSD via M Mᵀ. Scale chosen so signal dominates noise.
        Mq = rng.standard_normal((n, n))
        Q  = 0.05 * (Mq @ Mq.T) / n

        Mr = rng.standard_normal((m, m))
        R  = 0.05 * (Mr @ Mr.T) / m

        # Initial state: stationary distribution
        mu_0 = np.zeros(n)
        P_0  = solve_discrete_lyapunov(A, Q)

        return LDSParams(A=A, B=B, C=C, Q=Q, R=R, mu_0=mu_0, P_0=P_0)
    

            
    
    


    






















