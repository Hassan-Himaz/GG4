''' going to be using Prediction Error Minimization (PEM) as a framework for understanding how the brain works.


    going to be choosing number of hidden states, number of observed states(fixed by neurons) and number of inputs

    then will randomly initialise the matrices A, B, C, Q, R

    then will try to use iterative methods to fit the data.


'''


from calendar import c

import numpy as np
from scipy.optimize import minimize
import test
from tqdm import tqdm

class PEM_Framework():

    def __init__(
        self,
        number_hidden_states: int = 2,
        number_observed_states: int = 16,
        number_inputs: int = 0,
        number_restarts: int = 2,
        ):

        self.number_hidden_states = number_hidden_states
        self.number_observed_states = number_observed_states
        self.number_inputs = number_inputs
        self.number_restarts = number_restarts

        self.A = np.random.rand(number_hidden_states, number_hidden_states)  # Dynamics matrix
        self.B = np.random.rand(number_hidden_states, number_inputs)  # Control matrix

        self.C = np.random.rand(number_observed_states, number_hidden_states)  # Observation matrix
        #process noise are multivariate normal with covariance Q
        self.Q = np.random.rand(number_hidden_states, number_hidden_states)  # Process covariance
        self.R = np.random.rand(number_observed_states, number_observed_states)  # Observation covariance


    def generate_trial(
            self,
            length: int = 60,

    ) -> np.ndarray:
        '''
        Use to generate a single trial of data using the current parameters of the model. This will be used to test the fitting procedure.

        parameters
        ----------
        length : int
            The number of timepoints in the trial.  
        
        returns
        -------
        data : np.ndarray
            A 2D array of shape (timepoints, observed_states) representing the observed neural activity over time.
        '''
        #this is like existing simulator class method

        #initiialise the latent state as random vector with correct shape
        latent_state = np.random.rand(self.A.shape[0])
        #us y = Cx + noise to get the observed state
        observed_state = np.matmul(self.C,latent_state) + np.random.multivariate_normal(np.zeros(self.R.shape[0]),self.R)
        data = [observed_state]  

        for time in range(length):
            #update the latent state using x = Ax + Bu + noise
            latent_state = np.matmul(self.A,latent_state) + np.random.multivariate_normal(np.zeros(self.Q.shape[0]),self.Q)
            #get the observed state using y = Cx + noise
            observed_state = np.matmul(self.C,latent_state) + np.random.multivariate_normal(np.zeros(self.R.shape[0]),self.R)
            data.append(observed_state)

        return np.array(data)

    def generate_dataset(
            self,
            num_trials: int = 5,
            ) -> np.ndarray:
        
        """
        Use to generate a dataset of multiple trials using the current parameters of the model. This will be used to test the fitting procedure.

        parameters
        ----------
        num_trials : int
            The number of trials to generate.

        returns
        -------
        dataset : np.ndarray
            shape (trials, timepoints, observed_states) 
        
        
        """
        for trial in range(num_trials):
            trial_data = self.generate_trial()
            if trial == 0:
                generated_dataset = np.zeros((num_trials, trial_data.shape[0], trial_data.shape[1]))
            generated_dataset[trial] = trial_data

        return generated_dataset
    

    def calculate_innovations(self, real_data: np.ndarray,trial,lds_params:np.ndarray,u:np.ndarray) -> [np.ndarray,np.ndarray]:
        '''
        Use to calculate the innovations (error between our predicted next output and the actual next output) of the model on the real data. 

        parameters
        ----------
        real_data : np.ndarray
            A 3D array of shape (trials, timepoints, observed_states) representing the observed neural activity over time for multiple trials.

        lds_params : np.ndarray
            A 3D array of shape (A,B,C,Q,R,mu_0,P_0) representing the current parameters of the model.
            where mu_0 is the inital mean of the latent state
            and P_0 is the initial covariance of the latent state.

        u : np.ndarray
            A 3D array of shape (trials, timepoints, number_inputs) representing the control inputs over time for multiple trials. If no control inputs, can be set to None or an array of zeros.


        returns
        -------
        innovations : np.ndarray
            A 2D array of shape (timepoints-1, observed_states) representing the prediction errors of the model on the real data.
        '''
        #this is where we will implement a kalman filter to get the predicted next observed state given all 
        # previous observed states, and then calculate the innovations as the difference between the predicted and actual observed states.

        A = lds_params[0]
        B = lds_params[1]
        C = lds_params[2]   
        Q = lds_params[3]
        R = lds_params[4]
        mu_0 = lds_params[5]
        P_0 = lds_params[6]

        if u is None:
            u = np.zeros((real_data.shape[1],B.shape[1])) #shape (timepoints, number_inputs)

        data_time_series = real_data[trial]
        timepoints, neuron = data_time_series.shape

        num_hidden_states = mu_0.shape[0]

        innovations = np.zeros((timepoints, neuron))
        innovations_covariance = np.zeros((timepoints, neuron, neuron))

        mu = mu_0.copy()
        P = P_0.copy()

        for time in range(0,timepoints):
            #prediction next hidden state

            if time == 0:
                mu_pred = mu_0.copy()
                P_pred  = P_0.copy()
            else:
                mu_pred = A @ mu + B @ u[time-1]
                P_pred  = A @ P @ A.T + Q

            #predicting next observed state
            y_pred = C @ mu_pred
            S = C @ P_pred @ C.T + R


            #innovation
            e = data_time_series[time] - y_pred

            #add to innovations array
            innovations[time] = e
            innovations_covariance[time] = S

            #update using kalman filter equations
            #where K is the kalman gain --> minimum current output covariance

            K = np.linalg.solve(S.T, (P_pred @ C.T).T).T

            mu = mu_pred + K @ e
            P = (np.eye(num_hidden_states) - K @ C) @ P_pred

        return innovations, innovations_covariance
    


    def calculate_NLL(self,real_data:np.ndarray,lds_params:np.ndarray,u: np.ndarray) -> float:
        '''
        Use to return the negative log likelihood of the innovations on real data

        parameters
        -----------
        real_data : np.ndarray
            A 3D array of shape (trials, timepoints, observed_states) representing the observed neural activity over time for multiple trials.
        lds_params : np.ndarray
            A 3D array of shape (A,B,C,Q,R,mu_0,P_0) representing the current parameters of the model.
            where mu_0 is the inital mean of the latent state
            and P_0 is the initial covariance of the latent state.
        
        returns
        -------
        NLL : float
            The negative log likelihood of the innovations on the real data given the current parameters of the model.
        
        
        
        '''
        num_trials = real_data.shape[0]
        num_observed_states = real_data.shape[2] # num of neurons

        total = 0.0

        for trial in range(num_trials):
            innovations, innovations_covariance = self.calculate_innovations(real_data,trial,lds_params,u)

            for time in range(innovations.shape[0]):
                e = innovations[time]
                S = innovations_covariance[time]

                #calculate negative log likelihood of this innovation
                log_likelihood = 0.5 * (e.T @ np.linalg.inv(S) @ e + np.log(np.linalg.det(S)) + num_observed_states * np.log(2 * np.pi))
                total += log_likelihood
        
        return total
    


    #---------------------------------------------------------------------------------------------

    #helper methods
    
    def random_init(self, d: int, n: int, m: int, seed: int = None, 
                    stability_radius: float = 0.9) -> list:
        """
        helper to randomly initialise the parameters of the model 

        d = latent state dim
        n = observation dim
        m = input dim
        
        
        """
        rng = np.random.default_rng(seed)
        
        # A: scale to stability_radius
        A_raw = rng.standard_normal((d, d))
        spec_rad = max(np.abs(np.linalg.eigvals(A_raw)))
        A = A_raw * (stability_radius / spec_rad)
        
        B = rng.standard_normal((d, m)) * 0.1
        C = rng.standard_normal((n, d)) * 0.5
        Q = 0.1 * np.eye(d)
        R = 0.5 * np.eye(n)
        mu_0 = rng.standard_normal(d) * 0.1
        P_0 = np.eye(d)
        
        return [A, B, C, Q, R, mu_0, P_0]

    

    def pack_params(self, A, B, C, Q, R, mu_0, P_0) -> np.ndarray:
        """Flatten all parameters into a 1-D vector for scipy.optimize."""
        L_Q  = np.linalg.cholesky(Q)
        L_R  = np.linalg.cholesky(R)
        L_P0 = np.linalg.cholesky(P_0)
        return np.concatenate([
            A.ravel(),
            B.ravel(),
            C.ravel(),
            L_Q.ravel(),
            L_R.ravel(),
            mu_0.ravel(),
            L_P0.ravel(),
    ])

    def unpack_params(self, flat: np.ndarray, d: int, n: int, m: int) -> list:
        """
        Reverse of pack_params.
        
        d = latent state dim
        n = observation dim
        m = input dim
        """
        i = 0
        A    = flat[i:i+d*d].reshape(d, d);   i += d*d
        B    = flat[i:i+d*m].reshape(d, m);   i += d*m
        C    = flat[i:i+n*d].reshape(n, d);   i += n*d
        
        L_Q  = np.tril(flat[i:i+d*d].reshape(d, d));  i += d*d
        Q    = L_Q @ L_Q.T + 1e-6 * np.eye(d)
        
        L_R  = np.tril(flat[i:i+n*n].reshape(n, n));  i += n*n
        R    = L_R @ L_R.T + 1e-6 * np.eye(n)
        
        mu_0 = flat[i:i+d];                   i += d
        
        L_P0 = np.tril(flat[i:i+d*d].reshape(d, d));  i += d*d
        P_0  = L_P0 @ L_P0.T + 1e-6 * np.eye(d)
        
        return [A, B, C, Q, R, mu_0, P_0]
    

    def objective(self, flat_params, real_data, dims, u=None):
        """NLL as a function of flat parameter vector."""
        try:
            params = self.unpack_params(flat_params, **dims)
            return self.calculate_NLL(real_data, params, u)
        except np.linalg.LinAlgError:
            return 1e10  # filter blew up, return huge penalty

    


    #---------------------------------------------------------------------------------------------
    
    def fit(self,real_data: np.ndarray, u: np.ndarray):
        '''
        Use to now fit the LDS parameters by minimising the negative log-likelihood of the innovations

        parameters
        ----------
        real_data : np.ndarray
            A 3D array of shape (trials, timepoints, observed_states) representing the observed neural activity over time for multiple trials.

        

        u : np.ndarray
            A 3D array of shape (trials, timepoints, number_inputs) representing the control inputs over time for multiple trials. If no control inputs, can be set to None or an array of zeros.

        return
        -------
        fitted parameters : np.ndarray
            A 3D array of shape (A,B,C,Q,R,mu_0,P_0) representing the fitted parameters of the model after minimising the negative log-likelihood of the innovations on the real data.

        
        '''
        num_observed_neurons = real_data.shape[2]

        best_nll = np.inf
        best_params = None

        latent_state_dimension = self.number_hidden_states
        observed_state_dimension = self.number_observed_states
        input_dimension = self.number_inputs

        dims = {'d': latent_state_dimension, 'n': observed_state_dimension, 'm': input_dimension}

        overall_bar = tqdm(total=self.number_restarts, desc="PEM restarts")


        for restart in range(self.number_restarts):
            init_params = self.random_init(latent_state_dimension, observed_state_dimension, input_dimension, seed=restart)
            flat_init = self.pack_params(*init_params)

            # Inner bar that counts iterations within this restart
            iter_count = [0]
            inner_bar = tqdm(desc=f"  Restart {restart}", leave=False, position=1)
            
            def callback(x):
                iter_count[0] += 1
                inner_bar.update(1)
            
            result = minimize(
                self.objective,
                flat_init,
                args=(real_data, dims, u),
                method='L-BFGS-B',
                options={'maxiter': 20},
                callback=callback,
                )
            
            inner_bar.close()
            overall_bar.update(1)
            
            
            #print(f"Restart {restart}: final NLL = {result.fun:.2f}")
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = self.unpack_params(result.x, **dims)

        overall_bar.close()
        
        return best_params, best_nll
    
    def pem_main(
            self,
            data_set: np.ndarray,
            training_trials: list ,
            testing_trials: list,
    ) -> tuple[np.ndarray, float,float]:
        '''
        Use to run the whole PEM procedure on a given dataset, which will return the fitted parameters of the model.

        parameters
        ----------
        data_set : np.ndarray
            A 3D array of shape (trials, timepoints, observed_states) representing the observed neural activity over time for multiple trials.

        training_trials : list
            A list of trial indices to use for training the model.

        returns
        -------
        fitted parameters : np.ndarray
            A 3D array of shape (A,B,C,Q,R,mu_0,P_0) representing the fitted parameters of the model after minimising the negative log-likelihood of the innovations on the real data.

        
        '''

        #then we can fit the model to the data using our fit method

        fitted_params, nll = self.fit(data_set[training_trials], u=None)

        # then want to see how well the model performs on held out trial data
        test_nll = self.calculate_NLL(data_set[testing_trials], fitted_params, u=None)

        training_nll_per_trial = nll / len(training_trials)
        test_nll_per_trial = test_nll / len(testing_trials)

        print(f"Training NLL per trial: {training_nll_per_trial:.2f}")
        print(f"Test NLL per trial:     {test_nll_per_trial:.2f}")

        return fitted_params, nll,test_nll

    


    