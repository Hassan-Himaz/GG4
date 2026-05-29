from GG4.Illustrator_old import Illustrator
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from typing import Callable
from scipy.linalg import solve_discrete_lyapunov

class Simulator:

    """
    A class to simulate neural activity data.
    Initialise with list or matrix parameters: A,B,C,Q,R
    """

    def __init__(self,parameters,illustrator: Illustrator,controller):
        """
        Initialize the Model with set Parameters
        Accepts a 3D numpy array of shape (A,B,C,Q,R,mu_0,P_0) and stores it for data generation
        Parameters:
            parameters (np.ndarray): 3D numpy array of shape (A,B,C,Q,R,mu_0,P_0)  - these need to be correct else will throw error
        """
        self.A = parameters.A    # using the dataclass
        self.B = parameters.B
        self.C = parameters.C
        self.Q = parameters.Q
        self.R = parameters.R
        self.mu_0 = parameters.mu_0
        self.P_0 = parameters.P_0


        self.x_dimensions = len(self.Q)
        self.y_dimensions = len(self.R)
        self.illustrator = illustrator
        self.trial_cnt = self.illustrator.trial_cnt
        self.timestep_cnt = self.illustrator.timestep_cnt
        self.neuron_cnt = self.illustrator.neuron_cnt
        self.observation = self.illustrator.observation
        self.controller = controller   #input
    

    def generate_data(self, trials: int, lengths: int, seed: tuple[np.ndarray,np.ndarray], input_function: Callable):
        '''Generates the data, given set number of trials, lenghts of times, seed for x, and a provided control function'''
        dataset = []
        for trial in range(trials):
            dataset.append(self._generate_trial(length=lengths, seed = seed, input_function=self.controller))
        return np.array(dataset)
    
    def _generate_trial(self,
                         length:int, #
                         seed:tuple[np.ndarray,np.ndarray], #  seed the inital hidden state value with the infered starting mean - seed should be tuple(m_0,P_0)
                         input_function: Callable) -> np.ndarray:
        '''
        will return timepoint x neuron dimensional array  for a single simulated trial
        
        '''
        #intial latent state
        latent_state = rng.multivariate_normal(seed[0],seed[1])
        observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
        data = [observed_state]
        for time in range(length):
            latent_state = np.matmul(self.A,latent_state) + np.matmul(self.B,input_function(time,data))+ rng.multivariate_normal(np.zeros(self.x_dimensions),self.Q)
            observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
            data.append(observed_state)
        return np.ndarray(data)
    
    def compare_plot(self):
        '''
        method that will plot real trial next to generated trial
        
        '''

        
        
        real_data = self.observation[0] # choose to compare with trial 1
        time = np.arange(self.timestep_cnt)
        em_seed = (self.mu_0,self.P_0)
        simulated_data = self._generate_trial(length = self.timestep_cnt,seed = em_seed, input_function = self.input_pulse)

        plt.figure(1)
        for neuron in range(self.illustrator._neuron_cnt):
            plt.plot(time,real_data[:,neuron])
            plt.plot(time,simulated_data[:,neuron])
        plt.xlabel('time')
        plt.ylabel('')

    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------Input Functions --------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###

    def input_pulse(self,
                    time:int,
                    data:np.ndarray,
                    )-> np.ndarray:
        t_on = 0
        t_off = 100
        amplitude = 10
        num_inputs = 2
        if t_on <= time < t_off:
            return np.full(num_inputs,amplitude)
        else: 
            return np.zeros(num_inputs)
        
    ### ---------------------------------------------------------------------------------------------------- ###
    ###----------------------------------------System Analysis --------------------------------------------- ###
    ### ---------------------------------------------------------------------------------------------------- ###

    def calculate_transfer_function(self, max_frequency=500,points=1000):
        
        # Calculate the transfer function H(s) = C * (sI - A)^(-1) * B
        w = 1j * np.linspace(0, max_frequency, points)  # Frequency range for analysis
        I = np.eye(self.x_dimensions)
        H_s = np.zeros((self.y_dimensions, self.B.shape[1], len(w)), dtype=complex)

        for i in range(len(w)):
            H_s[:, :, i] = self.C @ la.inv(w[i] * I - self.A) @ self.B

        return H_s
    
    def calculate_eigen(self):
        """
        Calculate eigen values for stablity of system
        Calculate eigen vectors for mode of system
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

    ### Can add a method to come up with balanced realization for state space model reduction based on the Hankel singular values.
    
