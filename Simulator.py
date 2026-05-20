from Illustrator import Illustrator
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from typing import Callable

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
        

#input function factory
def make_pulse(t_on: int, t_off: int, amplitude: float, num_inputs: int) -> Callable:
    def pulse(time: int, data: list) -> np.ndarray:
        if t_on <= time < t_off:
            return np.full(num_inputs, amplitude)
        return np.zeros(num_inputs)
    return pulse

def make_sine(freq: float, amplitude: float, num_inputs: int, dt: float = 1.0):
    def sine(time, data):
        return np.full(num_inputs, amplitude * np.sin(2 * np.pi * freq * time * dt))
    return sine

def make_zero(num_inputs: int):
    return lambda time, data: np.zeros(num_inputs)


 #Example usage
def step_controller(time,
                    output,
                    ):
    
    ''' 

    '''
    return [1]

def sinusoidal_controller(time, states):

    return np.array([
        np.sin(time / 5),
        np.cos(time / 10)
    ])




# dt = 0.1
# illustrator = Illustrator(np.load("ExampleDataset.npy"))

# sim = Simulator([

#     # A : dynamics matrix
#     [
#         [1, 0, dt, 0],
#         [0, 1, 0, dt],
#         [0, 0, 0.98, 0],
#         [0, 0, 0, 0.98]
#     ],

#     # B : control matrix
#     [
#         [0, 0],
#         [0, 0],
#         [1, 0],
#         [0, 1]
#     ],

#     # C : observation matrix
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, 0]
#     ],

#     # Q : process covariance
#     [
#         [0.01, 0, 0, 0],
#         [0, 0.01, 0, 0],
#         [0, 0, 0.05, 0],
#         [0, 0, 0, 0.05]
#     ],

#     # R : observation covariance
#     [
#         [0.5, 0],
#         [0, 0.5]
#     ]
    
# ],
# illustrator)


# data = sim.generate_data(
#     trials=10,
#     lengths=100,
#     seed=[0, 0, 1, 1],
#     control_function=sinusoidal_controller
# )
