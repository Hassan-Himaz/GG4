#can improve performance in number of ways if required 
import numpy as np
rng = np.random.default_rng()
class Simulator:

    """
    A class to simulate neural activity data.
    Initialise with list or matrix parameters: A,B,C,Q,R
    """

    def __init__(self,parameters: np.ndarray):
        """
        Initialize the Model with set Parameters
        Accepts a 3D numpy array of shape (A,B,C,Q,R) and stores it for data generation
        Parameters:
            parameters (np.ndarray): 3D numpy array of shape (A,B,C,Q,R) - these need to be correct else will throw error
        """
        self.A, self.B, self.C, self.Q, self.R = map(np.array, parameters)
        self.x_dimensions = len(self.Q)
        self.y_dimensions = len(self.R)

    def generate_data(self, trials: int, lengths: int, seed: np.array, control_function: callable):
        '''Generates the data, given set number of trials, lenghts of times, seed for x, and a provided control function'''
        dataset = []
        for trial in range(trials):
            dataset.append(self.__generate_trial(length=lengths, seed = seed, control_function=control_function))
        return np.array(dataset)
    
    def __generate_trial(self, length, seed, control_function):
        latent_state = seed
        observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
        data = [observed_state]
        for time in range(length):
            latent_state = np.matmul(self.A,latent_state) + np.matmul(self.B,control_function(time,data))+ rng.multivariate_normal(np.zeros(self.x_dimensions),self.Q)
            observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
            data.append(observed_state)
        return np.array(data)
    
#Example usage
def step_controller(time,output):
    return [1]

def sinusoidal_controller(time, states):

    return np.array([
        np.sin(time / 5),
        np.cos(time / 10)
    ])


dt = 0.1

sim = Simulator([

    # A : dynamics matrix
    [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 0.98, 0],
        [0, 0, 0, 0.98]
    ],

    # B : control matrix
    [
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ],

    # C : observation matrix
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ],

    # Q : process covariance
    [
        [0.01, 0, 0, 0],
        [0, 0.01, 0, 0],
        [0, 0, 0.05, 0],
        [0, 0, 0, 0.05]
    ],

    # R : observation covariance
    [
        [0.5, 0],
        [0, 0.5]
    ]
])


data = sim.generate_data(
    trials=10,
    lengths=100,
    seed=[0, 0, 1, 1],
    control_function=sinusoidal_controller
)
