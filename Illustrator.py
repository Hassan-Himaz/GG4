import numpy as np
import matplotlib.pyplot as plt

class Illustrator:
    """
    A class to visualize neural activity data.
    Please finish the implementation and annotations of this class.
    """
    __slot__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt']

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator with the given observation data.
        Accepts a 3D numpy array of shape (Trials, Timepoints, Neurons) and stores it for visualization.
        Parameters:
            observation (np.ndarray): A 3D array containing neural activity data with dimensions (Trials, Timepoints, Neurons).
        """
        if not isinstance(observation, np.ndarray):
            raise ValueError('Ensure observation passed as a np array')

        if len(observation.shape)!=3:
            raise ValueError('Incorrect format of the observation. Should be a 3-D array')
        self._observation = observation # keep as protected for keeping data set
        self._trial_cnt, self._timestep_cnt, self._neuron_cnt = observation.shape # keep as protected for keeping data safe


    """
    Prefer working with propeties so that the data is safe.
    Particularly useful in changing the observations
    """
    @property
    def neuron_cnt(self):
        return self._neuron_cnt
    
    @property
    def trial_cnt(self):
        return self._trial_cnt
    
    @property
    def timestep_cnt(self):
        return self._timestep_cnt

    @property
    def observation(self):
        return self._observation
    
    @observation.setter
    def observation(self, new_observation: np.ndarray)->None:
        self._observation = new_observation
        self._trial_cnt, self._timestep_cnt, self._neuron_cnt = new_observation.shape
    
    def get_trial_averaged_mean(self, chosen_neurons: list[int]|np.ndarray|int|None=None)->np.ndarray:
        """
        Use this function to obtain the trial averaged values for the desired neurons
        Eg: for each time step accross 5 trials average the values for each neuron
        If left None, returns time average for all the neurons accross trials
        If list: ensure the list is 0 indexed count of neuron eg: [0,5] returns the 1st and 6th neurons time averaged activity
        The same logic applies for int
        """
        if isinstance(chosen_neurons, list) or isinstance(chosen_neurons, np.ndarray):
            if max(chosen_neurons)>self.neuron_cnt:
                raise ValueError('One of the chosen neurons not in the dataset. Check indexing')
            filtered_neurons = self.observation[:,:,chosen_neurons] # get neural data for the filtered neurons
            return np.mean(filtered_neurons, axis=0) 
        elif isinstance(chosen_neurons, int):
            if chosen_neurons > self.neuron_cnt:
                raise ValueError('The chosen neuron is not in the dataset. Check indexing')
            filtered_neurons = self.observation[:,:,chosen_neurons] # get neural data for the filtered neurons
            return np.atleast_2d(np.mean(filtered_neurons, axis=0)).T # Use atleast 2D such that structure preserved (timesteps, recording count)
        
        elif chosen_neurons is None:
            return np.mean(self.observation[:,:,:],axis=0)
        else:
            raise ValueError('Invalid data type') 

    def get_covariance_matrix(self)->np.ndarray:
        """Use this to calculate the covariance matrix accross the neurons
        A_{ij} = E[X_iX_j] - E[X_i]E[X_j] for X_i, X_j neurons
        This can be studied how the overall activity levels influence from one neuron to another
        """
        reshaped_observations = self.observation.reshape(self.trial_cnt*self.timestep_cnt, self.neuron_cnt)
        return np.cov(reshaped_observations,rowvar=False)

    def get_autocovariance_matrix(self, neuron_index: int)->np.ndarray:
        """Use this to get the autocovariance matrix of specific neural process
        C_{t1,t2} = E[X_t1 X_t2] - E[X_t1]E[X_t2]
        This can help understand the temporal variance
        """
        individual_data = self.observation[:,:,neuron_index]
        return np.cov(individual_data, rowvar=False)
    
    def get_trial_covariance_matrix(self):
        """Use this to udnerstand the variance between trials
        This can help validate if the trials are independent or not and form sthe base of the statistical analysis.
        """
        reshaped = self.observation.reshape(self.trial_cnt, self.timestep_cnt*self.neuron_cnt)
        return np.cov(reshaped)
    
    





