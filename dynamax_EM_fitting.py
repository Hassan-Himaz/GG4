import numpy as np
import scipy.linalg as sla
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jr

from GG4.Illustrator import Illustrator

class dynamax_EM_Fitting():

    def __init__(self,
                 chosen_state_dimensions:int,
                 chosen_input_dimension:int,
                 illustrator:Illustrator,
                 num_restarts:int = 5,
                 num_em_iters = 100,
                 ):
        '''
        to intilise this EM estimator we need to provide an estimated number of hidden states 
        
        
        '''
        self.observation = illustrator.observation
        self._time_cnt = illustrator._timestep_cnt
        self._neuron_cnt = illustrator._neuron_cnt
        self._trial_cnt = illustrator._trial_cnt
        self.num_restarts = num_restarts
        self.num_em_iters = num_em_iters
        self.model = LinearGaussianSSM(state_dim = chosen_state_dimensions,emission_dim = self._neuron_cnt,input_dim = chosen_input_dimension)


    def fit(self,u = None):

        '''
        params returned are of this form
        
    

        mu_0 = params.initial.mean      # shape (d,)
        P_0  = params.initial.cov       # shape (d, d)
        A    = params.dynamics.weights
        B    = params.dynamics.input_weights
        Q    = params.dynamics.cov
        C    = params.emissions.weights
        R    = params.emissions.cov
                
        '''


        #difficult to do type annotations with these bespoke jax array objects
        real_data = self.observation
        #store observed data in emissions array
        emissions = jnp.asarray(real_data)
        inputs = jnp.asarray(u) if u is not None else None

        best_params, best_ll = None, -jnp.inf

        #now start the restarts loop

        for restart in range(self.num_restarts):
            #randomly generated key for intialising params
            key = jr.PRNGKey(restart)
            params, props = self.model.initialize(key)

            fitted_params, lls = self.model.fit_em(
                params, props,
                emissions=emissions,
                inputs=inputs,
                num_iters=self.num_em_iters,
            )
            if lls[-1] > best_ll:
                best_ll = lls[-1]
                best_params = fitted_params
        return best_params, float(best_ll)




    
        


    