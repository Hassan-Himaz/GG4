import numpy as np
import scipy.linalg as sla
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from LDSParams import LDSParams


import jax
jax.config.update("jax_enable_x64", True)

from Illustrator import Illustrator


from sklearn.decomposition import PCA
from dynamax.parameters import ParameterProperties

class dynamax_EM_Fitting():

    def __init__(self,
                 chosen_state_dimension:int,
                 chosen_input_dimension:int,
                 observation:np.ndarray,
                 num_restarts:int = 5,
                 num_em_iters = 100,
                 ):
        '''
        to intilise this EM estimator we need to provide an estimated number of hidden states 

        parameters
        -----------
        chosen_state_dimension:int

        chosen_input_dimension:int

        observation:np.ndarray
            needs to be used in Estimator function, has shape (timepoints,neuron)

        num_restarts:int

        num_em_iters: int


        
        
        '''
        self.observation = observation
        self._time_cnt = observation.shape[0]
        self._neuron_cnt = observation.shape[1]
        self.num_restarts = num_restarts
        self.num_em_iters = num_em_iters
        self.state_dim = chosen_state_dimension
        self.emission_dim = self._neuron_cnt
        self.model = LinearGaussianSSM(state_dim = chosen_state_dimension,emission_dim = self._neuron_cnt,input_dim = chosen_input_dimension)
        print('state_dim  :  {0}'.format(self.state_dim))


    def fit(self,u = None) ->Tuple[LDSParams, float]:

        '''
        params returned are of this form
        

        mu_0 = params.initial.mean      # shape (d,)
        P_0  = params.initial.cov       # shape (d, d)
        A    = params.dynamics.weights
        B    = params.dynamics.input_weights
        Q    = params.dynamics.cov
        C    = params.emissions.weights
        R    = params.emissions.cov

        returns
        --------
        best_params, best_ll :Tuple[LDSParams, float]
                
        '''

        print("obs shape:", self.observation.shape)
        print("obs range:", self.observation.min(), self.observation.max())
        print("obs std:", self.observation.std())
        print("any nan/inf in obs?", np.any(~np.isfinite(self.observation)))


        #difficult to do type annotations with these bespoke jax array objects
        real_data = self.observation

                #store observed data in emissions array
        emissions = jnp.asarray(real_data)
        inputs = jnp.asarray(u) if u is not None else None

        best_params, best_ll = None, float("-inf")

        #now start the restarts loop

        progress_bar = tqdm(range(self.num_restarts),desc = "EM restarts")
        print(self.state_dim)

        for restart in progress_bar:
            
            key = jr.PRNGKey(restart)


            
            obs_flat = real_data.reshape(-1, self.emission_dim)
            pca = PCA(n_components=self.state_dim).fit(obs_flat)
            C_init = jnp.asarray(pca.components_.T)
            
            k_perturb, k_init = jr.split(key)
            C_init_pert = C_init + 0.05 * jr.normal(k_perturb, C_init.shape)   # (emission_dim, state_dim)

            params, props = self.model.initialize(
                key,
                emission_weights=C_init,                         # 
                initial_mean=jnp.zeros(self.state_dim),
                initial_covariance=jnp.eye(self.state_dim),
            )

            # Freeze initial-state params so EM doesn't update them
            
            props = props._replace(
                initial=props.initial._replace(
                    mean=ParameterProperties(trainable=False),
                    cov=ParameterProperties(trainable=False, constrainer=props.initial.cov.constrainer),
                )
            )

            
            params, lls = self.model.fit_em(
                params, props, emissions=emissions, inputs=inputs, num_iters=self.num_em_iters,
            )
        

            lls_arr = np.asarray(lls)
            if np.any(~np.isfinite(lls_arr)):
                tqdm.write(f"restart {restart}: diverged (NaN in lls), skipping")
                continue

            final_lls = float(lls_arr[-1])

            if final_lls > best_ll:
                best_ll = final_lls
                best_params = params

            progress_bar.set_postfix(best_ll=f"{best_ll:.2f}", current_ll=f"{lls[-1]:.2f}")



        if best_params is None:
            raise RuntimeError("All EM restarts diverged. Try smaller state_dim or different init.")
        
        
        return LDSParams.from_dynamax(best_params), best_ll




    
        


    