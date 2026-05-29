import numpy as np
import scipy.linalg as sla
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple,Callable
from ZLDSParams import LDSParams
import jax
from sklearn.decomposition import PCA
from dynamax.parameters import ParameterProperties
jax.config.update("jax_enable_x64", True)





class Dynamax_EM_Fitting():

    def __init__(self,
                 chosen_state_dimension:int,
                 chosen_input_dimension:int,
                 observation:np.ndarray,
                 inputs:np.ndarray,
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
        self.model:LinearGaussianSSM = LinearGaussianSSM(state_dim = chosen_state_dimension,emission_dim = self._neuron_cnt,input_dim = chosen_input_dimension)
        self.inputs = inputs


    def fit(self) ->Tuple[LDSParams, float,np.ndarray]:

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
    
        if self.inputs is not None and self.model.input_dim == 0:
            raise ValueError(
                f"Passed inputs of shape {self.inputs.shape}, but model was constructed "
                f"with input_dim=0. Reconstruct with input_dim={self.inputs.shape[-1]}."
            )
        if self.inputs is None and self.model.input_dim > 0:
            raise ValueError(
                f"Model expects input_dim={self.model.input_dim}, got inputs=None."
            )

        #difficult to do type annotations with these bespoke jax array objects
        real_data = self.observation

                #store observed data in emissions array
        emissions = jnp.asarray(real_data)
        inputs_array = jnp.asarray(self.inputs) if self.inputs is not None else None

        best_params, best_ll = None, float("-inf")

        #now start the restarts loop

        progress_bar = tqdm(range(self.num_restarts),desc = "EM restarts")


        #llr history across restarts
        llr_arr = list()
        llr_to_return = np.zeros(self.num_em_iters)
        
        for restart in progress_bar:
            
            key = jr.PRNGKey(restart)


            
            # obs_flat = real_data.reshape(-1, self.emission_dim)
            # pca = PCA(n_components=self.state_dim).fit(obs_flat)
            # C_init = jnp.asarray(pca.components_.T)
            
            # k_perturb, k_init = jr.split(key)
            # C_init_pert = C_init + 0.05 * jr.normal(k_perturb, C_init.shape)   # (emission_dim, state_dim)

            params, props = self.model.initialize(
                key,
                # emission_weights=C_init,                         # 
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

            
            params, llr = self.model.fit_em(
                params, props, emissions=emissions, inputs=inputs_array, num_iters=self.num_em_iters,
            )
        

            llr_local = np.asarray(llr)
            if np.any(~np.isfinite(llr_local)):
                tqdm.write(f"restart {restart}: diverged, skipping")
                continue
            llr_arr.append(llr_local)

            final_llr = float(llr[-1])

            if final_llr > best_ll:
                llr_to_return = np.asarray(llr)
                best_ll = final_llr
                best_params = params

            progress_bar.set_postfix(best_ll=f"{best_ll:.2f}", current_ll=f"{llr[-1]:.2f}")



        

        if best_params is None:
            raise RuntimeError("All EM restarts diverged. Try smaller state_dim or different init.")
        

        # mean_llr_arr = np.stack(llr_arr,axis=0)
        # mean_llr_arr = mean_llr_arr.mean(axis = 0)
        
        return LDSParams.from_dynamax(best_params), best_ll, llr_to_return




    
        


    