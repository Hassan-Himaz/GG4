import numpy as np
import scipy.linalg as sla
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jr

import jax
jax.config.update("jax_enable_x64", True)

from Illustrator import Illustrator


from sklearn.decomposition import PCA
from dynamax.parameters import ParameterProperties

class dynamax_EM_Fitting():

    def __init__(self,
                 chosen_state_dimension:int,
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
        self.state_dim = chosen_state_dimension
        self.emission_dim = illustrator._neuron_cnt
        self.model = LinearGaussianSSM(state_dim = chosen_state_dimension,emission_dim = self._neuron_cnt,input_dim = chosen_input_dimension)
        print('state_dim  :  {0}'.format(self.state_dim))


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

        print("obs shape:", self.observation.shape)
        print("obs range:", self.observation.min(), self.observation.max())
        print("obs std:", self.observation.std())
        print("any nan/inf in obs?", np.any(~np.isfinite(self.observation)))


        #difficult to do type annotations with these bespoke jax array objects
        real_data = self.observation

        # optional but recommended — standardise per neuron across (trials, time)
        real_data = real_data - real_data.mean(axis=(0, 1), keepdims=True)
        real_data = real_data / (real_data.std(axis=(0, 1), keepdims=True) + 1e-8)

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
            C_init = jnp.asarray(pca.components_.T)   # (emission_dim, state_dim)

            params, props = self.model.initialize(
                key,
                dynamics_weights=0.9 * jnp.eye(self.state_dim),
                dynamics_covariance=0.1 * jnp.eye(self.state_dim),
                emission_weights=C_init,                          # data-driven
                emission_covariance=jnp.eye(self.emission_dim),
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

            # Step EM one iteration at a time and inspect
            for step in range(3):
                params, lls = self.model.fit_em(
                    params, props, emissions=emissions, inputs=inputs, num_iters=1,
                )
                A = np.asarray(params.dynamics.weights)
                Q = np.asarray(params.dynamics.cov)
                C = np.asarray(params.emissions.weights)
                R = np.asarray(params.emissions.cov)
                mu_0 = np.asarray(params.initial.mean)
                P_0 = np.asarray(params.initial.cov)
                print(f"\n--- restart {restart}, EM step {step} ---")
                print(f"  ll: {float(lls[-1]):.2f}")
                print(f"  A:    nan={np.any(np.isnan(A))}, eigs={np.linalg.eigvals(A)}")
                print(f"  Q:    nan={np.any(np.isnan(Q))}, eigs={np.linalg.eigvalsh(Q)}")
                print(f"  C:    nan={np.any(np.isnan(C))}, max|.|={np.abs(C).max():.2e}")
                print(f"  R:    nan={np.any(np.isnan(R))}, eigs_min={np.linalg.eigvalsh(R).min():.2e}, eigs_max={np.linalg.eigvalsh(R).max():.2e}")
                print(f"  mu_0: nan={np.any(np.isnan(mu_0))}, val={mu_0}")
                print(f"  P_0:  nan={np.any(np.isnan(P_0))}, eigs={np.linalg.eigvalsh(P_0)}")
                if not np.isfinite(float(lls[-1])):
                    print("  ↑↑ NaN appeared this step")
                    break
            break    # only do one restart for now

        if best_params is None:
            raise RuntimeError("All EM restarts diverged. Try smaller state_dim or different init.")
        return best_params, best_ll




    
        


    