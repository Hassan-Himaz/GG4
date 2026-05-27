from dataclasses import dataclass, astuple,fields
import numpy as np
from pathlib import Path


DEFAULT_DIR = Path.cwd() / "LDSParams_Saves"

@dataclass
class LDSParams:
    A: np.ndarray      # (x, x)  dynamics
    B: np.ndarray      # (x, u)  input
    C: np.ndarray      # (y, x)  observation
    Q: np.ndarray      # (x, x)  process noise cov
    R: np.ndarray      # (y, y)  observation noise cov
    mu_0: np.ndarray   # (x,)    initial state mean
    P_0: np.ndarray    # (x, x)  initial state cov

    # --- Adapters ---
    @classmethod
    def from_dynamax(cls, params) -> "LDSParams":
        """dynamax ParamsLGSSM -> LDSParams."""
        return cls(
            A=np.asarray(params.dynamics.weights),
            B=np.asarray(params.dynamics.input_weights),
            C=np.asarray(params.emissions.weights),
            Q=np.asarray(params.dynamics.cov),
            R=np.asarray(params.emissions.cov),
            mu_0=np.asarray(params.initial.mean),
            P_0=np.asarray(params.initial.cov),
        )
    

    def to_dynamax(self):
        """LDSParams -> dynamax ParamsLGSSM."""
        from dynamax.linear_gaussian_ssm import ParamsLGSSM, ParamsLGSSMInitial, \
            ParamsLGSSMDynamics, ParamsLGSSMEmissions
        import jax.numpy as jnp
        return ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=jnp.asarray(self.mu_0),
                                       cov=jnp.asarray(self.P_0)),
            dynamics=ParamsLGSSMDynamics(weights=jnp.asarray(self.A),
                                         input_weights=jnp.asarray(self.B),
                                         bias=jnp.zeros(self.A.shape[0]),
                                         cov=jnp.asarray(self.Q)),
            emissions=ParamsLGSSMEmissions(weights=jnp.asarray(self.C),
                                           input_weights=jnp.zeros((self.C.shape[0], self.B.shape[1])),
                                           bias=jnp.zeros(self.C.shape[0]),
                                           cov=jnp.asarray(self.R)),
        )

    @classmethod
    def from_tuple(cls, parameters) -> "LDSParams":
        """Legacy tuple format -> LDSParams.
        
        needs format A, B, C, Q, R, mu_0, P_0
        
        
        """
        A, B, C, Q, R, mu_0, P_0 = (np.asarray(p) for p in parameters)
        return cls(A=A, B=B, C=C, Q=Q, R=R, mu_0=mu_0, P_0=P_0)
    
    def to_tuple(self):
        return astuple(self)

    

    def save(self, name: str = "default") -> None:
        '''Save all matrices to ~/LDSParams_Saves/<name>.npz'''
        DEFAULT_DIR.mkdir(exist_ok=True)
        path = DEFAULT_DIR / f"{name}.npz"
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, name: str = "default") -> "LDSParams":
        '''Load matrices from ~/LDSParams_Saves/<name>.npz'''
        path = DEFAULT_DIR / f"{name}.npz"
        with np.load(path) as data:
            return cls(**{f.name: data[f.name] for f in fields(cls)})
