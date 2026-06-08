import numpy as np
from typing import Callable


class Controllers():
    '''
    class to store the inputs function factory to return Callable inputs for use in the Simulator
    
    
    '''
    def __init__(self):
        pass

    #input function factory
    def make_pulse(self,t_on: int, t_off: int, amplitude: float, num_inputs: int) -> Callable:
        def pulse(time: int, data: list) -> np.ndarray:
            if t_on <= time < t_off:
                return np.full(num_inputs, amplitude)
            return np.zeros(num_inputs)
        return pulse

    def make_sine(self,freq: float, amplitude: float, num_inputs: int, dt: float = 1.0):
        def sine(time, data):
            return np.full(num_inputs, amplitude * np.sin(2 * np.pi * freq * time * dt))
        return sine

    def make_zero(self,num_inputs: int):
        return lambda time, data: np.zeros(num_inputs)
    
    def make_ramp(self, t_on: int, t_off: int, slope: float, num_inputs: int) -> Callable:
        """Linear ramp from 0 at t_on, growing by `slope` per step, held flat after t_off."""
        def ramp(time: int, data: list) -> np.ndarray:
            if time < t_on:
                return np.zeros(num_inputs)
            elapsed = min(time, t_off) - t_on
            return np.full(num_inputs, slope * elapsed)
        return ramp
    
#state estimator used instead of reacalcualting the state matrices each time
import numpy as np
import scipy.linalg as la

class AugmentedKalmanFilter:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                 Q_process: np.ndarray, R_measure: np.ndarray, Q_disturbance: float = 0.01):
        """
        Initializes the Augmented Kalman Filter for Offset-Free Tracking.
        
        Q_disturbance: A tuning scalar. Higher values make the filter trust the 
                       disturbance estimate more (faster adaptation to steady-state error, 
                       but more sensitive to random noise).
        """
        self.d = A.shape[0]
        self.m = B.shape[1]
        self.N = C.shape[0]
        
        # 1. Build Augmented Matrices
        # A_aug = [A  0]
        #         [0  I]
        self.A_aug = np.block([
            [A, np.zeros((self.d, self.N))],
            [np.zeros((self.N, self.d)), np.eye(self.N)]
        ])
        
        # B_aug = [B]
        #         [0]
        self.B_aug = np.vstack([B, np.zeros((self.N, self.m))])
        
        # C_aug = [C  I]
        self.C_aug = np.hstack([C, np.eye(self.N)])
        
        # 2. Build Augmented Process Noise Covariance
        Q_d = np.eye(self.N) * Q_disturbance
        self.Q_aug = la.block_diag(Q_process, Q_d)
        
        # 3. Calculate Steady-State Kalman Gain (L)
        P_prior = la.solve_discrete_are(self.A_aug.T, self.C_aug.T, self.Q_aug, R_measure)
        S = self.C_aug @ P_prior @ self.C_aug.T + R_measure
        self.L_aug = P_prior @ self.C_aug.T @ la.inv(S)
        
        # 4. Initialize Internal Augmented State: [x; d]
        self.x_aug = np.zeros((self.d + self.N, 1))

    def update(self, y_k: float, u_prev: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes the current measurement and the previous input, updates the filter, 
        and returns the separated state and disturbance estimates.
        """
        # Ensure correct shapes
        y_k = np.array(y_k).reshape(-1, 1)
        u_prev = np.array(u_prev).reshape(-1, 1)
        
        # 1. Predict current output based on previous state
        y_pred = self.C_aug @ self.x_aug
        
        # 2. Filter Step: Update estimate using the measurement error
        self.x_aug = self.x_aug + self.L_aug @ (y_k - y_pred)
        
        # 3. Extract the separated states
        x_est = self.x_aug[:self.d]
        d_est = self.x_aug[self.d:]
        
        # 4. Predict Step: Advance internal model for the *next* loop iteration
        self.x_aug = self.A_aug @ self.x_aug + self.B_aug @ u_prev
        
        return x_est, d_est
    
from scipy.optimize import minimize
import numpy as np

class ConstrainedMPC:
    def __init__(self, A, B, C, horizon, Q_cost, R_cost):
        self.A = A
        self.B = B
        self.C = C
        self.horizon = horizon
        self.N = horizon
        self.Q = Q_cost  
        self.R = R_cost  
        
        self.d = A.shape[0]
        self.m = B.shape[1]
        
        # Warm start guess
        self.u_guess = np.zeros(self.N * self.m)

    def get_input(self, x_current, target_trajectory_window):
        def cost_function(u_flat):
            u_seq = u_flat.reshape(self.horizon, self.m)
            cost = 0
            x = x_current.copy().reshape(-1, 1)
            
            for i in range(self.horizon):
                u_i = u_seq[i].reshape(-1, 1)
                x = self.A @ x + self.B @ u_i
                y_pred = self.C @ x
                
                # --- UPDATE THIS LINE ---
                # Force the target slice to be a column vector to match y_pred
                y_ref = target_trajectory_window[i].reshape(-1, 1)
                error = y_pred - y_ref
                
                cost += (error.T @ self.Q @ error)[0, 0]
                cost += (u_i.T @ self.R @ u_i)[0, 0]
                
            return cost

        # Strict bounds
        bounds = [(0.0, 1.0) for _ in range(self.horizon * self.m)]
        
        # RELAXED SPEED CONSTRAINTS: 
        # ftol tightened for maximum precision, maxiter removed to allow full convergence
        result = minimize(
            cost_function, 
            self.u_guess, 
            bounds=bounds, 
            method='L-BFGS-B',
            options={'ftol': 1e-6, 'disp': False} 
        )
        
        optimal_u_seq = result.x.reshape(self.N, self.m)
        self.u_guess[:-self.m] = result.x[self.m:]
        self.u_guess[-self.m:] = optimal_u_seq[-1] 
        
        return optimal_u_seq[0]
    
import numpy as np
from scipy.linalg import solve_discrete_are

class LQR:
    def __init__(self, A, B, C, Q_cost_y, R_cost):
        """
        Initializes the LQR controller and pre-computes the optimal gain matrix.
        """
        self.A = A
        self.B = B
        self.C = C
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]
        
        # 1. Transform the Y-space penalty into an X-space penalty
        # Since MPC cost is (Cx - y_ref)^T Q (Cx - y_ref), the state penalty is C^T Q C
        self.Q_x = self.C.T @ Q_cost_y @ self.C
        self.R = R_cost
        
        # 2. Solve the Discrete Algebraic Riccati Equation (DARE) once offline
        P = solve_discrete_are(self.A, self.B, self.Q_x, self.R)
        
        # 3. Compute the optimal feedback gain matrix K
        # K = (B^T P B + R)^-1 B^T P A
        self.K = np.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)
        
        # 4. Precompute the pseudo-inverse for Steady-State Reference Tracking
        # We need to find the ideal steady-state [x_ss, u_ss] that yields y_ref.
        # Matrix M solves: [ I - A,  -B ] [x_ss] = [   0   ]
        #                  [   C  ,   0 ] [u_ss] = [ y_ref ]
        M = np.block([
            [np.eye(self.n_states) - self.A, -self.B],
            [self.C,                         np.zeros((self.n_outputs, self.n_inputs))]
        ])
        
        # We use pinv (pseudo-inverse) because m > n (16 > 2), 
        # meaning perfect tracking of all 16 channels is impossible. 
        # pinv automatically finds the best "least-squares" compromise.
        self.M_pinv = np.linalg.pinv(M)

    def get_input(self, x_current, target_trajectory_window):
        """
        Calculates the fast LQR input. 
        Signature matches MPC, but LQR only needs the immediate next target, 
        so it ignores the rest of the window.
        """
        # LQR does not look ahead, so grab just the immediate target
        y_ref = target_trajectory_window[0].reshape(-1, 1)
        x_curr = x_current.reshape(-1, 1)

        # 1. Find the target steady-state state and input to hit y_ref
        target_vector = np.vstack([np.zeros((self.n_states, 1)), y_ref])
        ss_targets = self.M_pinv @ target_vector
        
        x_ss = ss_targets[:self.n_states]
        u_ss = ss_targets[self.n_states:]

        # 2. Apply the LQR control law: u = -K(x - x_ss) + u_ss
        u_optimal = -self.K @ (x_curr - x_ss) + u_ss
        
        # 3. Enforce the hard boundaries natively handled by MPC
        u_clipped = np.clip(u_optimal.flatten(), 0.0, 1.0)
        
        return u_clipped