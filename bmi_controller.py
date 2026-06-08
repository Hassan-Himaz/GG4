import numpy as np

# =====================================================================
# SPATIAL & TEMPORAL PLANNING
# =====================================================================
def generate_shape_blueprint(shape_type: str, size: float, start_pos: tuple, num_points: int = 1000) -> np.ndarray:
    """Generates a dense array of [X, Y] coordinates for a given shape."""
    blueprint = np.zeros((num_points, 2))
    if shape_type.lower() == "circle":
        angles = np.linspace(0, 2 * np.pi, num_points)
        blueprint[:, 0] = start_pos[0] + size * np.cos(angles) - size 
        blueprint[:, 1] = start_pos[1] + size * np.sin(angles)
    elif shape_type.lower() == "square":
        side = size
        q = num_points // 4
        blueprint[:q, 0] = np.linspace(start_pos[0], start_pos[0] + side, q)
        blueprint[:q, 1] = start_pos[1]
        blueprint[q:2*q, 0] = start_pos[0] + side
        blueprint[q:2*q, 1] = np.linspace(start_pos[1], start_pos[1] + side, q)
        blueprint[2*q:3*q, 0] = np.linspace(start_pos[0] + side, start_pos[0], q)
        blueprint[2*q:3*q, 1] = start_pos[1] + side
        blueprint[3*q:, 0] = start_pos[0]
        blueprint[3*q:, 1] = np.linspace(start_pos[1] + side, start_pos[1], num_points - 3*q)
    return blueprint

def generate_reference_trajectory(blueprint_xy: np.ndarray, target_speed: float, dt: float) -> tuple:
    """Moves a Virtual Target along the blueprint at a constant speed."""
    diffs = np.diff(blueprint_xy, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0.0)
    total_distance = cumulative_dist[-1]
    
    total_time = total_distance / target_speed
    num_steps = int(total_time / dt)
    
    target_distances = np.linspace(0, total_distance, num_steps)
    
    pos_seq = np.zeros((num_steps, 2))
    pos_seq[:, 0] = np.interp(target_distances, cumulative_dist, blueprint_xy[:, 0])
    pos_seq[:, 1] = np.interp(target_distances, cumulative_dist, blueprint_xy[:, 1])
    
    vel_seq = np.zeros((num_steps, 2))
    vel_seq[:-1, :] = np.diff(pos_seq, axis=0) / dt
    vel_seq[-1, :] = vel_seq[-2, :] 
    
    return pos_seq, vel_seq

# =====================================================================
# TRANSLATION ENGINE
# =====================================================================
class LatentFeedforwardEngine:
    """Translates physical velocity vectors into Neural Chords and Power Locks."""
    def __init__(self, freqs: list, power_rhythm_hz: float):
        self.f_R, self.f_L, self.f_U, self.f_D = freqs
        self.power_hz = power_rhythm_hz
        self.feedforward_gain = 5
        self.max_amp = 3.5          
        
    def calculate_latent_trajectory(self, velocity_seq: np.ndarray, horizon: int) -> np.ndarray:
        num_steps = len(velocity_seq)
        total_steps = num_steps + horizon
        latent_target = np.zeros((total_steps, 2))
        t = np.arange(total_steps)
        
        for k in range(total_steps):
            vx = velocity_seq[k, 0] if k < num_steps else 0.0
            vy = velocity_seq[k, 1] if k < num_steps else 0.0
            
            amp_R = np.clip(self.feedforward_gain * vx, 0, self.max_amp) if vx > 0 else 0.0
            amp_L = np.clip(-self.feedforward_gain * vx, 0, self.max_amp) if vx < 0 else 0.0
            amp_U = np.clip(self.feedforward_gain * vy, 0, self.max_amp) if vy > 0 else 0.0
            amp_D = np.clip(-self.feedforward_gain * vy, 0, self.max_amp) if vy < 0 else 0.0
            
            latent_target[k, 0] = (
                amp_R * np.sin(2 * np.pi * self.f_R * t[k]) +
                amp_L * np.sin(2 * np.pi * self.f_L * t[k]) +
                amp_U * np.sin(2 * np.pi * self.f_U * t[k]) +
                amp_D * np.sin(2 * np.pi * self.f_D * t[k])
            )
            latent_target[k, 1] = 1.0 + 0.5 * np.sin(2 * np.pi * self.power_hz * t[k])
            
        return latent_target

# =====================================================================
# LATENT EXECUTION WRAPPER
# =====================================================================
class LatentBrainWrapper:
    """Shifts the system boundary to output [x1, x2] directly."""
    def __init__(self, full_system, W_total: np.ndarray):
        self.system = full_system
        self.input_dim = 2 
        self.W_total = W_total
        
    def next_state(self, u):
        self.system.next_state(u)
        
    def measure(self) -> np.ndarray:
        y_raw = self.system._brain.measure()
        return self.W_total @ y_raw

import numpy as np

class ContinuousTrajectory:
    def __init__(self, shape_type: str, size: float, start_pos: tuple, speed: float):
        self.shape_type = shape_type.lower()
        self.R = size
        self.X_start = start_pos[0]
        self.Y_start = start_pos[1]
        
        # Calculate angular velocity (omega) based on desired linear speed
        # v = r * omega -> omega = v / r
        self.omega = speed / self.R 
        
    def get_state_at_time(self, t: float) -> tuple:
        """
        Returns the exact, smooth Position and Velocity at time t 
        using analytical derivatives. No arrays, no interpolation.
        """
        if self.shape_type == "circle":
            # Offset center so the circle starts exactly at start_pos
            X_center = self.X_start - self.R
            Y_center = self.Y_start
            
            # Continuous Position
            pos_x = X_center + self.R * np.cos(self.omega * t)
            pos_y = Y_center + self.R * np.sin(self.omega * t)
            
            # Continuous Velocity (Analytical Derivative)
            vel_x = -self.R * self.omega * np.sin(self.omega * t)
            vel_y =  self.R * self.omega * np.cos(self.omega * t)
            
            return np.array([pos_x, pos_y]), np.array([vel_x, vel_y])
            
        else:
            raise NotImplementedError("Only circle implemented for continuous example.")

    def generate_feedforward_sequence(self, num_steps: int, dt: float) -> tuple:
        """Generates the perfectly smooth velocity array, and the position array for plotting."""
        pos_seq = np.zeros((num_steps, 2))
        vel_seq = np.zeros((num_steps, 2))
        
        for k in range(num_steps):
            time_sec = k * dt
            pos, vel = self.get_state_at_time(time_sec)
            pos_seq[k, :] = pos
            vel_seq[k, :] = vel
            
        return pos_seq, vel_seq
    
import numpy as np

def execute_2dof_tracking(wrapper, estimator, mpc, C, latent_target_seq: np.ndarray, num_steps: int):
    """Runs the live closed-loop MPC tracking."""
    horizon = mpc.horizon
    history_y = np.zeros((num_steps, 2))
    history_hand = np.zeros((num_steps, 2))
    u_prev = np.zeros(2)
    
    # 1. Pre-calculate the Pseudo-Inverse of C outside the loop for speed
    C_pinv = np.linalg.pinv(C)
    
    for k in range(num_steps):
        y_k = wrapper.measure()
        current_pos = wrapper.system.hand_pos
        
        x_hat, d_hat = estimator.update(y_k, u_prev)
        
        # 2. Extract the physical 2D target window
        target_window = latent_target_seq[k : k + horizon]
        
        # 3. Apply the disturbance correction (Output Space)
        adjusted_physical_target = target_window - d_hat.flatten().reshape(-1, 2)
        
        # 4. THE FIX: Project the 2D Physical Window -> Abstract State Space
        # This converts the (horizon, 2) array into a (horizon, n_states) array
        abstract_adjusted_target = adjusted_physical_target @ C_pinv.T
        
        # 5. Feed the Abstract Target to the MPC
        u_k = mpc.get_input(x_hat, abstract_adjusted_target)
        
        wrapper.next_state(u_k)
        
        u_prev = u_k
        history_y[k, :] = y_k
        history_hand[k, :] = current_pos
        
    return history_hand, history_y

import matplotlib.pyplot as plt
import numpy as np

def plot_bmi_results(target_pos_seq, target_latent_seq, actual_hand_pos=None, actual_latent_seq=None):
    """
    Diagnostic plotting for 2-DOF Cartesian tracking and Neural Latent Space.
    Pass None to actual_hand_pos and actual_latent_seq to just verify the plans.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ==========================================
    # Plot 1: Cartesian Task Space (The Circle)
    # ==========================================
    ax1.plot(target_pos_seq[:, 0], target_pos_seq[:, 1], 'k--', alpha=0.7, label='Target Path')
    
    if actual_hand_pos is not None:
        ax1.plot(actual_hand_pos[:, 0], actual_hand_pos[:, 1], 'm-', linewidth=2, label='Actual Arm Path')
        ax1.plot(actual_hand_pos[0, 0], actual_hand_pos[0, 1], 'go', label='Start')
    
    ax1.set_title('2-DOF Tracking (Circle)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.axis('equal')  # Forces the X and Y axes to scale equally so the circle doesn't look like an oval
    ax1.grid(True)
    ax1.legend()

    # ==========================================
    # Plot 2: Neural Latent Space (The Brain)
    # ==========================================
    time_steps = np.arange(len(target_latent_seq))
    
    # Plot the Feedforward "Whisper"
    ax1_plan = ax2.plot(time_steps, target_latent_seq[:, 0], 'k--', alpha=0.5, label='Planned x1 (Steering)')
    ax2_plan = ax2.plot(time_steps, target_latent_seq[:, 1], 'g--', alpha=0.5, label='Planned x2 (Power)')
    
    # Plot the Closed-Loop Actuation
    if actual_latent_seq is not None:
        # Trim time_steps if the simulation aborted early
        actual_len = len(actual_latent_seq)
        ax2.plot(time_steps[:actual_len], actual_latent_seq[:, 0], 'b-', linewidth=1.0, label='Actual x1')
        ax2.plot(time_steps[:actual_len], actual_latent_seq[:, 1], 'r-', linewidth=1.0, label='Actual x2')
    
    ax2.set_title('Neural Latent Space Tracking')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Latent Activation')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt

def validate_latent_plan(wrapper, latent_plan: np.ndarray, target_pos_seq: np.ndarray):
    """
    Validates the latent plan by passing it open-loop through the ANN's
    muscle head, then integrating the resulting muscle activations through
    the arm model — without touching the Brain at all.

    This lets you check:
      1. Whether the planned (x1, x2) signals survive the convolutional
         muscle head and produce reasonable activations.
      2. Whether the resulting arm path approximates the target trajectory.

    Args:
        wrapper       : LatentBrainWrapper (used only to access the ANN and arm)
        latent_plan   : np.ndarray of shape (T, 2) — the (x1, x2) plan
        target_pos_seq: np.ndarray of shape (T, 2) — the intended XY path

    Returns:
        dict with keys:
            'predicted_pos'   : (T, 2) hand positions from open-loop rollout
            'muscle_acts'     : (T, 4) muscle activations
            'latent_decoded'  : (T, 2) what x1/x2 looked like after W_total projection (for sanity)
    """
    ann  = wrapper.system.ann          # _NeuralActivityToMuscleANN
    arm  = wrapper.system._arm         # _Arm

    # ── 1. Reset stateful components so we get a clean rollout ──────────
    ann.muscle_head.reset_state(batch_size=1)

    # Save and reset arm angles so the rollout starts from the same
    # position the live simulation would use
    saved_shoulder = arm._shoulder_angle
    saved_elbow    = arm._elbow_angle
    arm._shoulder_angle = 0.0
    arm._elbow_angle    = 0.0

    # ── 2. Pass the full latent sequence through the muscle head ─────────
    # muscle_head.forward() accepts (time, 2) directly — no Brain involved
    latent_tensor = torch.tensor(latent_plan, dtype=torch.float32)   # (T, 2)
    with torch.no_grad():
        muscle_acts_tensor, debug = ann.muscle_head(latent_tensor, return_debug=True)
    muscle_acts = muscle_acts_tensor.numpy()   # (T, 4)

    # ── 3. Integrate muscle activations through the arm ──────────────────
    T = len(latent_plan)
    predicted_pos = np.zeros((T, 2))

    for k in range(T):
        arm.move(*muscle_acts[k])          # updates shoulder/elbow angles
        predicted_pos[k] = arm.hand_pos   # forward kinematics → XY

    # ── 4. Restore arm state so we don't corrupt the live system ─────────
    arm._shoulder_angle = saved_shoulder
    arm._elbow_angle    = saved_elbow

    # ── 5. Diagnostic metrics ─────────────────────────────────────────────
    T_plot = min(len(predicted_pos), len(target_pos_seq))
    pos_error = np.linalg.norm(predicted_pos[:T_plot] - target_pos_seq[:T_plot], axis=1)
    rmse = np.sqrt(np.mean(pos_error**2))
    print(f"Open-loop RMSE vs target: {rmse:.3f} units")
    print(f"Mean muscle activation:   {muscle_acts.mean(axis=0).round(3)}  [sh+, sh-, el+, el-]")
    print(f"Power state range:        [{debug['power'].min():.3f}, {debug['power'].max():.3f}]")
    print(f"Selector range:           [{debug['selector'].min():.3f}, {debug['selector'].max():.3f}]")

    # ── 6. Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Path comparison
    ax = axes[0]
    ax.plot(target_pos_seq[:, 0], target_pos_seq[:, 1], 'k--', alpha=0.6, label='Target')
    ax.plot(predicted_pos[:, 0],  predicted_pos[:, 1],  'b-',  linewidth=1.5, label='Open-loop rollout')
    ax.plot(predicted_pos[0, 0],  predicted_pos[0, 1],  'go',  label='Start')
    ax.set_title('Open-loop path vs target')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.axis('equal'); ax.grid(True); ax.legend()

    # Muscle activations over time
    ax = axes[1]
    labels = ['Shoulder +', 'Shoulder −', 'Elbow +', 'Elbow −']
    for i, label in enumerate(labels):
        ax.plot(muscle_acts[:, i], label=label, linewidth=0.8)
    ax.set_title('Muscle activations (open-loop)')
    ax.set_xlabel('Time steps'); ax.set_ylabel('Activation [0, 1]')
    ax.legend(); ax.grid(True)

    # Power and selector internals — key diagnostic
    ax = axes[2]
    power_np    = debug['power'].numpy()       # (T,)
    selector_np = debug['selector'].numpy()    # (T, 4)
    ax.plot(power_np, 'k-', linewidth=1.5, label='Power gate')
    for i in range(4):
        ax.plot(selector_np[:, i], linewidth=0.8, alpha=0.7, label=f'Selector {i}')
    ax.set_title('Muscle head internals')
    ax.set_xlabel('Time steps'); ax.set_ylabel('Value')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'predicted_pos':  predicted_pos,
        'muscle_acts':    muscle_acts,
        'debug':          {k: v.numpy() for k, v in debug.items()},
    }