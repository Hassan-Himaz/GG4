#can improve performance in number of ways if required 
#add controlability and observability things
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
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
        for time in range(length-1):
            latent_state = np.matmul(self.A,latent_state) + np.matmul(self.B,control_function(time,data))+ rng.multivariate_normal(np.zeros(self.x_dimensions),self.Q)
            observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
            data.append(observed_state)
        return np.array(data)
    
    def get_controllability_gramian(self) -> np.ndarray:
        """
        Calculates the discrete-time Controllability Gramian.
        Solves: W_c = A W_c A^T + B B^T
        """
        return la.solve_discrete_lyapunov(self.A, self.B @ self.B.T)

    def get_observability_gramian(self) -> np.ndarray:
        """
        Calculates the discrete-time Observability Gramian.
        Solves: W_o = A^T W_o A + C^T C
        """
        return la.solve_discrete_lyapunov(self.A.T, self.C.T @ self.C)
    
    def plot_phase_portrait(self, scale: float = 2.0):
        """
        Plots the autonomous phase portrait (vector field and eigenvectors)
        derived directly from the A matrix. Supports 2D and 3D systems.
        """
        dim = self.x_dimensions
        
        if dim not in [2, 3]:
            print(f"Phase portraits are visually defined for 2D and 3D systems. Your system is {dim}D.")
            return

        fig = plt.figure(figsize=(10, 8))
        
        if dim == 2:
            ax = fig.add_subplot(111)
            
            # 2D Grid
            x_vals = np.linspace(-scale, scale, 25)
            y_vals = np.linspace(-scale, scale, 25)
            X1, X2 = np.meshgrid(x_vals, y_vals)
            
            # Discrete flow field
            dX1 = (self.A[0, 0] - 1) * X1 + self.A[0, 1] * X2
            dX2 = self.A[1, 0] * X1 + (self.A[1, 1] - 1) * X2
            
            ax.quiver(X1, X2, dX1, dX2, color='gray', alpha=0.6)
            
            eigvals, eigvecs = np.linalg.eig(self.A)
            for i in range(2):
                if np.isreal(eigvals[i]):
                    vec = np.real(eigvecs[:, i])
                    ax.plot([-vec[0] * scale * 2, vec[0] * scale * 2], 
                            [-vec[1] * scale * 2, vec[1] * scale * 2], 
                            '--', alpha=0.8, linewidth=2.5,
                            label=f'Eigenvector (λ={np.real(eigvals[i]):.3f})')
            
            ax.plot(0, 0, 'ro', markersize=8, label='Equilibrium (0,0)')
            ax.set_xlabel("State 0")
            ax.set_ylabel("State 1")
            ax.set_title("2D Autonomous Phase Portrait")
            
        elif dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            # 3D Grid (Kept sparse to prevent visual clutter)
            pts = 8
            vals = np.linspace(-scale, scale, pts)
            X1, X2, X3 = np.meshgrid(vals, vals, vals)
            
            # Discrete flow field
            dX1 = (self.A[0, 0] - 1) * X1 + self.A[0, 1] * X2 + self.A[0, 2] * X3
            dX2 = self.A[1, 0] * X1 + (self.A[1, 1] - 1) * X2 + self.A[1, 2] * X3
            dX3 = self.A[2, 0] * X1 + self.A[2, 1] * X2 + (self.A[2, 2] - 1) * X3
            
            # Normalize arrows so they don't overlap aggressively in 3D
            ax.quiver(X1, X2, X3, dX1, dX2, dX3, color='gray', alpha=0.4, length=scale*0.2, normalize=True)
            
            eigvals, eigvecs = np.linalg.eig(self.A)
            for i in range(3):
                if np.isreal(eigvals[i]):
                    vec = np.real(eigvecs[:, i])
                    ax.plot([-vec[0] * scale * 2, vec[0] * scale * 2], 
                            [-vec[1] * scale * 2, vec[1] * scale * 2], 
                            [-vec[2] * scale * 2, vec[2] * scale * 2], 
                            '--', alpha=0.9, linewidth=3,
                            label=f'Eigenvector (λ={np.real(eigvals[i]):.3f})')
                            
            ax.plot([0], [0], [0], 'ro', markersize=8, label='Equilibrium (0,0,0)')
            ax.set_xlabel("State 0")
            ax.set_ylabel("State 1")
            ax.set_zlabel("State 2")
            ax.set_title("3D Autonomous Phase Portrait")

        ax.grid(True, alpha=0.3)
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        if dim == 3:
            ax.set_zlim([-scale, scale])
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()

    def plot_gramian_directions(self):
        """
        Calculates and plots the Controllability and Observability Gramians.
        Supports 2D (Ellipses) and 3D (Wireframe Ellipsoids).
        """
        import scipy.linalg as la
        
        dim = self.x_dimensions
        if dim not in [2, 3]:
            print(f"Gramian visualization is explicitly set up for 2D and 3D systems. Your system is {dim}D.")
            return
            
        # 1. Calculate the Gramians
        W_c = la.solve_discrete_lyapunov(self.A, self.B @ self.B.T)
        W_o = la.solve_discrete_lyapunov(self.A.T, self.C.T @ self.C)
        
        # 2. Setup Plot
        fig = plt.figure(figsize=(14, 6))
        
        # Conditionally create 3D axes
        proj = '3d' if dim == 3 else None
        ax1 = fig.add_subplot(121, projection=proj)
        ax2 = fig.add_subplot(122, projection=proj)
        
        def plot_energy_bounds(ax, W, title, is_controllability=True):
            evals, evecs = np.linalg.eigh(W)
            
            if dim == 2:
                theta = np.linspace(0, 2*np.pi, 100)
                circle = np.vstack((np.cos(theta), np.sin(theta)))
                transform = evecs @ np.diag(np.sqrt(np.abs(evals)))
                ellipse = transform @ circle
                
                ax.plot(ellipse[0, :], ellipse[1, :], 'k-', alpha=0.3)
                
                colors = ['tab:red', 'tab:green']
                for i in range(2):
                    vec = evecs[:, i]
                    val = evals[i]
                    length = np.sqrt(np.abs(val))
                    
                    if is_controllability:
                        label = "Easiest" if val == max(evals) else "Hardest"
                    else:
                        label = "Most Observable" if val == max(evals) else "Least Observable"
                        
                    ax.quiver(0, 0, vec[0]*length, vec[1]*length, angles='xy', scale_units='xy', scale=1, 
                              color=colors[i], width=0.015, label=f'{label} (λ={val:.2f})')
                ax.axis('equal')
                
            elif dim == 3:
                # Generate parametric sphere
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones_like(u), np.cos(v))
                sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))
                
                # Stretch into ellipsoid
                transform = evecs @ np.diag(np.sqrt(np.abs(evals)))
                ellipsoid = transform @ sphere
                
                X = ellipsoid[0, :].reshape(x.shape)
                Y = ellipsoid[1, :].reshape(y.shape)
                Z = ellipsoid[2, :].reshape(z.shape)
                
                ax.plot_wireframe(X, Y, Z, color='k', alpha=0.15)
                
                colors = ['tab:red', 'tab:blue', 'tab:green']
                for i in range(3):
                    vec = evecs[:, i]
                    val = evals[i]
                    length = np.sqrt(np.abs(val))
                    
                    if is_controllability:
                        if val == max(evals): label = "Easiest"
                        elif val == min(evals): label = "Hardest"
                        else: label = "Intermediate"
                    else:
                        if val == max(evals): label = "Most Observable"
                        elif val == min(evals): label = "Least Observable"
                        else: label = "Intermediate"
                        
                    ax.quiver(0, 0, 0, vec[0]*length, vec[1]*length, vec[2]*length, 
                              color=colors[i], label=f'{label} (λ={val:.2f})', linewidth=3)
                
                # Fix aspect ratio for 3D by setting limits to the max radius
                max_radius = np.max(np.sqrt(np.abs(evals))) * 1.1
                ax.set_xlim([-max_radius, max_radius])
                ax.set_ylim([-max_radius, max_radius])
                ax.set_zlim([-max_radius, max_radius])
                ax.set_zlabel("State 2")

            ax.set_title(title)
            ax.set_xlabel("State 0")
            ax.set_ylabel("State 1")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize='small')

        # 3. Render both plots
        plot_energy_bounds(ax1, W_c, "Controllability\n(Unit Energy Reachability)", is_controllability=True)
        plot_energy_bounds(ax2, W_o, "Observability\n(Unit Initial State Energy)", is_controllability=False)
        
        plt.tight_layout()
        plt.show()
        
    def check_controllability(self) -> bool:
        """
        Checks if the system is controllable by ensuring the Kalman 
        Controllability Matrix has full rank (rank == n).
        """
        n = self.x_dimensions
        
        # Initialize the controllability matrix with B
        controllability_matrix = self.B
        current_term = self.B
        
        # Horizontally stack A^k * B
        for _ in range(1, n):
            current_term = self.A @ current_term
            controllability_matrix = np.hstack((controllability_matrix, current_term))

        # Check the rank
        rank = np.linalg.matrix_rank(controllability_matrix)
        is_controllable = (rank == n)
        
        print(f"Controllability Matrix Rank: {rank}/{n} -> {'Controllable' if is_controllable else 'Uncontrollable'}")
        return is_controllable

    def check_observability(self) -> bool:
        """
        Checks if the system is observable by ensuring the Kalman 
        Observability Matrix has full rank (rank == n).
        """
        n = self.x_dimensions
        
        # Initialize the observability matrix with C
        observability_matrix = self.C
        current_term = self.C
        
        # Vertically stack C * A^k
        for _ in range(1, n):
            current_term = current_term @ self.A
            observability_matrix = np.vstack((observability_matrix, current_term))

        # Check the rank
        rank = np.linalg.matrix_rank(observability_matrix)
        is_observable = (rank == n)
        
        print(f"Observability Matrix Rank: {rank}/{n} -> {'Observable' if is_observable else 'Unobservable'}")
        return is_observable
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