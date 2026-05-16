import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.decomposition import PCA
from pykalman import KalmanFilter
from hmmlearn import hmm
import json

class Explorer:
    """
    An experiment-tracking Explorer for neural population dynamics.
    Features the ability to learn dynamics via EM and push them to the Simulator.
    """
    def __init__(self, baseline_observation: np.ndarray):
        self.experiments = {
            'Baseline (Real Data)': self._create_experiment_dict(baseline_observation)
        }
        self.active_exp_name = 'Baseline (Real Data)'
        self.simulator = None
        self.SimulatorClass = None
        self.control_functions = {}

    def _create_experiment_dict(self, data: np.ndarray) -> dict:
        return {
            'data': data,
            'trial_cnt': data.shape[0],
            'timestep_cnt': data.shape[1],
            'neuron_cnt': data.shape[2],
            'kf_model': None,
            'smoothed_states': None,
            'hmm_model': None,
            'discrete_states': None
        }

    def get_active(self, key: str):
        return self.experiments[self.active_exp_name][key]

    def set_active(self, key: str, value):
        self.experiments[self.active_exp_name][key] = value

    def attach_simulator(self, simulator_instance, simulator_class, control_functions: dict):
        self.simulator = simulator_instance
        self.SimulatorClass = simulator_class
        self.control_functions = control_functions

    def fit_lds_em(self, state_dim: int = 2, n_iter: int = 10):
        data = self.get_active('data')
        neuron_cnt = self.get_active('neuron_cnt')
        trial_cnt = self.get_active('trial_cnt')
        timestep_cnt = self.get_active('timestep_cnt')
        
        flat_data = data.reshape(-1, neuron_cnt)
        kf_model = KalmanFilter(n_dim_state=state_dim, n_dim_obs=neuron_cnt)
        kf_model = kf_model.em(flat_data, n_iter=n_iter)
        
        smoothed_states = np.zeros((trial_cnt, timestep_cnt, state_dim))
        for i in range(trial_cnt):
            smoothed_states[i], _ = kf_model.smooth(data[i])
            
        self.set_active('kf_model', kf_model)
        self.set_active('smoothed_states', smoothed_states)

    def fit_hmm(self, n_states: int = 3):
        data = self.get_active('data')
        smoothed_states = self.get_active('smoothed_states')
        
        if smoothed_states is not None:
            latent_data = smoothed_states.reshape(-1, smoothed_states.shape[-1])
        else:
            pca = PCA(n_components=3)
            latent_data = pca.fit_transform(data.reshape(-1, self.get_active('neuron_cnt')))
            
        hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
        hmm_model.fit(latent_data)
        discrete_states = hmm_model.predict(latent_data).reshape(self.get_active('trial_cnt'), self.get_active('timestep_cnt'))
        
        self.set_active('hmm_model', hmm_model)
        self.set_active('discrete_states', discrete_states)

    def interactive_dashboard(self):
        # --- TAB 1: ANALYSIS UI ---
        exp_dropdown = widgets.Dropdown(
            options=list(self.experiments.keys()), 
            value=self.active_exp_name, 
            description='Experiment:', 
            layout=widgets.Layout(width='400px')
        )
        view_dropdown = widgets.Dropdown(
            options=['1. Single Neuron Raw Trace', '2. Population Heatmap', '3. PCA Trajectory', '4. EM Smoothed State (LDS)', '5. HMM Transitions'],
            value='3. PCA Trajectory', description='Select View:', layout=widgets.Layout(width='400px')
        )
        trial_slider = widgets.IntSlider(min=0, max=self.get_active('trial_cnt') - 1, step=1, value=0, description='Trial ID:')
        plot_output = widgets.Output()

        def update_plot(change=None):
            self.active_exp_name = exp_dropdown.value
            trial_slider.max = self.get_active('trial_cnt') - 1
            
            with plot_output:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                v = view_dropdown.value
                tidx = trial_slider.value
                data = self.get_active('data')
                
                if '1.' in v:
                    ax.plot(data[tidx, :, 0], color='black')
                    ax.set_title(f"Raw Activity (Neuron 0) | {self.active_exp_name}")
                elif '2.' in v:
                    im = ax.imshow(data[tidx].T, aspect='auto', cmap='viridis')
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"Heatmap (Trial {tidx}) | {self.active_exp_name}")
                elif '3.' in v:
                    pca_result = PCA(n_components=2).fit_transform(data.reshape(-1, self.get_active('neuron_cnt')))
                    traj = pca_result.reshape(self.get_active('trial_cnt'), self.get_active('timestep_cnt'), 2)[tidx]
                    ax.plot(traj[:, 0], traj[:, 1], marker='o')
                    ax.set_title(f"PCA Trajectory | {self.active_exp_name}")
                elif '4.' in v:
                    smoothed = self.get_active('smoothed_states')
                    if smoothed is not None:
                        ax.plot(smoothed[tidx, :, 0], smoothed[tidx, :, 1], marker='o', color='teal')
                        ax.set_title(f"LDS Dynamics | {self.active_exp_name}")
                    else: ax.text(0.5, 0.5, "Run explorer.fit_lds_em() for this experiment first!", color='red')
                elif '5.' in v:
                    discrete = self.get_active('discrete_states')
                    if discrete is not None:
                        smoothed = self.get_active('smoothed_states')
                        traj = smoothed[tidx, :, :2] if smoothed is not None else PCA(n_components=2).fit_transform(data.reshape(-1, self.get_active('neuron_cnt'))).reshape(self.get_active('trial_cnt'), self.get_active('timestep_cnt'), 2)[tidx]
                        ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.5)
                        ax.scatter(traj[:, 0], traj[:, 1], c=discrete[tidx], cmap='Set1', s=50)
                        ax.set_title(f"HMM Phases | {self.active_exp_name}")
                    else: ax.text(0.5, 0.5, "Run explorer.fit_hmm() for this experiment first!", color='red')
                
                plt.tight_layout()
                plt.show()

        exp_dropdown.observe(update_plot, names='value')
        view_dropdown.observe(update_plot, names='value')
        trial_slider.observe(update_plot, names='value')
        update_plot()
        
        analysis_ui = widgets.VBox([
            widgets.HBox([exp_dropdown]),
            widgets.HBox([view_dropdown, trial_slider]), 
            plot_output
        ])

        # --- TAB 2: SIMULATION STUDIO ---
        if getattr(self, 'simulator', None) is not None:
            txt_layout = widgets.Layout(width='300px', height='120px')
            A_txt = widgets.Textarea(value=json.dumps(self.simulator.A.tolist(), indent=2), description='A Matrix:', layout=txt_layout)
            B_txt = widgets.Textarea(value=json.dumps(self.simulator.B.tolist(), indent=2), description='B Matrix:', layout=txt_layout)
            C_txt = widgets.Textarea(value=json.dumps(self.simulator.C.tolist(), indent=2), description='C Matrix:', layout=txt_layout)
            Q_txt = widgets.Textarea(value=json.dumps(self.simulator.Q.tolist(), indent=2), description='Q (Process Noise):', layout=txt_layout)
            R_txt = widgets.Textarea(value=json.dumps(self.simulator.R.tolist(), indent=2), description='R (Obs Noise):', layout=txt_layout)
            
            trials_in = widgets.IntText(value=10, description='Trials:')
            len_in = widgets.IntText(value=100, description='Length:')
            seed_in = widgets.Text(value='[0, 0]', description='Seed:')
            ctrl_drop = widgets.Dropdown(options=list(self.control_functions.keys()), description='Control Input:')
            
            exp_name_in = widgets.Text(value='Sim: Learned Dynamics', description='Save Name:', layout=widgets.Layout(width='300px'))
            
            # --- THE NEW DATA PULL MECHANISM ---
            pull_btn = widgets.Button(description='Pull Learned Matrices', button_style='info')
            sim_btn = widgets.Button(description='Run & Save Simulation', button_style='success')
            sim_output = widgets.Output()

            def pull_learned_matrices(b):
                with sim_output:
                    clear_output()
                    kf = self.get_active('kf_model')
                    if kf is not None:
                        # Extract the fitted parameters and round them so the UI remains readable
                        A_txt.value = json.dumps(np.round(kf.transition_matrices, 4).tolist(), indent=2)
                        C_txt.value = json.dumps(np.round(kf.observation_matrices, 4).tolist(), indent=2)
                        Q_txt.value = json.dumps(np.round(kf.transition_covariance, 4).tolist(), indent=2)
                        R_txt.value = json.dumps(np.round(kf.observation_covariance, 4).tolist(), indent=2)
                        
                        # Generate a zero B-matrix matching the learned latent dimensions and assumed control width
                        state_dim = kf.transition_matrices.shape[0]
                        dummy_B = np.zeros((state_dim, 2)).tolist() 
                        B_txt.value = json.dumps(dummy_B, indent=2)
                        
                        # Set a default seed matching the latent dimension size
                        seed_in.value = str([0] * state_dim)
                        
                        print(f"Successfully pulled matrices from '{self.active_exp_name}' LDS fit!")
                    else:
                        print("No LDS model found. Run fit_lds_em() on the active experiment first.")

            def run_sim(b):
                with sim_output:
                    clear_output()
                    try:
                        A, B, C = json.loads(A_txt.value), json.loads(B_txt.value), json.loads(C_txt.value)
                        Q, R = json.loads(Q_txt.value), json.loads(R_txt.value)
                        seed = json.loads(seed_in.value)
                        
                        new_sim = self.SimulatorClass([A, B, C, Q, R])
                        new_data = new_sim.generate_data(
                            trials=trials_in.value, lengths=len_in.value, 
                            seed=seed, control_function=self.control_functions[ctrl_drop.value]
                        )
                        
                        new_exp_name = exp_name_in.value
                        self.experiments[new_exp_name] = self._create_experiment_dict(new_data)
                        exp_dropdown.options = list(self.experiments.keys())
                        exp_dropdown.value = new_exp_name 
                        
                        print(f"Saved as '{new_exp_name}'. Switch to Analysis tab to view!")
                        
                    except Exception as e:
                        print(f"Simulation Failed. Error: {e}")

            pull_btn.on_click(pull_learned_matrices)
            sim_btn.on_click(run_sim)
            
            row1 = widgets.HBox([A_txt, B_txt, C_txt])
            row2 = widgets.HBox([Q_txt, R_txt])
            row3 = widgets.HBox([trials_in, len_in, seed_in, ctrl_drop])
            row4 = widgets.HBox([pull_btn, exp_name_in, sim_btn])
            simulation_ui = widgets.VBox([row1, row2, row3, row4, sim_output])
            
            tabs = widgets.Tab(children=[analysis_ui, simulation_ui])
            tabs.set_title(0, 'Analysis Engine')
            tabs.set_title(1, 'Simulation Studio')
            display(tabs)
        else:
            display(analysis_ui)