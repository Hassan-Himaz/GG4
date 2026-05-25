from cProfile import label


from Illustrator import Illustrator
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from typing import Callable
from Controllers import Controllers
from LDSParams import LDSParams


class Simulator:

    """
    A class to simulate neural activity data.
    Initialise with list or matrix parameters: A,B,C,Q,R
    """

    def __init__(self,parameters:LDSParams|None,illustrator: Illustrator,controller:Controllers):
        """
        Initialize the Model with set Parameters
        Accepts a 3D numpy array of shape (A,B,C,Q,R,mu_0,P_0) and stores it for data generation
        Parameters:
            parameters (np.ndarray): 3D numpy array of shape (A,B,C,Q,R,mu_0,P_0)  - these need to be correct else will throw error
        """
        # if controller is None:
        #     defaul
        #     self.controller = Controllers
        

        if parameters is not None:
            self.A = parameters.A    # using the dataclass
            self.B = parameters.B
            self.C = parameters.C
            self.Q = parameters.Q
            self.R = parameters.R
            self.mu_0 = parameters.mu_0
            self.P_0 = parameters.P_0
        else:
            self.A = np.eye(2)    # default
            self.B = np.eye(2)
            self.C = np.eye(2)
            self.Q = np.eye(2)
            self.R = np.eye(2)
            self.mu_0 = [0,0]
            self.P_0 = np.eye(2)


        self.x_dimensions = len(self.Q)
        self.y_dimensions = len(self.R)
        self.illustrator = illustrator
        self.trial_cnt = self.illustrator.trial_cnt
        self.timestep_cnt = self.illustrator.timestep_cnt
        self.neuron_cnt = self.illustrator.neuron_cnt
        self.observation = self.illustrator.observation
        self.controller = controller   #input
    

    def generate_data(self, trials: int, lengths: int, seed: tuple[np.ndarray,np.ndarray], input_function: Callable):
        '''Generates the data, given set number of trials, lenghts of times, seed for x, and a provided control function'''
        dataset = []
        for trial in range(trials):
            dataset.append(self._generate_trial(length=lengths, seed = seed, input_function=self.controller))
        return np.array(dataset)
    
    def _generate_trial(self,
                         length:int, #
                         seed:tuple[np.ndarray,np.ndarray], #  seed the inital hidden state value with the infered starting mean - seed should be tuple(m_0,P_0)
                         input_function: Callable,
                         return_state=False):
        '''
        will return timepoint x neuron dimensional array  for a single simulated trial
        
        '''

        assert seed[0].ndim ==1 and seed[1].shape == (self.x_dimensions,self.x_dimensions)
        #intial latent state
        latent_state = rng.multivariate_normal(seed[0],seed[1])
        observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
        data, states = [observed_state], [latent_state]
        for time in range(length-1):
            latent_state = np.matmul(self.A,latent_state) + np.matmul(self.B,input_function(time,data))+ rng.multivariate_normal(np.zeros(self.x_dimensions),self.Q)
            observed_state = np.matmul(self.C,latent_state) + rng.multivariate_normal(np.zeros(self.y_dimensions),self.R)
            data.append(observed_state)
            states.append(latent_state)
        data = np.asarray(data)
        states = np.asarray(states)
            
        return (data, states) if return_state else data
    
    def compare_plot(self,seed):
        '''
        method that will plot real trial next to generated trial

        seed should be tuple of (mu_0, P_0)
        
        '''
        print('compare plot')

        
        
        real_data = self.observation[0] # choose to compare with trial 1
        print(real_data)
        simulated_data = self._generate_trial(length = self.timestep_cnt,seed = seed, input_function = self.input_pulse)
        time = np.arange(simulated_data.shape[0])

        plt.figure(1)
        for neuron in range(simulated_data.shape[1]):
            plt.plot(time,real_data[:,neuron],label = 'real neuron, neuron number{}'.format(neuron))
            plt.plot(time,simulated_data[:,neuron],label = 'simulated neuron, neuron number{}'.format(neuron))
        plt.xlabel('time')
        plt.ylabel('neuron activation')
        plt.legend()
        plt.show()

    # ------------------------------------------------------------------
    # Gramians
    # ------------------------------------------------------------------

    def finite_controllability_gramian(self, horizon=20):
        Wc = np.zeros((self.x_dimensions, self.x_dimensions))
        A_power = np.eye(self.x_dimensions)
        for _ in range(horizon):
            Wc += A_power @ self.B @ self.B.T @ A_power.T
            A_power = A_power @ self.A
        return Wc

    def finite_observability_gramian(self, horizon=20):
        Wo = np.zeros((self.x_dimensions, self.x_dimensions))
        A_power = np.eye(self.x_dimensions)
        for _ in range(horizon):
            Wo += A_power.T @ self.C.T @ self.C @ A_power
            A_power = A_power @ self.A
        return Wo

    def gramian_summary(self, horizon=20):
        Wc = self.finite_controllability_gramian(horizon)
        Wo = self.finite_observability_gramian(horizon)
        eig_Wc = np.linalg.eigvalsh(Wc)
        eig_Wo = np.linalg.eigvalsh(Wo)
        return {
            "Wc": Wc, "Wo": Wo,
            "eig_Wc": eig_Wc, "eig_Wo": eig_Wo,
            "min_eig_Wc": np.min(eig_Wc), "max_eig_Wc": np.max(eig_Wc),
            "min_eig_Wo": np.min(eig_Wo), "max_eig_Wo": np.max(eig_Wo),
        }

    def plot_gramians(self, horizon=20):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, M, title in zip(
            axes,
            [self.finite_controllability_gramian(horizon),
             self.finite_observability_gramian(horizon)],
            [f"Controllability Gramian (h={horizon})",
             f"Observability Gramian (h={horizon})"],
        ):
            im = ax.imshow(M, aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        plt.show()

    def input_pulse(self,
                    time:int,
                    data:np.ndarray,
                    )-> np.ndarray:
        t_on = 0
        t_off = 100
        amplitude = 10
        num_inputs = 2
        if t_on <= time < t_off:
            return np.full(num_inputs,amplitude)
        else: 
            return np.zeros(num_inputs)
        
    @staticmethod
    def load_latest_dump(folder: str = "trial_dumps"):
        import os, pandas as pd

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"folder '{folder}' does not exist — save a trial first")

        files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
        if not files:
            raise FileNotFoundError(f"no CSVs in {folder}")

        path = os.path.join(folder, files[-1])

        params = {}
        skip = 0
        with open(path) as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    if line.startswith("#"):
                        key, _, val = line.lstrip("# ").partition(":")
                        if key.strip():
                            params[key.strip()] = val.strip()
                    skip += 1
                else:
                    break
        params["_filename"] = files[-1]

        df = pd.read_csv(path, skiprows=skip)
        return params, df            

    def _open_comparison_window(self):
            """Pop-up window for running Illustrator methods on real and saved sim data.
                to be used in explore
            
            """
            import tkinter as tk
            import os

            # check there's actually a dump to compare against
            if not os.path.isdir("trial_dumps") or not os.listdir("trial_dumps"):
                print("no dumps in trial_dumps/ — click 'Save trial to CSV' first")
                return

            win = tk.Toplevel()
            win.title("Compare real vs simulated")

            tk.Label(win, text="Illustrator method:").pack(anchor="w", padx=8, pady=(8, 2))

            methods = [m for m in dir(self.illustrator)
                    if not m.startswith("_")
                    and callable(getattr(self.illustrator, m))]
            method_var = tk.StringVar(value=methods[0] if methods else "")
            tk.OptionMenu(win, method_var, *methods).pack(fill=tk.X, padx=8)

            info = tk.Label(win, text="", fg="gray", justify="left", anchor="w",
                            wraplength=300)

            def run_comparison():
                method = method_var.get()

                # load latest dump
                try:
                    params, df = Simulator.load_latest_dump()
                except Exception as e:
                    info.config(text=f"load err: {e}"); return
                sim_cols = sorted(c for c in df.columns if c.startswith("sim_"))
                sim_obs = df[sim_cols].to_numpy()[np.newaxis, ...]

                # run method on real, then on sim — swap observation in/out
                original = self.illustrator.observation
                try:
                    self.illustrator.observation = self.observation
                    getattr(self.illustrator, method)()
                    self.illustrator.observation = sim_obs
                    getattr(self.illustrator, method)()
                    latest = sorted(os.listdir("trial_dumps"))[-1]
                    param_lines = "\n".join(f"  {k}: {v}" for k, v in params.items()
                                            if k.startswith("input"))
                    info.config(text=f"ran {method} on real and on:\n{latest}\n{param_lines}")
                except Exception as e:
                    info.config(text=f"viz err: {e}")
                finally:
                    self.illustrator.observation = original

            tk.Button(win, text="Compare", command=run_comparison).pack(fill=tk.X, padx=8, pady=4)
            info.pack(fill=tk.X, padx=8, pady=(4, 8))


    #graphics pop-up window
    def explore(self):

    

        """Live parameter tweaking with a tkinter panel."""
        import tkinter as tk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title("LDS parameter explorer")

        controls = tk.Frame(root)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        fig = Figure(figsize=(7, 4))
        ax_obs = fig.add_subplot(111)
        ax_obs.set_ylabel("activation")
        ax_obs.set_xlabel("time")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        entries = {}
        self._controllers = Controllers()    # if not already on self

        # --- dimension selector ---
        dim_frame = tk.LabelFrame(controls, text="dimensions", padx=4, pady=2)
        dim_frame.pack(fill=tk.X, pady=3)

        tk.Label(dim_frame, text="hidden states (x_dim)").grid(row=0, column=0, sticky="w")
        x_dim_var = tk.IntVar(value=self.x_dimensions)
        tk.Spinbox(dim_frame, from_=1, to=5, width=4, textvariable=x_dim_var,
                command=lambda: resize_state(x_dim_var.get())).grid(row=0, column=1, padx=4)
        
        tk.Label(dim_frame, text="inputs (u_dim)").grid(row=1, column=0, sticky="w")
        u_dim_var = tk.IntVar(value=self.B.shape[1])
        tk.Spinbox(dim_frame, from_=1, to=5, width=4, textvariable=u_dim_var,
                command=lambda: resize_inputs(u_dim_var.get())).grid(row=1, column=1, padx=4)
        
        #spectra block setup
        spectra = tk.Frame(root)
        spectra.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        tk.Label(spectra, text="spectra", font=("", 9, "bold")).pack(anchor="w", pady=(0, 4))

        spectra_labels = {}
        for name in ["A", "B", "C", "Q", "R", "P_0"]:
            frame = tk.LabelFrame(spectra, text=name, padx=4, pady=2)
            frame.pack(fill=tk.X, pady=3)
            lbl = tk.Label(frame, text="", justify="left", font=("Courier", 9),
                        anchor="w", width=28)
            lbl.pack(anchor="w")
            spectra_labels[name] = lbl

        #neuron toggle block setup
        # --- plot controls column ---
        plot_ctrl = tk.Frame(root)
        plot_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        tk.Label(plot_ctrl, text="plot controls", font=("", 9, "bold")).pack(anchor="w", pady=(0, 4))

        import matplotlib.cm as _cm
        _N_COLORS = 20
        NEURON_COLORS = [_cm.tab20(i / _N_COLORS) for i in range(_N_COLORS)]

        def _hex(c):
            return "#{:02x}{:02x}{:02x}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255))

        # --- neuron toggles (scrollable, split real / sim) ---
        neuron_outer = tk.LabelFrame(plot_ctrl, text="neurons", padx=4, pady=2)
        neuron_outer.pack(fill=tk.X, pady=(8, 4))

        n_scroll = tk.Canvas(neuron_outer, width=120, height=220, highlightthickness=0)
        n_sb = tk.Scrollbar(neuron_outer, orient="vertical", command=n_scroll.yview)
        n_scroll.configure(yscrollcommand=n_sb.set)
        n_sb.pack(side=tk.RIGHT, fill=tk.Y)
        n_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        n_inner = tk.Frame(n_scroll)
        n_scroll.create_window((0, 0), window=n_inner, anchor="nw")
        n_inner.bind("<Configure>", lambda e: n_scroll.configure(scrollregion=n_scroll.bbox("all")))

        real_neuron_frame = tk.LabelFrame(n_inner, text="real", padx=2, pady=2)
        real_neuron_frame.pack(fill=tk.X, padx=2, pady=(2, 0))
        sim_neuron_frame  = tk.LabelFrame(n_inner, text="simulated", padx=2, pady=2)
        sim_neuron_frame.pack(fill=tk.X, padx=2, pady=(4, 2))

        real_neuron_vars = {}
        sim_neuron_vars  = {}

        def _set_all_neurons(d, v):
            for var in d.values():
                var.set(v)
            redraw()

        def build_neuron_checkboxes():
            for child in real_neuron_frame.winfo_children():
                child.destroy()
            for child in sim_neuron_frame.winfo_children():
                child.destroy()
            real_neuron_vars.clear()
            sim_neuron_vars.clear()

            n_neurons = self.observation.shape[-1]
            for i in range(n_neurons):
                col = _hex(NEURON_COLORS[i % _N_COLORS])
                rv = tk.BooleanVar(value=True)
                real_neuron_vars[i] = rv
                tk.Checkbutton(real_neuron_frame, text=f"n{i}", variable=rv,
                               fg=col, selectcolor="black",
                               command=lambda: redraw()).grid(row=i // 2, column=i % 2, sticky="w")
                sv = tk.BooleanVar(value=True)
                sim_neuron_vars[i] = sv
                tk.Checkbutton(sim_neuron_frame, text=f"n{i}", variable=sv,
                               fg=col, selectcolor="black",
                               command=lambda: redraw()).grid(row=i // 2, column=i % 2, sticky="w")

            r_btns = tk.Frame(n_inner)
            r_btns.pack(fill=tk.X, padx=2, pady=(0, 1))
            tk.Button(r_btns, text="r:all",  command=lambda: _set_all_neurons(real_neuron_vars, True )).pack(side=tk.LEFT, expand=True, fill=tk.X)
            tk.Button(r_btns, text="r:none", command=lambda: _set_all_neurons(real_neuron_vars, False)).pack(side=tk.LEFT, expand=True, fill=tk.X)

            s_btns = tk.Frame(n_inner)
            s_btns.pack(fill=tk.X, padx=2, pady=(0, 2))
            tk.Button(s_btns, text="s:all",  command=lambda: _set_all_neurons(sim_neuron_vars, True )).pack(side=tk.LEFT, expand=True, fill=tk.X)
            tk.Button(s_btns, text="s:none", command=lambda: _set_all_neurons(sim_neuron_vars, False)).pack(side=tk.LEFT, expand=True, fill=tk.X)

        build_neuron_checkboxes()

        #states and inputs checkboxes
        input_plot_frame = tk.LabelFrame(plot_ctrl, text="inputs", padx=4, pady=2)
        input_plot_frame.pack(fill=tk.X, pady=4)
        input_vars = {}

        state_plot_frame = tk.LabelFrame(plot_ctrl, text="states", padx=4, pady=2)
        state_plot_frame.pack(fill=tk.X, pady=4)
        state_vars = {}

        def build_input_checkboxes():
            for child in input_plot_frame.winfo_children():
                child.destroy()
            input_vars.clear()
            for i in range(self.B.shape[1]):
                var = tk.BooleanVar(value=True)
                input_vars[i] = var
                tk.Checkbutton(input_plot_frame, text=f"input {i}", variable=var,
                            command=lambda: redraw()).pack(anchor="w")

        def build_state_checkboxes():
            for child in state_plot_frame.winfo_children():
                child.destroy()
            state_vars.clear()
            for i in range(self.x_dimensions):
                var = tk.BooleanVar(value=True)
                state_vars[i] = var
                tk.Checkbutton(state_plot_frame, text=f"state {i}", variable=var,
                            command=lambda: redraw()).pack(anchor="w")

        build_input_checkboxes()
        build_state_checkboxes()

        # --- matrix panels (rebuild-able) ---
        panel_container = tk.Frame(controls)
        panel_container.pack(fill=tk.X)

        def panel(name, value):
            arr = np.atleast_2d(value)
            frame = tk.LabelFrame(panel_container, text=name, padx=4, pady=2)
            frame.pack(fill=tk.X, pady=3)
            grid = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(arr.shape[1]):
                    e = tk.Entry(frame, width=7, justify="right")
                    e.insert(0, f"{arr[i, j]:.4g}")
                    e.grid(row=i, column=j, padx=1, pady=1)
                    e.bind("<Return>", lambda *_: redraw())
                    row.append(e)
                grid.append(row)
            entries[name] = grid

        def build_panels():
            """(Re)create all matrix panels from current self.* values."""
            for child in panel_container.winfo_children():
                child.destroy()
            entries.clear()
            for name, val in [("A", self.A), ("B", self.B), ("C", self.C),
                            ("Q", self.Q), ("R", self.R),
                            ("mu_0", self.mu_0.reshape(1, -1)), ("P_0", self.P_0)]:
                panel(name, val)

        def resize_state(new_dim: int):
            """Reshape A, B, Q, mu_0, P_0, and C's columns to new_dim. Preserve overlap; fill new entries sensibly."""
            old = self.x_dimensions
            if new_dim == old:
                return

            def resize(M, shape, fill=0.0, identity_diag=False):
                out = np.full(shape, fill)
                if identity_diag:
                    np.fill_diagonal(out, 0.9 if shape[0] == shape[1] else 1.0)
                r = min(M.shape[0], shape[0])
                c = min(M.shape[1], shape[1]) if M.ndim == 2 else 0
                if M.ndim == 1:
                    out[:r] = M[:r]
                else:
                    out[:r, :c] = M[:r, :c]
                return out

            n_in = self.B.shape[1]
            y    = self.y_dimensions

            self.A    = resize(self.A,   (new_dim, new_dim), identity_diag=True)
            self.B    = resize(self.B,   (new_dim, n_in))
            self.C    = resize(self.C,   (y, new_dim))
            self.Q    = resize(self.Q,   (new_dim, new_dim), identity_diag=True)
            self.P_0  = resize(self.P_0, (new_dim, new_dim), identity_diag=True)
            self.mu_0 = resize(self.mu_0, (new_dim,))
            self.x_dimensions = new_dim

            build_panels()
            build_state_checkboxes() 
            redraw()


        def resize_inputs(new_n_in: int):
            """Reshape B's columns to new_n_in. Preserve overlap; new columns get zeros."""
            if new_n_in == self.B.shape[1]:
                return
            x_dim = self.x_dimensions
            out = np.zeros((x_dim, new_n_in))
            c = min(self.B.shape[1], new_n_in)
            out[:, :c] = self.B[:, :c]
            self.B = out
            build_panels()
            build_state_checkboxes() 
            redraw()

        build_panels()
        # --- input controller panel ---
        input_frame = tk.LabelFrame(controls, text="input", padx=4, pady=2)
        input_frame.pack(fill=tk.X, pady=6)

        input_type = tk.StringVar(value="pulse")
        input_params = {}            # {type_name: {param_name: Entry}}
        input_subframes = {}         # {type_name: Frame}

        def make_subframe(name, fields):
            """fields: list of (label, default) tuples."""
            sub = tk.Frame(input_frame)
            params = {}
            for r, (label, default) in enumerate(fields):
                tk.Label(sub, text=label, width=10, anchor="w").grid(row=r, column=0)
                e = tk.Entry(sub, width=7, justify="right")
                e.insert(0, str(default))
                e.bind("<Return>", lambda *_: redraw())
                e.grid(row=r, column=1, padx=2, pady=1)
                params[label] = e
            input_params[name] = params
            input_subframes[name] = sub

        make_subframe("pulse", [("t_on", 5), ("t_off", 15), ("amplitude", 1.0)])
        make_subframe("ramp",  [("t_on", 5), ("t_off", 40), ("slope", 0.2)])
        make_subframe("sine",  [("freq", 0.05), ("amplitude", 1.0), ("dt", 1.0)])
        make_subframe("zero",  [])

        def switch_input(*_):
            for sub in input_subframes.values():
                sub.pack_forget()
            input_subframes[input_type.get()].pack(fill=tk.X)
            redraw()

        dropdown = tk.OptionMenu(input_frame, input_type,
                                "pulse", "ramp", "sine", "zero",
                                command=switch_input)
        dropdown.pack(fill=tk.X, pady=2)
        input_subframes["pulse"].pack(fill=tk.X)  # default visible

        def build_input():
            """Construct the current input function from the active panel."""
            n_in = self.B.shape[1]
            p = {k: float(v.get()) for k, v in input_params[input_type.get()].items()}
            t = input_type.get()
            if t == "pulse":
                return self._controllers.make_pulse(int(p["t_on"]), int(p["t_off"]),
                                                    p["amplitude"], n_in)
            if t == "ramp":
                return self._controllers.make_ramp(int(p["t_on"]), int(p["t_off"]),
                                                p["slope"], n_in)
            if t == "sine":
                return self._controllers.make_sine(p["freq"], p["amplitude"], n_in, p["dt"])
            return self._controllers.make_zero(n_in)
        
        def update_spectra():
            for name in ["A", "Q", "R", "P_0"]:
                M = getattr(self, name)
                try:
                    eigs = np.linalg.eigvals(M)
                    eigs = eigs[np.argsort(-np.abs(eigs))]   # sort by magnitude
                    lines = []
                    for e in eigs:
                        if abs(e.imag) > 1e-9:
                            lines.append(f"{e.real:+.3f}{e.imag:+.3f}j")
                        else:
                            lines.append(f"{e.real:+.3f}")
                    if name == "A":
                        lam = np.abs(eigs).max()
                        tag = "stable" if lam < 1 else "unstable" if lam > 1 else "marginal"
                        lines.append(f"|λ|max = {lam:.3f}  ({tag})")
                    spectra_labels[name].config(text="\n".join(lines))
                except Exception as e:
                    spectra_labels[name].config(text=f"err: {e}")

            # B and C are typically rectangular → singular values, not eigenvalues
            for name in ["B", "C"]:
                M = getattr(self, name)
                try:
                    sv = np.linalg.svd(M, compute_uv=False)
                    text = "σ:\n" + "\n".join(f"{s:.3f}" for s in sv)
                    cond = sv.max() / sv.min() if sv.min() > 1e-12 else float("inf")
                    text += f"\ncond = {cond:.1f}"
                    spectra_labels[name].config(text=text)
                except Exception as e:
                    spectra_labels[name].config(text=f"err: {e}")

        # last sim data — shared with input/state popup windows
        _last = {"t": None, "sim": None, "states": None, "input_fn": None}

        # --- read / redraw (modified to use build_input) ---
        def read(name):
            return np.array([[float(e.get()) for e in row] for row in entries[name]])

        def redraw():
            try:
                self.A, self.B, self.C = read("A"), read("B"), read("C")
                self.Q, self.R, self.P_0 = read("Q"), read("R"), read("P_0")
                self.mu_0 = read("mu_0").ravel()
                input_fn = build_input()
            except (ValueError, KeyError) as e:
                status.config(text=f"parse error: {e}"); return

            try:
                sim, states = self._generate_trial(length=self.timestep_cnt,
                                                seed=(self.mu_0, self.P_0),
                                                input_function=input_fn,
                                                return_state=True)
            except Exception as e:
                status.config(text=f"sim error: {e}"); return

            ax_obs.clear()

            t = np.arange(sim.shape[0])
            real = self.observation[0]

            # store for popup windows
            _last["t"] = t
            _last["sim"] = sim
            _last["states"] = states
            _last["input_fn"] = input_fn

            # observations — colour fixed by neuron index, not plot order
            any_obs = False
            for n, v in real_neuron_vars.items():
                if v.get() and n < real.shape[1]:
                    ax_obs.plot(t, real[:, n], color=NEURON_COLORS[n % _N_COLORS],
                                alpha=0.6, label=f"real {n}")
                    any_obs = True
            for n, v in sim_neuron_vars.items():
                if v.get() and n < sim.shape[1]:
                    ax_obs.plot(t, sim[:, n], color=NEURON_COLORS[n % _N_COLORS],
                                linestyle="--", label=f"sim {n}")
                    any_obs = True
            if any_obs:
                ax_obs.legend(fontsize=8, loc="upper right")

            ax_obs.set_ylabel("activation")
            ax_obs.set_xlabel("time")
            fig.tight_layout()
            canvas.draw()
            update_spectra()
            status.config(text="ok")
                        
        def open_input_window():
            if _last["t"] is None:
                return
            t, fn = _last["t"], _last["input_fn"]
            win = tk.Toplevel(root)
            win.title("Input u(t)")
            f = Figure(figsize=(7, 3))
            ax = f.add_subplot(111)
            u_trace = np.array([fn(ti, []) for ti in t])
            selected_in = [i for i, v in input_vars.items() if v.get()]
            for i in selected_in:
                ax.plot(t, u_trace[:, i], label=f"u_{i}")
            if selected_in:
                ax.legend(fontsize=8)
            ax.set_ylabel("input")
            ax.set_xlabel("time")
            f.tight_layout()
            c = FigureCanvasTkAgg(f, master=win)
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            c.draw()

        def open_state_window():
            if _last["t"] is None:
                return
            t, states = _last["t"], _last["states"]
            win = tk.Toplevel(root)
            win.title("Latent states x(t)")
            f = Figure(figsize=(7, 3))
            ax = f.add_subplot(111)
            selected_st = [i for i, v in state_vars.items() if v.get()]
            for i in selected_st:
                ax.plot(t, states[:, i], label=f"x_{i}")
            if selected_st:
                ax.legend(fontsize=8)
            ax.set_ylabel("latent state")
            ax.set_xlabel("time")
            f.tight_layout()
            c = FigureCanvasTkAgg(f, master=win)
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            c.draw()

        def save_csv():
            import os, csv, datetime

            try:
                input_fn = build_input()
                sim = self._generate_trial(length=self.timestep_cnt,
                                        seed=(self.mu_0, self.P_0),
                                        input_function=input_fn)
            except Exception as e:
                status.config(text=f"sim error: {e}"); return

            save_dir = "trial_dumps"
            os.makedirs(save_dir, exist_ok=True)

            fname = f"trial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path = os.path.join(save_dir, fname)

            real = self.observation[0]
            T = sim.shape[0]
            n_in = self.B.shape[1]

            with open(path, "w", newline="") as f:
                w = csv.writer(f)

                # --- header: parameters as comments ---
                w.writerow([f"# input_type: {input_type.get()}"])
                for k, v in input_params[input_type.get()].items():
                    w.writerow([f"# input_{k}: {v.get()}"])
                for name in ["A", "B", "C", "Q", "R", "mu_0", "P_0"]:
                    M = getattr(self, name)
                    w.writerow([f"# {name}: {M.flatten().tolist()}  shape={M.shape}"])
        
                w.writerow([])

                # --- data ---
                y_dim = sim.shape[1]
                w.writerow(["t"]
                        + [f"real_{i}"  for i in range(y_dim)]
                        + [f"sim_{i}"   for i in range(y_dim)]
                        + [f"input_{i}" for i in range(n_in)])
                for t in range(T):
                    u = input_fn(t, [])
                    w.writerow([t] + list(real[t]) + list(sim[t]) + list(u))

            status.config(text=f"saved {fname}")

        tk.Button(spectra, text="Update (or hit Return)", command=redraw).pack(fill=tk.X, pady=(12, 4))
        tk.Button(spectra, text="Show inputs",  command=open_input_window).pack(fill=tk.X, pady=2)
        tk.Button(spectra, text="Show states",  command=open_state_window).pack(fill=tk.X, pady=2)
        tk.Button(spectra, text="Save trial to CSV",      command=save_csv).pack(fill=tk.X, pady=4)
        tk.Button(spectra, text="Plot Gramians",
                command=lambda: self.plot_gramians()).pack(fill=tk.X, pady=4)
        tk.Button(spectra, text="Open comparison window",
                command=lambda: self._open_comparison_window()).pack(fill=tk.X, pady=4)
        status = tk.Label(spectra, text="", fg="gray", wraplength=240, justify="left")
        status.pack(fill=tk.X, pady=(8, 0))

        redraw()
        root.mainloop()
