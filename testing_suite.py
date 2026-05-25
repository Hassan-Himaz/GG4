import numpy as np
from dynamax_EM_fitting import dynamax_EM_Fitting
from PEM_framework import PEM_Framework
from Illustrator import Illustrator
from Simulator import Simulator
from PEM_framework import PEM_Framework
from LDSParams import LDSParams
from Controllers import Controllers
#--------------------------------------------


##   we want this to showcase everything that our

    


#----------------------------------------------

def testing_suite():
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    # illustrator.run_all()

    controller_factory = Controllers()
    controller = controller_factory.make_pulse(0,10,10,2)


    #2 hidden state 2 input state matrices
    A = np.array([[0.01,0 ],
                [0, 0.01]])
    B = np.array([[0.2, 0],
                [0,0.2 ]])
    C = np.array([[0.95, 0],
                [0,0.95 ]])
    Q = np.array([[0.95, 0],
                [0,0.95 ]])
    R = np.array([[0.95, 0],
                [0,0.95 ]])
    mu_0 = np.array([0,0])
    P_0 = np.array([[0.95, 0],
                [0,0.95 ]])


    params = (A,B,C,Q,R,mu_0,P_0)

    params = LDSParams.from_tuple(params)

    sim = Simulator(params,illustrator,controller)
    seed = (mu_0,P_0)
    # sim.compare_plot(seed)
    sim.generate_data()
    sim.explore()

testing_suite()