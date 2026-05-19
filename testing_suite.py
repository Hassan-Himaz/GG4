import numpy as np
from PEM_framework import PEM_Framework
from Illustrator import Illustrator
from Simulator import Simulator
from PEM_framework import PEM_Framework
#--------------------------------------------


##   we want this to showcase everything that our




#----------------------------------------------
class 
def testing_suite():
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    illustrator.plot_scatter()
    illustrator.plot_scatter(neuron_list=np.ndarray([0,1,8]), trial_list=np.ndarray([0,1,2]))
    illustrator.plot_heatmap()



    sim = Simulator()



    pem = PEM_Framework(number_hidden_states=1, number_observed_states=16, number_inputs=2,number_restarts = 2)

    pem.pem_main(data_set=np.load("ExampleDataset.npy"), training_trials=[0,1,2,3], testing_trials=[4])