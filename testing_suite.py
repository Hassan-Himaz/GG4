import numpy as np
from dynamax_EM_fitting import dynamax_EM_Fitting
from PEM_framework import PEM_Framework
from Illustrator import Illustrator
from Simulator import Simulator, make_pulse,make_sine,make_zero
from PEM_framework import PEM_Framework
from LDSParams import LDSParams
#--------------------------------------------


##   we want this to showcase everything that our




#----------------------------------------------

def testing_suite():
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    # illustrator.spam_everything()



    # sim = Simulator(params = None, illustrator)



    # pem = PEM_Framework(number_hidden_states=1, number_observed_states=16, number_inputs=2,number_restarts = 2)

    # pem.pem_main(data_set=np.load("ExampleDataset.npy"), training_trials=[0,1,2,3], testing_trials=[4])


    estimator = dynamax_EM_Fitting(chosen_state_dimension=2,chosen_input_dimension=0 ,num_em_iters=100, num_restarts = 10,illustrator = illustrator, )
    em_fitted_params,best_lls = estimator.fit()
    controller = make_pulse(0,5,2,2)
    fitted_params = LDSParams.from_dynamax(em_fitted_params)

    sim = Simulator(fitted_params,illustrator,controller)
    sim.compare_plot()



testing_suite()