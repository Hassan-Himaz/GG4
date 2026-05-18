import numpy as np

from PEM_framework import PEM_Framework





#so want model data using 1 hidden state, 16 observed states and 2 input

pem = PEM_Framework(number_hidden_states=1, number_observed_states=16, number_inputs=2,number_restarts = 2)

pem.pem_main(data_set=np.load("ExampleDataset.npy"), training_trials=[0,1,2,3], testing_trials=[4])