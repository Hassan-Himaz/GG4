import numpy as np
import matplotlib.pyplot as plt
from Illustrator import Illustrator
from Simulator import Simulator

data = np.load("ExampleDataset.npy")
print("Dataset loaded successfully. Shape:", data.shape)  # (Trials, Timepoints, Neurons) - so we batch 
print(data[0])#first trial
print(data.shape)


illustrator  = Illustrator(data)


neurons_to_plot = [0,1,2]  # Example neuron indices to plot

#illustrator.plot_neuron(0)  # Plot the first trial's neuron activity


illustrator.plot_neurons_single_trial(neurons_to_plot, 3)  # Plot the second trial's neuron activity
illustrator.neuron_statistics()
illustrator.plot_correlation_matrix()