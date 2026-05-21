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
    