import unittest
import numpy as np
from Illustrator import Illustrator

class TestIllustrator(unittest.TestCase):

    def setUp(self) -> None:
        self.test_observation_set = np.array(
            [
                [  # Trial 0
                    [10,15,20], # Step 0
                    [-2,-3,-2], # Step 1
                    [200,400,100], # Step 2
                ],
                [   # Trial 1
                    [5,4,5], # Step 0
                    [-10,-11,-12], # Step 1
                    [100,140,150] # Step 2
                ],
                [   # Trial 2
                    [5,4,5], # Step 0
                    [-4,-3,-2], # Step 1
                    [104,120,454] # Step 2
                ]
            ]
        )
        self.test_illustrator = Illustrator(self.test_observation_set)

    def test_properties(self):
        
        self.assertTrue(np.allclose(self.test_illustrator.observation, self.test_observation_set))
        self.assertEqual(self.test_illustrator.neuron_cnt, 3)
        self.assertEqual(self.test_illustrator.timestep_cnt, 3)
        self.assertEqual(self.test_illustrator.trial_cnt, 3)
    
    def test_setter(self):
        incorrect_new_data_set = np.array(
            [
                [[
                    [5,6,2],
                    [2,4,5]
                ],
                [
                    [5,6,2],
                    [2,4,5]
                ]]
            ]
        )
        
        with self.assertRaises(ValueError):
            self.test_illustrator.observation = incorrect_new_data_set
        
        new_data_set = np.array(
            [
                [
                    [5,6,2],
                    [2,4,5]
                ],
                [
                    [5,6,2],
                    [2,4,5]
                ]
            ]
        )
        self.test_illustrator.observation = new_data_set

        self.assertTrue(np.allclose(self.test_illustrator.observation, new_data_set))
        self.assertEqual(self.test_illustrator.neuron_cnt, 3)
        self.assertEqual(self.test_illustrator.timestep_cnt, 2)
        self.assertEqual(self.test_illustrator.trial_cnt, 2)
    
    def test_time_mean_none(self):
        self.test_illustrator.observation = self.test_observation_set

        expected_result = np.array(
            [
                [20/3,23/3,10],
                [-16/3,-17/3,-16/3],
                [404/3, 220, 704/3]
            ]
        )
        self.assertTrue(np.allclose(self.test_illustrator.get_trial_averaged_mean(), expected_result))
    
    def test_time_mean_index(self):
        self.test_illustrator.observation = self.test_observation_set
        # Test for neuron 2 (index 1)
        expected_result = np.array(
            [
                [23/3],
                [-17/3],
                [220]
            ]
        )
        result = self.test_illustrator.get_trial_averaged_mean(1)
        self.assertTrue(np.allclose(result, expected_result))
        with self.assertRaises(ValueError):
            result = self.test_illustrator.get_trial_averaged_mean(10)
    
    def test_time_mean_list(self):
        self.test_illustrator.observation = self.test_observation_set
        # Test for neuron 0,1 
        expected_result = np.array(
            [
                [20/3, 23/3],
                [-16/3,-17/3],
                [404/3,220]
            ]
        )
        result = self.test_illustrator.get_trial_averaged_mean([0,1])
        self.assertTrue(np.allclose(result, expected_result))
        with self.assertRaises(ValueError):
            result = self.test_illustrator.get_trial_averaged_mean([0,2,10])



        

    