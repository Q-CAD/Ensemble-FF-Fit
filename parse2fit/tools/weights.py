import numpy as np
import random

class WeightedSampler:
    def __init__(self, values, dist_params):
        self.values = values
        self.dist_params = dist_params
        
        if self.dist_params.get('seed') is not None:
            np.random.seed(self.dist_params.get('seed'))

    def sample(self):
        dist_type = self.dist_params.get('type')
        if dist_type == 'uniform':
            return self.uniform_weighting()
        elif dist_type == 'normal':
            return self.normal_weighting()
        elif dist_type == 'magnitude':
            return self.magnitude_weighting()
        elif dist_type == 'binary':
            return self.binary_weighting()
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def set_max_min(self, min_val, max_val, values):
        m = (max_val - min_val)/(np.max(values) - np.min(values))
        b = min_val - m*(np.min(values))
        ys = [int(m*val+b) for val in values]
        correct_vals = []
        for y in ys:
            if y == max(ys):
                correct_vals.append(float(max_val))
            elif y == min(ys):
                correct_vals.append(float(min_val))
            else:
                correct_vals.append(float(y))
        return correct_vals

    def uniform_weighting(self):
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        spread = self.dist_params.get('spread') # positive integer
        scale = self.dist_params.get('scale')
        if len(self.values) == 1: 
            return np.round(np.multiply(scale, [float(max_val)]), 3)
        elif len(self.values) == 0:
            return None
        else:
            values = list(np.random.randint(0, spread, size=len(self.values)))
            weights = np.round(np.multiply(scale, self.set_max_min(min_val, max_val, values)), 3)
            return weights

    def normal_weighting(self):
        # Return positive values from normal distribution
        # Similar to Normal, but sparser
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        sigma = self.dist_params.get('sigma') # positive float
        scale = self.dist_params.get('scale')
        if len(self.values) == 1:
            return np.round(np.multiply(scale, [float(max_val)]), 3)
        elif len(self.values) == 0:
            return None
        else:
            values = np.abs(np.random.normal(0, sigma, size=len(self.values)))
            weights = np.round(np.multiply(scale, self.set_max_min(min_val, max_val, values)), 3)
            return weights

    def magnitude_weighting(self):
        # Larger magnitudes weighted more
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        kT = self.dist_params.get('kT') # float
        scale = self.dist_params.get('scale')
        if len(self.values) == 1:
            return np.round(np.multiply(scale, [float(max_val)]), 3)
        elif len(self.values) == 0:
            return None
        else:
            starting_values = np.exp(np.array(np.divide(self.values, kT)))
            shift_values = np.random.normal(0, np.min(starting_values)*0.68, size=len(self.values)) # Add noise to the distribution
            values = [starting_values[i] + shift_values[i] for i in range(len(starting_values))]
            weights = np.round(np.multiply(scale, self.set_max_min(min_val, max_val, values)), 3)
            return weights

    def binary_weighting(self):
        # One value or other
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        split = float(self.dist_params.get('split')) # positive float between 0 and 1
        number = 10**str(split)[::-1].find('.') # Give the split based on number of decimal places provided
        low, high = np.rint(number * split), np.rint(number*(1.0-split))
        low_vals, high_vals = [float(min_val) for i in range(int(low))], [float(max_val) for i in range(int(high))]
        choose_from = low_vals + high_vals
        if len(self.values) == 1:
            return random.choices([float(min_val), float(max_val)], weights=[split, 1-split]) 
            #return [float(max_val)]
        elif len(self.values) == 0:
            return None
        else:
            assignments = list(np.random.randint(low=0, high=number, size=len(self.values)))
            weights = [choose_from[i] for i in assignments]
            return weights


