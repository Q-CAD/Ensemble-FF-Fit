import numpy as np
import random

class WeightedSampler:
    def __init__(self, values, dist_params):
        self.values = np.array(values, dtype=float) if values else []
        self.dist_params = dist_params
        
        # Set the seed for reproducibility
        if 'seed' in self.dist_params:
            np.random.seed(self.dist_params['seed'])
            random.seed(self.dist_params['seed'])

    def sample(self):
        dist_type = self.dist_params.get('type', 'uniform')
        method = {
            'uniform': self.uniform_weighting,
            'normal': self.normal_weighting,
            'magnitude': self.magnitude_weighting,
            'binary': self.binary_weighting,
            'lognormal': self.lognormal_weighting
        }.get(dist_type)

        if method is None:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        return method()

    def set_max_min(self, min_val, max_val, values):
        """ Scale values to a defined range [min_val, max_val]. """
        if np.ptp(values) == 0:  # If all values are the same, return min_val for all
            return np.full_like(values, min_val, dtype=float)
        
        scaled_values = np.rint(np.interp(values, (np.min(values), np.max(values)), (min_val, max_val)))
        return np.round(scaled_values, 3)

    def validate_params(self, required_keys):
        """ Check that all required keys are present in dist_params. """
        missing_keys = [key for key in required_keys if key not in self.dist_params or self.dist_params[key] is None]
        if missing_keys:
            #raise ValueError(f"Missing required dist_params keys for {self.dist_params.get('type', 'unknown')}: {missing_keys}")
            raise ValueError(f"Missing required dist_params keys for method '{self.dist_params.get('type', 'unknown')}': {missing_keys}")

    def uniform_weighting(self):
        """ Assign weights uniformly across a range. """
        self.validate_params(['min', 'max', 'spread', 'scale'])
        min_val, max_val, spread, scale = [float(self.dist_params[key]) for key in ['min', 'max', 'spread', 'scale']]

        if len(self.values) == 0:
            return None
        if len(self.values) == 1:
            return np.round(scale * max_val, 3)

        random_values = np.random.randint(0, spread, size=len(self.values))
        return np.round(self.set_max_min(min_val, max_val, scale * random_values), 3)

    def normal_weighting(self):
        """ Assign weights based on a normal distribution. """
        self.validate_params(['min', 'max', 'sigma', 'scale'])
        min_val, max_val, sigma, scale = [float(self.dist_params[key]) for key in ['min', 'max', 'sigma', 'scale']]

        if len(self.values) == 0:
            return None
        if len(self.values) == 1:
            return np.round(scale * max_val, 3)

        normal_values = np.abs(np.random.normal(0, sigma, size=len(self.values)))
        return np.round(self.set_max_min(min_val, max_val, scale * normal_values), 3)

    def magnitude_weighting(self):
        """ Assign weights with higher values getting more weight. """
        self.validate_params(['min', 'max', 'kT', 'scale'])
        min_val, max_val, kT, scale = [float(self.dist_params[key]) for key in ['min', 'max', 'kT', 'scale']]

        if len(self.values) == 0:
            return None
        if len(self.values) == 1:
            return np.round(scale * max_val, 3)

        exp_values = np.exp(self.values / kT)
        noise = np.random.normal(0, np.min(exp_values) * 0.68, size=len(self.values))
        noisy_values = exp_values + noise
        return np.round(self.set_max_min(min_val, max_val, scale * noisy_values), 3)

    def lognormal_weighting(self):
        """ Assign weights based on a lognormal distribution. """
        self.validate_params(['min', 'max', 'mu', 'sigma', 'scale'])
        min_val, max_val, mu, sigma, scale = [float(self.dist_params[key]) for key in ['min', 'max', 'mu', 'sigma', 'scale']]

        if len(self.values) == 0:
            return None
        if len(self.values) == 1:
            return np.round(scale * max_val, 3)

        lognormal_values = np.random.lognormal(mu, sigma, size=len(self.values))
        return np.round(self.set_max_min(min_val, max_val, scale * lognormal_values), 3)

    def binary_weighting(self):
        """ Assign weights as either `min_val` or `max_val` based on probability `split`. """
        self.validate_params(['min', 'max', 'split'])
        min_val, max_val, split = [float(self.dist_params[key]) for key in ['min', 'max', 'split']]
        
        if len(self.values) == 0:
            return None
        if len(self.values) == 1:
            return np.round(random.choices([min_val, max_val], weights=[split, 1 - split]), 3)

        choices = np.random.choice([min_val, max_val], size=len(self.values), p=[split, 1 - split])
        return np.round(choices, 3)


