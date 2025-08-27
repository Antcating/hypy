import numpy as np

def compute_distance_weight(distance, distance_cutoff, cutoff_width): 
    weight = 1.0 / (1.0 + np.exp((distance - distance_cutoff) / (cutoff_width/4*np.e)))

    return weight