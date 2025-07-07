from star_data import get_star_data, visualize_data
from star_spqc import create_spqc_circuit, visualize_circuit, bind_params, create_random_weights
import numpy as np

# Get all the star classification data
train_features, test_features, train_labels, test_labels, train_onehot, test_onehot, boundary = get_star_data()
# visualize_data()

# Create the SPQC circuit
spqc_frame = create_spqc_circuit(t=0, m=2, n=2, r=1)

# Create and bind random weights + features
random_weights = create_random_weights(spqc_frame, seed=42)
example_input = [0.1, 0.2]
spqc = bind_params(spqc_frame, example_input, random_weights)

visualize_circuit(spqc)