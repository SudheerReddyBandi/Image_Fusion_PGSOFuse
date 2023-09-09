import numpy as np

def generate_noise(batch_size, latent_dim):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    return noise
