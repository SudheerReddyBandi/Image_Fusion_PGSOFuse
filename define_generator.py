import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Reshape, Flatten, Activation
from tensorflow.keras.models import Model

def define_generator(latent_dim, num_components, image_size):
    # Inputs
    noise_input = Input(shape=(latent_dim,))
    sar_pca_input = Input(shape=(num_components,))
    optical_pca_input = Input(shape=(num_components,))

    # Concatenate the noise and PCA components
    concatenated_input = Concatenate()([noise_input, sar_pca_input, optical_pca_input])

    # Dense layers for generating the fused image
    x = Dense(256, activation='relu')(concatenated_input)
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(np.prod(image_size), activation='tanh')(x)  # Output layer with tanh activation

    # Reshape the output to match the image size
    generated_image = Reshape(image_size)(x)

    # Create the generator model
    generator = Model(inputs=[noise_input, sar_pca_input, optical_pca_input], outputs=generated_image)

    return generator

# Example usage:
latent_dim = 100
num_components = 3
image_size = (256, 256, 3)
generator = define_generator(latent_dim, num_components, image_size)
generator.summary()
