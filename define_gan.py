import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def define_gan(generator, discriminator):
    # Make the discriminator non-trainable during GAN training
    discriminator.trainable = False

    # Inputs to the GAN: noise, SAR PCA components, and optical PCA components
    noise_input = Input(shape=(latent_dim,))
    sar_pca_input = Input(shape=(num_components,))
    optical_pca_input = Input(shape=(num_components,))

    # Generate an image using the generator
    generated_image = generator([noise_input, sar_pca_input, optical_pca_input])

    # Pass the generated image through the discriminator
    validity = discriminator(generated_image)

    # Create the GAN model that combines the generator and discriminator
    gan = Model(inputs=[noise_input, sar_pca_input, optical_pca_input], outputs=validity)

    return gan

# Example usage:
gan = define_gan(generator, discriminator)
gan.summary()
