import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Model

def define_discriminator(image_size):
    # Input for the fused image
    input_image = Input(shape=image_size)

    # Flatten the image
    flat_image = Flatten()(input_image)

    # Dense layers for discriminator
    x = Dense(1024)(flat_image)
    x = LeakyReLU(alpha=0.2)(x)  # LeakyReLU activation
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Output layer for binary classification (real or fake)
    validity = Dense(1, activation='sigmoid')(x)

    # Create the discriminator model
    discriminator = Model(inputs=input_image, outputs=validity)

    return discriminator

# Example usage:
image_size = (256, 256, 3)
discriminator = define_discriminator(image_size)
discriminator.summary()
