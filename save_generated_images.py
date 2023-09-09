import matplotlib.pyplot as plt

def save_generated_images(epoch, generator, pca_sar, pca_optical):
    # Generate fused images from random noise, SAR PCA, and optical PCA
    noise = generate_noise(1, latent_dim)
    generated_image = generator.predict([noise, pca_sar, pca_optical])

    # Rescale generated_image from [-1, 1] to [0, 1]
    generated_image = 0.5 * generated_image + 0.5

    # Reshape the generated image if needed
    generated_image = generated_image.reshape(image_size)

    # Save the generated image
    plt.imshow(generated_image)
    plt.axis('off')
    plt.savefig(f'generated_image_epoch_{epoch}.png')
    plt.close()
