from sklearn.decomposition import PCA

def perform_pca(sar_batch, optical_batch, num_components):
    # Reshape the SAR and optical batches if necessary
    sar_batch = sar_batch.reshape(-1, sar_batch.shape[1] * sar_batch.shape[2] * sar_batch.shape[3])
    optical_batch = optical_batch.reshape(-1, optical_batch.shape[1] * optical_batch.shape[2] * optical_batch.shape[3])

    # Perform PCA separately for SAR and optical
    pca_sar = PCA(n_components=num_components)
    pca_optical = PCA(n_components=num_components)

    pca_sar.fit(sar_batch)
    pca_optical.fit(optical_batch)

    sar_components = pca_sar.components_
    optical_components = pca_optical.components_

    return sar_components, optical_components
