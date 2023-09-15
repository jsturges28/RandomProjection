import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors

def batch_random_projection(data, k):
    transformer = GaussianRandomProjection(n_components=k)
    return transformer.fit_transform(data)

def knn_reconstruction(original_data, reduced_data, k_neighbors=5):
    reconstructed_data = []

    # Initialize the nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(reduced_data)  # +1 to include the point itself

    for i, point in enumerate(reduced_data):
        # Find nearest neighbors indices in reduced space
        _, indices = nbrs.kneighbors(point.reshape(1, -1))
        indices = indices[0][1:]  # Exclude the point itself and flatten

        # Debugging code
        if len(indices) == 0:
            print(f"No neighbors found for point {i}.")
            continue

        nearest_neighbors = original_data[indices]
        # Use the average of the nearest neighbors in the original space as the reconstruction
        reconstruction = np.mean(nearest_neighbors, axis=0)
        reconstructed_data.append(reconstruction)

    return np.array(reconstructed_data)

def visualize_reconstructions(data, reconstructed_data, k_values, num_samples=10):
    # Select a random subset of images
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    data_subset = data[indices]
    reconstructed_data_subset = reconstructed_data[indices]

    fig, axes = plt.subplots(len(k_values) + 1, num_samples, figsize=(20, 15))  # Increase the width

    # Displaying the original images
    for idx, ax in enumerate(axes[0]):
        ax.imshow(data_subset[idx].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if idx == 0:
            ax.set_title('Original', fontsize=10)  # Reduce the font size

    # Displaying the reconstructed images
    for k_idx, k in enumerate(k_values):
        for idx, ax in enumerate(axes[k_idx + 1]):
            reconstruction = reconstructed_data_subset[idx]
            rmse = np.sqrt(np.mean((data_subset[idx] - reconstruction) ** 2))
            ax.imshow(reconstruction.reshape(28, 28), cmap='gray')
            ax.axis('off')
            ax.set_title(f'k={k}, RMSE={rmse:.4f}', fontsize=10)  # Display the RMSE for each image


    plt.tight_layout()
    plt.savefig("reconstructed_images_RP.png")
    plt.show()

if __name__ == '__main__':

    # Load the MNIST dataset
    mnist_df = pd.read_csv("mnist_train.csv")

    # Select a subset of the dataset for our experiment
    num_samples = 500
    k = 50
    data = mnist_df.iloc[:num_samples, 1:].values
    reduced_data = batch_random_projection(data, k)

    reconstructed_data = knn_reconstruction(data, reduced_data)

    visualize_reconstructions(data, reconstructed_data, [k])
