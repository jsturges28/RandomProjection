import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import time

def matrix_random_projection(X, k):
    """
    Apply random projection to a matrix X.
    
    Parameters:
    - X: Input matrix of shape (n_samples, n_features)
    - k: Target dimensionality after projection
    
    Returns:
    - X_tilde: Reduced matrix of shape (n_samples, k)
    - X_hat: Reconstructed matrix of shape (n_samples, n_features)
    """
    
    p = X.shape[1]  # Original dimensionality
    
    # Check if k is less than or equal to p
    if k > p:
        raise ValueError("k should be less than or equal to p")
    
    # Measure time before generating W
    start_time = time.time()
    
    # Generate random weight matrix W and orthogonalize its columns
    W = np.random.randn(p, k)
    Q, R = np.linalg.qr(W)

    # Measure time after generating W
    end_time = time.time()
    
    # Compute the time difference
    w_gen_time = end_time - start_time

    W = Q # W is now an orthogonal matrix

    # Project data to reduced space
    X_tilde = np.dot(X, W)

    # Reconstruct data back to original space
    X_hat = np.dot(X_tilde, W.T)
    
    return X_tilde, X_hat, w_gen_time

def pca_projection_and_reconstruction(X, k):
    """
    Apply PCA to matrix X and then reconstruct back to original space.
    
    Parameters:
    - X: Input matrix of shape (n_samples, n_features)
    - k: Target dimensionality after projection
    
    Returns:
    - X_pca: Reduced matrix after PCA
    - X_reconstructed: Reconstructed matrix after inverse PCA transformation
    """
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    return X_pca, X_reconstructed

def compare_rmse_vs_k(X, k_values):
    """
    Compare RMSE values for Random Projection and PCA against k values.
    
    Parameters:
    - X: Input matrix of shape (n_samples, n_features)
    - k_values: List of k values to consider for dimensionality reduction
    """
    rmse_rp_values = []
    rmse_pca_values = []
    w_gen_times = []

    for k in k_values:
        # Compute RMSE for Random Projection
        _, X_hat_rp, w_gen_time = matrix_random_projection(X, k)
        rmse_rp = np.sqrt(np.mean((X - X_hat_rp)**2))
        rmse_rp_values.append(rmse_rp)
        w_gen_times.append(w_gen_time)

        # Compute RMSE for PCA
        _, X_hat_pca = pca_projection_and_reconstruction(X, k)
        rmse_pca = np.sqrt(np.mean((X - X_hat_pca)**2))
        rmse_pca_values.append(rmse_pca)

    plt.figure(figsize=(12, 8))
    plt.plot(k_values, rmse_rp_values, marker='o', linestyle='-', label='Random Projection')
    plt.plot(k_values, rmse_pca_values, marker='x', linestyle='-', label='PCA')
    
    plt.title('k vs. RMSE for crime.csv Data')
    plt.xlabel('k (Number of Projected Dimensions)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig("rmse_comparison_k.png")
    plt.show()

    # Plot time taken to generate W against k values
    plot_time_vs_k(k_values, w_gen_times)

def plot_time_vs_k(k_values, w_generation_times):
    """
    Plot the time taken to generate W against k values.
    
    Parameters:
    - k_values: List of k values considered
    - w_generation_times: List of times taken to generate W for each k
    """
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, w_generation_times, marker='o', linestyle='-')
    
    plt.title('k vs. Time taken to generate W')
    plt.xlabel('k (Number of Projected Dimensions)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig("time_vs_k.png")
    plt.show()

def plot_reconstructed_images(X, k_values):

    # Calculate the number of rows based on the number of k_values
    num_rows = len(k_values) // 2 + len(k_values) % 2 + 1  # +1 for the original image
    
    # Define a figure
    fig = plt.figure(figsize=(10, 2 * num_rows))
    
    # Plot the original image
    ax = plt.subplot(num_rows, 2, 1)
    plt.imshow(X.reshape(28, 28), cmap='gray')
    ax.set_title('Original Image')
    plt.axis('off')
    
    for i, k in enumerate(k_values, start=1):
        _, X_hat, w_gen_time = matrix_random_projection(X.reshape(1, -1), k) # flatten from 28x28 -> 1x768

        # Compute the RMSE for the reconstruction
        rmse = np.sqrt(np.mean((X - X_hat)**2))
        
        # Plot
        ax = plt.subplot(num_rows, 2, i + 2)  # +2 to account for the original image
        plt.imshow(X_hat.reshape(28, 28), cmap='gray')
        ax.set_title(f'k={k}, RMSE={rmse:.4f}, W gen time={w_gen_time:.4f} s')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("reconstructed_images_RP.png")
    plt.show()

if __name__ == '__main__':

    # Load the CSV file into a DataFrame
    data_df = pd.read_csv("data.csv")
    mnist_df = pd.read_csv("mnist_test.csv")

    # Define the range of k values
    #k_values_range = list(range(1, 101))  # Starting from 1 as 0 dimensions doesn't make sense

    # Call the function to generate the comparison plot
    #compare_rmse_vs_k(data_df.values, k_values_range)



sample_digit_pixels = mnist_df.iloc[0, 1:].values

# Visualize reconstructions for RP 
plot_reconstructed_images(sample_digit_pixels, [2, 20, 50, 75, 100, 200, 300, 400, 500, 600, 700, 784])

