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

if __name__ == '__main__':

    # Load the CSV file into a DataFrame
    data_df = pd.read_csv("data.csv")

    # Define the range of k values
    k_values_range = list(range(1, 101))  # Starting from 1 as 0 dimensions doesn't make sense

    # Call the function to generate the comparison plot
    compare_rmse_vs_k(data_df.values, k_values_range)

    # Test the function to see if it captures time correctly
    #_, _, example_time = matrix_random_projection(data_df.values, 50)
    #print(example_time)