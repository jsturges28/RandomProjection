import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def random_projection(x, k, p):
    # Ensure p >= k
    if p < k:
        raise ValueError("k should be less than or equal to p")
    
    # Step 1: Generate random weight matrix W and orthogonalize columns
    W = np.random.randn(p, k)
    Q, R = np.linalg.qr(W)
    W = Q

    # Step 2: Project data to compressed latent space
    x_tilde = np.dot(W.T, x)

    # Step 3: Reconstruct data back to original space
    x_hat = np.dot(W, x_tilde)

    return x_hat

def compute_rmse(x, x_hat):
    return np.sqrt(np.mean((x - x_hat)**2))

def plot_rmse_vs_k(k_values, p_values):
    plt.figure(figsize=(12, 8))
    
    for p in p_values:
        rmse_values = []
        x = np.random.randn(p)  # Generate random data matrix of dimension p

        for k in k_values:
            # Check if k is less than or equal to p
            if k <= p:
                x_hat = random_projection(x, k, p)
                rmse = compute_rmse(x, x_hat)
                rmse_values.append(rmse)
            else:
                rmse_values.append(np.nan)  # Append NaN if k > p
            print(f'k={k}, p={p}, RMSE={rmse}')

        plt.plot(k_values, rmse_values, marker='o', linestyle='-', label=f'p={p}')

    plt.title('k vs. RMSE for different values of p')
    plt.xlabel('k (Number of Projected Dimensions)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig("rmse_p_k.png")

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
    
    # Generate random weight matrix W and orthogonalize its columns
    W = np.random.randn(p, k)
    Q, R = np.linalg.qr(W)
    W = Q

    # Project data to reduced space
    X_tilde = np.dot(X, W)

    # Reconstruct data back to original space
    X_hat = np.dot(X_tilde, W.T)
    
    return X_tilde, X_hat

# Define the new function to plot RMSE vs k for matrix data
def plot_matrix_rmse_vs_k(X, k_values):
    """
    Plot RMSE values against k for matrix data.
    
    Parameters:
    - X: Input matrix of shape (n_samples, n_features)
    - k_values: List of k values to consider for random projection
    """
    rmse_values = []

    for k in k_values:
        _, X_hat = matrix_random_projection(X, k)
        rmse = np.sqrt(np.mean((X - X_hat)**2))
        rmse_values.append(rmse)

    plt.figure(figsize=(12, 8))
    plt.plot(k_values, rmse_values, marker='o', linestyle='-')
    
    plt.title('k vs. RMSE for Matrix Data')
    plt.xlabel('k (Number of Projected Dimensions)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig("rmse_matrix_k.png")
    plt.show()

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

    for k in k_values:
        # Compute RMSE for Random Projection
        _, X_hat_rp = matrix_random_projection(X, k)
        rmse_rp = np.sqrt(np.mean((X - X_hat_rp)**2))
        rmse_rp_values.append(rmse_rp)

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

if __name__ == '__main__':
    
    #k_values = list(range(2, 200))  # k values from 2 to 200
    #p_values = list(range(200, 2000, 200))  # p values from 200 to 2000 in increments of 200
    #plot_rmse_vs_k(k_values, p_values)

    # Load the CSV file into a DataFrame
    data_df = pd.read_csv("data.csv")

    # Apply random projection to the matrix data
    #k_example = 50
    #X_tilde, X_hat = matrix_random_projection(data_df.values, k_example)

    # Compute RMSE between the original and reconstructed matrices
    #rmse_matrix = np.sqrt(np.mean((data_df.values - X_hat)**2))

    #print(rmse_matrix)

    # Define the range of k values
    #k_values = list(range(0, 101))
    
    # Plot RMSE vs k for matrix data
    #plot_matrix_rmse_vs_k(data_df.values, k_values)

    # Apply PCA to the data and get the PCA-reduced and reconstructed versions
    #k_pca_example = 50
    #X_pca, X_reconstructed = pca_projection_and_reconstruction(data_df.values, k_pca_example)

    # Compute RMSE between the original and reconstructed matrices
    #rmse_pca = np.sqrt(np.mean((data_df.values - X_reconstructed)**2))

    #print(rmse_pca)

    # Define the range of k values
    k_values_range = list(range(1, 101))  # Starting from 1 as 0 dimensions doesn't make sense

    # Call the function to generate the comparison plot
    compare_rmse_vs_k(data_df.values, k_values_range)