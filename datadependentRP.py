import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import time

def generate_random_column(x):
    """Generate a data-dependent random column based on the features of x."""
    # Number of features in x
    p = x.shape[1]

    # Draw random coefficients from a standard Gaussian distribution
    beta = np.random.randn(p)

    # Construct the data-dependent random column as a linear combination of the features of x
    r_i = np.dot(x, beta)

    return r_i

def gram_schmidt_columns(X):
    """Orthogonalize the columns of matrix X using the Gram-Schmidt process."""
    Q, R = np.linalg.qr(X)
    return Q

def data_dependent_random_projection(x, n_components):
    """Perform data-dependent random projection on data x."""
    # Number of samples in x
    m = x.shape[0]
    
    # Initialize the random matrix R
    R = np.empty((m, n_components))
    
    # Generate each column R_i of R
    for i in range(n_components):
        R[:, i] = generate_random_column(x)

    # Orthogonalize the columns of R
    R = gram_schmidt_columns(R)
    
    # Project the data
    x_projected = np.dot(R.T, x)
    
    return R, x_projected

def compute_rmse(original, reconstructed):
    """Compute the RMSE between the original and reconstructed data."""
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def plot_ddrp_rmse_vs_k(X, k_values):
    """
    Plot RMSE values against k for data-dependent random projection.
    
    Parameters:
    - X: Input matrix of shape (n_samples, n_features)
    - k_values: List of k values to consider for random projection
    """
    rmse_values = []

    for k in k_values:
        R_k, x_projected = data_dependent_random_projection(X, k)
        x_reconstructed = np.dot(R_k, x_projected)
        rmse = compute_rmse(X, x_reconstructed)
        rmse_values.append(rmse)

    plt.figure(figsize=(12, 8))
    plt.plot(k_values, rmse_values, marker='o', linestyle='-')
    
    plt.title('k vs. RMSE for Data-Dependent Random Projection')
    plt.xlabel('k (Number of Projected Dimensions)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig("rmse_ddrp_k.png")
    plt.show()

if __name__ == '__main__':

    # Load the CSV file into a DataFrame
    data_df = pd.read_csv("data.csv")

    # Convert the DataFrame to a NumPy array
    X = data_df.values

    # Specify the dimension (number of components) we want to project to
    #n_components = 10

    # Perform data-dependent random projection
    #R, x_projected = data_dependent_random_projection(X, n_components)

    # Print the shape of the projected data
    #print("Shape of the projected data: ", x_projected.shape)

    # Perform data-dependent reconstruction
    #x_reconstructed = np.dot(R, x_projected)

    # Print the shape of the reconstructed data
    #print("Shape of the reconstructed data: ", x_reconstructed.shape)

    # Compute the RMSE between the original and reconstructed data
    #rmse = compute_rmse(X, x_reconstructed)
    #print("RMSE between original and reconstructed data:", rmse)

    # Define the range of k values
    k_values = list(range(0, 101))

    # Plot RMSE vs k for data-dependent random projection
    plot_ddrp_rmse_vs_k(X, k_values)



