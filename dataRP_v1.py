# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:05:56 2023

@author: lan0003
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import time

def generate_random_column(x):
    """Generate a data-dependent random column based on the features of x."""
    # Number of features in x
    n = x.shape[0] # x in R^(nxp)

    # Draw random coefficients from a standard Gaussian distribution
    beta = np.random.randn(n) # beta in R^(nx1)
    
    beta = np.expand_dims(beta,1)

    # Construct the data-dependent random column as a linear combination of the features of x
    r_i = np.dot(x.T, beta) # want x.T in R^(pxn) => r_i = (pxn)x(nx1) = (px1)
    
    return r_i

def gram_schmidt_columns(X):
    """Orthogonalize the columns of matrix X using the Gram-Schmidt process."""
    Q, R = np.linalg.qr(X)
    return Q.T

def data_dependent_random_projection(x, n_components):
    """Perform data-dependent random projection on data x."""
    # Number of features in x
    p = x.shape[1]
    
    # Initialize the random matrix R
    R = np.empty((p, n_components))
    
    # Generate each column R_i of R
    for i in range(n_components):
        #R[] = np.c_[R,generate_random_column(x)]
        R[:, i] = generate_random_column(x)[:,0]

    # Orthogonalize the columns of R
    R = gram_schmidt_columns(R)
            # Project the data
    x_projected = np.dot(x, R)
    
    return R, x_projected

def data_dependent_random_projection_v2(X, n_dimensions):
    n, p = X.shape
    
    # Step 1: Formulating R
    R = np.zeros((p, n_dimensions))
    for i in range(n_dimensions):
        beta = np.random.randn(n)
        R[:, i] = X.T @ beta
    
    # Step 2: Making Rows of R Orthonormal
    Q, _ = np.linalg.qr(R[:n_dimensions, :].T)
    R[:n_dimensions, :] = Q.T
    
    # Step 3: Projecting X
    X_tilde = X @ R
    
    return R, X_tilde

def compute_rmse(original, reconstructed):
    """Compute the RMSE between the original and reconstructed data."""
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# Implementing functions for PCA and vanilla random projection

def pca_projection_reconstruction(X, k):
    """Project and reconstruct data using PCA."""
    pca = PCA(n_components=k)
    X_projected = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_projected)
    return X_reconstructed

def vanilla_random_projection(X, k):
    """Project and reconstruct data using vanilla random projection."""
    m, n = X.shape
    R = np.random.randn(n, k)

    # Orthogonalize the columns of R
    R = gram_schmidt_columns(R)

    X_projected = np.dot(X, R)
    X_reconstructed = np.dot(X_projected, R.T)
    return X_reconstructed

# Here's a corrected version of the normalized_data_dependent_random_projection function

def normalized_data_dependent_random_projection(x, n_components):
    """Perform data-dependent random projection on data x with normalization but without orthogonalization."""
    
    # Number of samples in x
    m = x.shape[1]
    
    # Initialize the random matrix R
    R = np.empty((m, n_components))
    
    # Generate each column R_i of R
    for i in range(n_components):
        #R[] = np.c_[R,generate_random_column(x)]
        R[:, i] = generate_random_column(x)[:,0]

  
    # Normalize each column
    for i in range(n_components):
        norm = np.linalg.norm(R[i, :])
        if norm != 0:  # Avoid division by zero
            R[i, :] /= norm
    
    # Project the data
    x_projected = np.dot(x,R)
        
    return R, x_projected


def plot_rmse_vs_k(X, k_values):
    """
    Plot RMSE values against k for original data-dependent random projection, 
    normalized data-dependent random projection, PCA, and vanilla random projection.
    """
    rmse_ddrp = []
    rmse_normalized_ddrp = []
    rmse_pca = []
    rmse_vrp = []

    for k in k_values:
        # Original Data-dependent random projection
        R_k, x_projected_ddrp = data_dependent_random_projection_v2(X, k)
        x_reconstructed_ddrp = x_projected_ddrp @ R_k.T
        print(R_k.T.shape)
        print(x_projected_ddrp.shape)
        print(x_reconstructed_ddrp.shape)
        rmse_ddrp.append(compute_rmse(X, x_reconstructed_ddrp))
        
        # Normalized Data-dependent random projection
        R_k_norm, x_projected_ddrp_norm = normalized_data_dependent_random_projection(X, k)
        x_reconstructed_ddrp_norm = np.dot(x_projected_ddrp_norm,R_k_norm.T)
        rmse_normalized_ddrp.append(compute_rmse(X, x_reconstructed_ddrp_norm))

        # PCA
        x_reconstructed_pca = pca_projection_reconstruction(X, k)
        rmse_pca.append(compute_rmse(X, x_reconstructed_pca))

        # Vanilla random projection
        #x_reconstructed_vrp = vanilla_random_projection(X, k)
        #rmse_vrp.append(compute_rmse(X, x_reconstructed_vrp))

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, rmse_ddrp, marker='o', linestyle='-', label="Original Data-Dependent Random Projection")
    plt.plot(k_values, rmse_normalized_ddrp, marker='s', linestyle=':', label="Normalized Data-Dependent Random Projection")
    plt.plot(k_values, rmse_pca, marker='x', linestyle='--', label="PCA")
    #plt.plot(k_values, rmse_vrp, marker='.', linestyle='-.', label="Vanilla Random Projection")
    
    plt.title('k vs. RMSE for Different Projection Methods')
    plt.xlabel('k (Number of Projected Dimensions)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig("extended_v2_rmse_vs_k_comparison.png")
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
    #k_values = list(range(20, 21))
    
    k_values = list(range(95, 101))

    # Plot RMSE vs k for data-dependent random projection
    plot_rmse_vs_k(X, k_values)



