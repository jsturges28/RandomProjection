import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def deep_random_projection(x, k):
    # Ensure p >= k
    p = x.shape[1]
    if p < k:
        raise ValueError("k should be less than or equal to p")
    
    # Step 1: Generate random weight matrices W1, W2 and orthogonalize columns

    # Measure time before generating W1, W2
    start_time = time.time()

    W1 = np.random.randn(p, k)
    W2 = np.random.randn(p, k)
    Q1, _ = np.linalg.qr(W1)
    Q2, _ = np.linalg.qr(W2)
    W1 = Q1
    W2 = Q2

    # Measure time after generating W1, W2
    end_time = time.time()
    
    # Compute the time difference
    w_gen_time = end_time - start_time

    # Step 2: Project data to compressed latent space
    x_tilde1 = np.dot(X, W1)
    x_tilde2 = np.dot(x_tilde1, W2)

    # Step 3: Reconstruct data back to original space
    x_hat1 = np.dot(W2.T, x_tilde2)
    x_hat2 = np.dot(W1.T, x_hat1)

    return x_tilde2, x_hat2

def compare_rmse_vs_k(X, k_values):
    """
    Compare RMSE values for Random Projection and PCA against k values.
    
    Parameters:
    - X: Input matrix of shape (n_samples, n_features)
    - k_values: List of k values to consider for dimensionality reduction
    """
    rmse_rp_values = []
    w_gen_times = []

    for k in k_values:
        # Compute RMSE for Random Projection
        _, X_hat_rp, w_gen_time = deep_random_projection(X, k)
        rmse_rp = np.sqrt(np.mean((X - X_hat_rp)**2))
        rmse_rp_values.append(rmse_rp)
        w_gen_times.append(w_gen_time)

    plt.figure(figsize=(12, 8))
    plt.plot(k_values, rmse_rp_values, marker='o', linestyle='-', label='Random Projection')
    
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

