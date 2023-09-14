import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()

if __name__ == '__main__':
    
    k_values = list(range(2, 200))  # k values from 2 to 200
    p_values = list(range(200, 2000, 200))  # p values from 200 to 2000 in increments of 200
    plot_rmse_vs_k(k_values, p_values)