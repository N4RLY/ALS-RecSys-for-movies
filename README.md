# ALS-RecSys-for-movies

# Movie Recommendation System using Alternating Least Squares (ALS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a movie recommendation system based on the Alternating Least Squares (ALS) matrix factorization algorithm using the MovieLens dataset. It provides both a standard ALS implementation and a parallelized version for improved performance. Additionally, it features an optimized approach for efficiently updating recommendations when a user adds new ratings (incremental updates), which is particularly useful for real-time applications.

## Features

- **Matrix Factorization** using Alternating Least Squares (ALS) algorithm 
- **Parallelized Implementation** for faster training on multi-core systems
- **Hyperparameter Tuning** to find optimal parameters
- **Cold Start Handling** with incremental updates for new user ratings
- **Performance Metrics** including RMSE for model evaluation
- **Recommendation Generation** to suggest movies for users

## Dataset

The project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), which contains:
- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Each user has rated at least 20 movies
- Data sparsity: 0.0513 (5.13% of the matrix has values)

## Mathematical Formulation

The ALS matrix factorization algorithm works by decomposing the user-item rating matrix into two lower-dimensional matrices representing latent factors. Here's the mathematical derivation:

### Problem Definition

Let:
- $R \in \mathbb{R}^{n \times m}$ be the sparse rating matrix, where $n$ is the number of users and $m$ is the number of items
- $U \in \mathbb{R}^{n \times f}$ be the user latent factor matrix, where $f$ is the number of latent factors
- $V \in \mathbb{R}^{m \times f}$ be the item latent factor matrix

The goal is to find $U$ and $V$ such that $R \approx UV^T$.

### Objective Function

The objective function to minimize:

$$\mathcal{L}(U, V) = \sum_{(i,j) \in \mathcal{R}} (r_{ij} - u_i^T v_j)^2 + \lambda (\|u_i\|^2 + \|v_j\|^2)$$

Where:
- $(i,j) \in \mathcal{R}$ represents observed ratings
- $\lambda$ is the regularization parameter
- $\|u_i\|^2$ and $\|v_j\|^2$ are L2 norms of user and item latent factors

### Alternating Optimization

The key insight in ALS is to alternate between fixing one set of variables and optimizing the other:

1. Fix $V$ and solve for $U$:
   - For each user $i$, compute:
   - $u_i = (V_i^T V_i + \lambda I)^{-1} V_i^T R_i$
   - Where $V_i$ is the matrix of item factors for items rated by user $i$

2. Fix $U$ and solve for $V$:
   - For each item $j$, compute:
   - $v_j = (U_j^T U_j + \lambda I)^{-1} U_j^T R_j$
   - Where $U_j$ is the matrix of user factors for users who rated item $j$

This alternating process continues until convergence or a maximum number of iterations is reached.

<details>
<summary><b>Detailed Mathematical Derivation</b> (click to expand)</summary>

### Complete Derivation from Objective Function to Closed-Form Solution

Starting with our objective function:

$$\mathcal{L}(U, V) = \sum_{(i,j) \in \mathcal{R}} (r_{ij} - u_i^T v_j)^2 + \lambda \left(\sum_i \|u_i\|^2 + \sum_j \|v_j\|^2\right)$$

To find the optimal $U$ and $V$, we alternate between optimizing one while keeping the other fixed.

#### Step 1: Optimizing $U$ with fixed $V$

For a specific user $i$, we need to minimize:

$$\mathcal{L}(u_i) = \sum_{j:(i,j) \in \mathcal{R}} (r_{ij} - u_i^T v_j)^2 + \lambda \|u_i\|^2$$

To find the minimum, we take the derivative with respect to $u_i$ and set it to zero:

$$\nabla_{u_i} \mathcal{L} = -2 \sum_{j:(i,j) \in \mathcal{R}} (r_{ij} - u_i^T v_j)v_j + 2\lambda u_i = 0$$

Rearranging:

$$\sum_{j:(i,j) \in \mathcal{R}} (u_i^T v_j)v_j + \lambda u_i = \sum_{j:(i,j) \in \mathcal{R}} r_{ij}v_j$$

We can rewrite the left side:

$$\sum_{j:(i,j) \in \mathcal{R}} v_j v_j^T u_i + \lambda I u_i = \sum_{j:(i,j) \in \mathcal{R}} r_{ij}v_j$$

Factoring out $u_i$:

$$\left(\sum_{j:(i,j) \in \mathcal{R}} v_j v_j^T + \lambda I\right) u_i = \sum_{j:(i,j) \in \mathcal{R}} r_{ij}v_j$$

Let's define:
- $V_i$ as the matrix whose rows are $v_j$ for all items $j$ rated by user $i$
- $R_i$ as the vector of ratings given by user $i$

Then we can rewrite the equation in matrix form:

$$(V_i^T V_i + \lambda I) u_i = V_i^T R_i$$

Solving for $u_i$:

$$u_i = (V_i^T V_i + \lambda I)^{-1} V_i^T R_i$$

This is the closed-form solution for updating user factors.

#### Step 2: Optimizing $V$ with fixed $U$

Similarly, for a specific item $j$, we need to minimize:

$$\mathcal{L}(v_j) = \sum_{i:(i,j) \in \mathcal{R}} (r_{ij} - u_i^T v_j)^2 + \lambda \|v_j\|^2$$

Taking the derivative with respect to $v_j$ and setting it to zero:

$$\nabla_{v_j} \mathcal{L} = -2 \sum_{i:(i,j) \in \mathcal{R}} (r_{ij} - u_i^T v_j)u_i + 2\lambda v_j = 0$$

Following the same process as above, we get:

$$v_j = (U_j^T U_j + \lambda I)^{-1} U_j^T R_j$$

Where:
- $U_j$ is the matrix whose rows are $u_i$ for all users $i$ who rated item $j$
- $R_j$ is the vector of ratings given to item $j$

#### Matrix Form Representation

For computational efficiency, we can express these update rules in matrix form:

For all users at once:
$$(V^T V + \lambda I) U^T = V^T R$$

For all items at once:
$$(U^T U + \lambda I) V^T = U^T R^T$$

But since each user/item may have rated/been rated by different subsets of items/users, we typically implement this by iterating through each user and item individually using the equations:

$$u_i = (V_i^T V_i + \lambda I)^{-1} V_i^T R_i$$
$$v_j = (U_j^T U_j + \lambda I)^{-1} U_j^T R_j$$

The regularization term $\lambda I$ ensures that the matrices are invertible, even when some users have rated very few items or some items have very few ratings.

</details>

### Gradient Descent Update (Alternative Approach)

The model can also be optimized using gradient descent:

$$u_i \leftarrow u_i - \alpha \nabla_{u_i} f(U, V)$$
$$v_j \leftarrow v_j - \alpha \nabla_{v_j} f(U, V)$$

Where:
- $\nabla_{u_i} f(U, V) = -2(r_{ij} - u_i^T v_j)v_j + 2\lambda u_i$
- $\nabla_{v_j} f(U, V) = -2(r_{ij} - u_i^T v_j)u_i + 2\lambda v_j$

## Implementation Details

### Standard ALS Implementation

The standard ALS implementation:
- Initializes user and item factor matrices with random normal values
- Alternates between updating user and item factors using the closed-form solution
- Computes RMSE after each iteration to monitor convergence

```python
def als_matrix_factorization(R, n_factors=20, lambda_reg=0.1, n_iterations=20, verbose=True):
    U, V = init_matrix_factorization(R, n_factors)
    n_users, n_items = R.shape
    I = np.eye(n_factors)
    train_errors = []
    
    for iteration in range(n_iterations):
        # Update user factors
        for i in range(n_users):
            rated_items = R[i].nonzero()[1]
            if len(rated_items) == 0:
                continue
            ratings = R[i, rated_items].toarray()[0]
            V_j = V[rated_items, :]
            U[i] = np.linalg.solve(V_j.T @ V_j + lambda_reg * I, V_j.T @ ratings.transpose()).transpose()
        
        # Update item factors
        for j in range(n_items):
            rating_users = R[:, j].nonzero()[0]
            if len(rating_users) == 0:
                continue
            ratings = R[rating_users, j].toarray().flatten()
            U_i = U[rating_users, :]
            V[j] = np.linalg.solve(U_i.T @ U_i + lambda_reg * I, U_i.T @ ratings.transpose()).transpose()
        
        train_rmse = calculate_rmse(R, U, V)
        train_errors.append(train_rmse)
        
        if verbose and (iteration + 1) % 1 == 0:
            print(f"Iteration {iteration+1}/{n_iterations} - RMSE: {train_rmse:.4f}")
    
    return U, V, train_errors
```

### Parallelized ALS Implementation

The parallelized implementation uses multiprocessing to update user and item factors in parallel:
- Utilizes shared memory to efficiently share matrix data across processes
- Divides users and items into chunks for parallel processing
- Achieves significant speedups on multi-core systems (up to 2.8x faster on test systems)

### Incremental Updates for New Ratings

The project includes an optimized method for updating user factors when new ratings are added:
- Updates only the factors for the user who provided new ratings
- Keeps item factors fixed
- Much faster than retraining the entire model (orders of magnitude speed improvement)
- Maintains comparable recommendation quality to full retraining

## Performance Metrics

### Convergence

ALS convergence over iterations (RMSE vs Iteration):

| Iteration | RMSE (Training) |
|-----------|-----------------|
| 1         | 0.5628          |
| 2         | 0.2580          |
| 5         | 0.1104          |
| 10        | 0.0641          |
| 15        | 0.0474          |
| 20        | 0.0385          |

### Hyperparameter Tuning Results

Results from grid search over hyperparameters:

| n_factors | lambda_reg | n_iterations | Train RMSE | Test RMSE |
|-----------|------------|--------------|------------|-----------|
| 100       | 0.1        | 20           | 0.0165     | 1.5802    |
| 100       | 0.1        | 10           | 0.0286     | 1.5893    |
| 50        | 0.1        | 20           | 0.1721     | 1.6000    |
| 50        | 0.1        | 10           | 0.2113     | 1.6055    |
| 100       | 0.01       | 20           | 0.0090     | 1.6130    |

### Parallel vs Sequential Performance

Performance comparison between parallel and sequential implementations:

| Implementation | Time (seconds) | Speedup |
|----------------|----------------|---------|
| Sequential ALS | 9.67           | 1.00x   |
| Parallel ALS   | 3.45           | 2.80x   |

### Incremental Update Performance

Performance comparison between full retraining and incremental updates:

| Method           | Time (seconds) | RMSE    |
|------------------|----------------|---------|
| Full Retraining  | 3.29           | 0.0384  |
| Incremental Update | 0.03           | 0.0401  |

## Using the System

### Training the Model
```python
# Load data
R = sparse.csr_matrix(ratings_matrix)

# Train model
U, V, train_errors = als_matrix_factorization_parallel(
    R, n_factors=100, lambda_reg=0.1, n_iterations=20, verbose=True
)
```

### Getting Recommendations
```python
# Get top N recommendations for a user
recommendations = recommend_items(user_id, U, V, R, top_n=10)
```

### Handling New Ratings
```python
# Add new ratings and incrementally update (fast)
user_id = 4
item_ids = [318, 306, 172, 659, 292, 990]
ratings = [5, 4, 3, 5, 4, 5]
U_updated = add_rating_optimized(user_id, item_ids, ratings, R)
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Multiprocessing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
