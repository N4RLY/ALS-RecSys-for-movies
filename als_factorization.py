import numpy as np

def init_matrix_factorization(R, n_factors=20, seed=42):
    np.random.seed(seed)
    n_users, n_items = R.shape 
    U = np.random.normal(0, 0.1, (n_users, n_factors))
    V = np.random.normal(0, 0.1, (n_items, n_factors))
    return U, V

def calculate_rmse(R, U, V):
    users, items = R.nonzero()    
    predictions = np.sum(U[users] * V[items], axis=1)
    actuals = R[users, items].A.flatten()
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    return rmse

def evaluate_rmse(U, V, test_data, user_id_map, item_id_map):
    test_errors = []
    
    for _, row in test_data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']
        
        if user_id not in user_id_map or item_id not in item_id_map:
            continue
            
        user_idx = user_id_map[user_id]
        item_idx = item_id_map[item_id]
        
        predicted_rating = predict(user_idx, item_idx, U, V)
        error = (actual_rating - predicted_rating) ** 2
        test_errors.append(error)
    
    rmse = np.sqrt(np.mean(test_errors))
    return rmse

def als_matrix_factorization(R, n_factors=20, lambda_reg=0.1, n_iterations=20, verbose=True) -> tuple[np.ndarray, np.ndarray, list[float]]:
    U, V = init_matrix_factorization(R, n_factors)
    n_users, n_items = R.shape
    I = np.eye(n_factors)
    train_errors = []
    
    for iteration in range(n_iterations):
        for i in range(n_users):
            rated_items = R[i].nonzero()[1]
            if len(rated_items) == 0:
                continue
            ratings = R[i, rated_items].toarray()[0]
            V_j = V[rated_items, :]
            U[i] = np.linalg.solve(V_j.T @ V_j + lambda_reg * I, V_j.T @ ratings.transpose()).transpose()
        
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


def partial_als_matrix_factorization(user_id, R, U, V, lambda_reg=0.1) -> np.ndarray:
    I = np.eye(U.shape[1])
    rated_items = R[user_id].nonzero()[1]
    ratings = R[user_id, rated_items].toarray()[0]
    V_j = V[rated_items, :]
    U[user_id] = np.linalg.solve(V_j.T @ V_j + lambda_reg * I, V_j.T @ ratings.transpose()).transpose()

    train_rmse = calculate_rmse(R, U, V)
    print(f"Train Error: {train_rmse}")

    return U

def predict(user_id, item_id, U, V, min_rating=1, max_rating=5):
    raw_prediction = U[user_id] @ V[item_id].T
    range_size = max_rating - min_rating
    return min_rating + range_size * (1 / (1 + np.exp(-raw_prediction)))

def recommend_items(user_id, U, V, R, top_n=10, exclude_rated=True, min_rating=1, max_rating=5):
    predictions = U[user_id] @ V.T
    range_size = max_rating - min_rating
    predictions = min_rating + range_size * (1 / (1 + np.exp(-predictions)))

    if exclude_rated:
        rated_items = R[user_id].nonzero()[1]
        predictions[rated_items] = -np.inf
    
    top_item_indices = np.argsort(predictions)[::-1][:top_n]
    recommendations = [(item_id, predictions[item_id]) for item_id in top_item_indices]
    return recommendations