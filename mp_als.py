import numpy as np
from typing import Tuple, List
import multiprocessing as mp
from functools import partial
from time import time
import ctypes
from als_factorization import init_matrix_factorization, calculate_rmse

# Global variables for shared memory
_shared_U = None
_shared_V = None
_U_shape = None
_V_shape = None
_R = None
_lambda_reg = None

def _init_worker(shared_u, shared_v, u_shape, v_shape, r, lambda_reg):
    """Initialize worker process with shared memory references"""
    global _shared_U, _shared_V, _U_shape, _V_shape, _R, _lambda_reg
    _shared_U = shared_u
    _shared_V = shared_v
    _U_shape = u_shape
    _V_shape = v_shape
    _R = r
    _lambda_reg = lambda_reg

def _compute_factors(mode, indices):
    """Compute factors for users (mode=0) or items (mode=1)"""
    global _shared_U, _shared_V, _U_shape, _V_shape, _R, _lambda_reg
    
    # Get shared memory references
    if mode == 0:  # User mode
        factors_local = np.frombuffer(_shared_U).reshape(_U_shape)
        factors_other = np.frombuffer(_shared_V).reshape(_V_shape)
    else:  # Item mode
        factors_local = np.frombuffer(_shared_V).reshape(_V_shape)
        factors_other = np.frombuffer(_shared_U).reshape(_U_shape)
    
    n_factors = factors_other.shape[1]
    I = np.eye(n_factors)
    
    # Pre-compute the identity matrix with regularization
    lambda_I = _lambda_reg * I
    
    for idx in indices:
        if mode == 0:  # User mode
            rated_indices = _R[idx].nonzero()[1]
            if len(rated_indices) == 0:
                continue
            ratings = _R[idx, rated_indices].toarray()[0]
            other_factors = factors_other[rated_indices, :]
        else:  # Item mode
            rated_indices = _R[:, idx].nonzero()[0]
            if len(rated_indices) == 0:
                continue
            ratings = _R[rated_indices, idx].toarray().flatten()
            other_factors = factors_other[rated_indices, :]
        
        AtA = other_factors.T @ other_factors + lambda_I
        Atb = other_factors.T @ ratings
        factors_local[idx] = np.linalg.solve(AtA, Atb)

def als_matrix_factorization_parallel(R, n_factors=100, lambda_reg=0.1, n_iterations=20, verbose=True, 
                             n_processes=None) -> Tuple[np.ndarray, np.ndarray, List[float]]:

    U, V = init_matrix_factorization(R, n_factors)
    n_users, n_items = R.shape
    train_errors = []
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Limit number of processes to a reasonable number based on problem size
    n_processes = min(n_processes, max(1, min(n_users, n_items) // 100))
    
    # Create shared memory for the matrices
    U_shape = U.shape
    V_shape = V.shape
    
    # Use shared memory more efficiently with ctypes arrays
    shared_U = mp.RawArray(ctypes.c_double, U.size)
    shared_V = mp.RawArray(ctypes.c_double, V.size)
    
    # Copy initial values to shared memory
    np.copyto(np.frombuffer(shared_U).reshape(U_shape), U)
    np.copyto(np.frombuffer(shared_V).reshape(V_shape), V)
    
    # Calculate optimal chunk size for better load balancing
    user_chunk_size = max(100, n_users // (n_processes * 2))
    item_chunk_size = max(100, n_items // (n_processes * 2))
    
    user_chunks = [range(i, min(i + user_chunk_size, n_users)) 
                 for i in range(0, n_users, user_chunk_size)]
    item_chunks = [range(j, min(j + item_chunk_size, n_items)) 
                 for j in range(0, n_items, item_chunk_size)]
    
    # Create a single process pool for the entire computation
    with mp.Pool(processes=n_processes, 
                 initializer=_init_worker, 
                 initargs=(shared_U, shared_V, U_shape, V_shape, R, lambda_reg)) as pool:
        
        # Main ALS iterations
        for iteration in range(n_iterations):
            start_time = time()
            
            # Update user factors in parallel (mode 0)
            pool.map(partial(_compute_factors, 0), user_chunks)
            
            # Update item factors in parallel (mode 1)
            pool.map(partial(_compute_factors, 1), item_chunks)
            
            # Copy updated factors back from shared memory
            U = np.frombuffer(shared_U).reshape(U_shape).copy()
            V = np.frombuffer(shared_V).reshape(V_shape).copy()
            
            train_rmse = calculate_rmse(R, U, V)
            train_errors.append(train_rmse)
            
            elapsed = time() - start_time
            if verbose:
                print(f"Iteration {iteration+1}/{n_iterations} - RMSE: {train_rmse:.4f} - Time: {elapsed:.2f}s")
    
    return U, V, train_errors
