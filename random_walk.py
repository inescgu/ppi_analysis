# random_walk.py
# Implementation of Random Walk with Restart algorithm for prioritizing genes

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp


def random_walk_with_restart(G, seed_genes, n_iterations=1000, restart_prob=0.7, max_steps=100):
    """
    Run a Random Walk with Restart algorithm to find genes that are close to seed genes in the network
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of node IDs for seed genes
    n_iterations : int
        Number of random walks to perform
    restart_prob : float
        Probability of restarting the walk from a seed gene
    max_steps : int
        Maximum number of steps for each random walk
        
    Returns:
    --------
    list of tuples
        List of (node_id, node_name, score) for all nodes, sorted by score (highest first)
    """
    # If no seed genes are provided, return an empty list
    if not seed_genes:
        return []
    
    # Filter seed genes to ensure they are in the network
    seed_genes = [gene for gene in seed_genes if gene in G.nodes()]
    
    if not seed_genes:
        return []
    
    # Dictionary to store visit counts for each node
    visit_counts = {node: 0 for node in G.nodes()}
    
    # Run random walks
    for _ in tqdm(range(n_iterations), desc="Running random walks"):
        # Randomly select a seed gene to start
        current_node = np.random.choice(seed_genes)
        
        # Perform a random walk
        for _ in range(max_steps):
            # Record visit
            visit_counts[current_node] += 1
            
            # With probability restart_prob, restart from a seed gene
            if np.random.random() < restart_prob:
                current_node = np.random.choice(seed_genes)
                continue
            
            # Otherwise, move to a random neighbor
            neighbors = list(G.neighbors(current_node))
            
            if not neighbors:
                # If there are no neighbors, restart from a seed gene
                current_node = np.random.choice(seed_genes)
            else:
                # Move to a random neighbor, weighted by edge weights if available
                weights = [G[current_node][neighbor].get('weight', 1.0) for neighbor in neighbors]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                    current_node = np.random.choice(neighbors, p=weights)
                else:
                    current_node = np.random.choice(neighbors)
    
    # Calculate scores as normalized visit counts
    total_visits = sum(visit_counts.values())
    
    if total_visits == 0:
        return []
        
    scores = {node: count / total_visits for node, count in visit_counts.items()}
    
    # Sort nodes by score (descending)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return as list of (node_id, node_name, score)
    # In this case, node_id and node_name are the same (both gene symbols)
    result = [(node, node, score) for node, score in sorted_nodes]
    
    return result


def fast_random_walk_with_restart(G, seed_genes, restart_prob=0.7, epsilon=1e-10, max_iter=100):
    """
    Run a fast version of Random Walk with Restart using matrix operations
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of node IDs for seed genes
    restart_prob : float
        Probability of restarting the walk from a seed gene
    epsilon : float
        Convergence threshold
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    list of tuples
        List of (node_id, node_name, score) for all nodes, sorted by score (highest first)
    """
    # If no seed genes are provided, return an empty list
    if not seed_genes:
        return []
    
    # Filter seed genes to ensure they are in the network
    seed_genes = [gene for gene in seed_genes if gene in G.nodes()]
    
    if not seed_genes:
        return []
    
    # Get the adjacency matrix
    A = nx.to_scipy_sparse_matrix(G, weight='weight', format='csr')
    
    # Normalize the adjacency matrix by row
    rowsum = np.array(A.sum(axis=1)).flatten()
    rowsum[rowsum == 0] = 1  # avoid division by zero
    D_inv = sp.diags(1.0 / rowsum)
    A_norm = D_inv.dot(A)
    
    # Create mapping from node to index
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create restart vector (uniform distribution over seed genes)
    restart_vec = np.zeros(len(nodes))
    for gene in seed_genes:
        if gene in node_to_idx:
            restart_vec[node_to_idx[gene]] = 1.0 / len(seed_genes)
    
    # Initialize probability vector
    p_vec = restart_vec.copy()
    
    # Power iteration until convergence
    for _ in tqdm(range(max_iter), desc="Running matrix RWR"):
        p_next = (1 - restart_prob) * A_norm.dot(p_vec) + restart_prob * restart_vec
        delta = np.linalg.norm(p_next - p_vec, 1)
        p_vec = p_next
        
        if delta < epsilon:
            break
    
    # Create scores dictionary
    scores = {node: p_vec[node_to_idx[node]] for node in nodes}
    
    # Sort nodes by score (descending)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return as list of (node_id, node_name, score)
    result = [(node, node, score) for node, score in sorted_nodes]
    
    return result


def rwr_multiple_iterations(G, seed_genes, n_iterations=10, **kwargs):
    """
    Run multiple iterations of RWR and aggregate the results
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of node IDs for seed genes
    n_iterations : int
        Number of iterations to run
    **kwargs : dict
        Additional parameters for random_walk_with_restart function
        
    Returns:
    --------
    list of tuples
        List of (node_id, node_name, score) for all nodes, sorted by score (highest first)
    """
    # If no seed genes are provided, return an empty list
    if not seed_genes:
        return []
    
    # Dictionary to store aggregated scores
    aggregated_scores = {node: 0.0 for node in G.nodes()}
    
    # Run multiple iterations
    for i in tqdm(range(n_iterations), desc="Running multiple RWR iterations"):
        result = random_walk_with_restart(G, seed_genes, **kwargs)
        
        # Update aggregated scores
        for node_id, _, score in result:
            aggregated_scores[node_id] += score
    
    # Normalize scores
    total_score = sum(aggregated_scores.values())
    
    if total_score == 0:
        return []
    
    normalized_scores = {node: score / total_score for node, score in aggregated_scores.items()}
    
    # Sort nodes by score (descending)
    sorted_nodes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return as list of (node_id, node_name, score)
    result = [(node, node, score) for node, score in sorted_nodes]
    
    return result