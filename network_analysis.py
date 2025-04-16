# network_analysis.py
# Functions for analyzing protein-protein interaction networks

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def calculate_network_stats(G):
    """
    Calculate basic network statistics
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    dict
        Dictionary containing network statistics
    """
    stats = {}
    
    # Basic stats
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)
    
    # Check if graph is connected
    if nx.is_connected(G):
        stats['is_connected'] = True
        stats['avg_distance'] = nx.average_shortest_path_length(G)
        stats['diameter'] = nx.diameter(G)
    else:
        stats['is_connected'] = False
        # Calculate stats on largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_subgraph = G.subgraph(largest_cc).copy()
        
        if len(largest_cc) > 1:
            stats['avg_distance'] = nx.average_shortest_path_length(largest_cc_subgraph)
            stats['diameter'] = nx.diameter(largest_cc_subgraph)
        else:
            stats['avg_distance'] = 0
            stats['diameter'] = 0
    
    # Clustering coefficient
    stats['avg_clustering'] = nx.average_clustering(G)
    
    return stats


def calculate_centrality_measures(G):
    """
    Calculate various centrality measures for nodes in the network
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    dict
        Dictionary containing centrality measures
    """
    centrality = {}
    
    # Only calculate these if there are nodes
    if G.number_of_nodes() > 0:
        # Degree centrality
        centrality['degree'] = nx.degree_centrality(G)
        
        # Betweenness centrality
        if G.number_of_nodes() > 1:
            centrality['betweenness'] = nx.betweenness_centrality(G)
        else:
            centrality['betweenness'] = {list(G.nodes())[0]: 0.0}
        
        # Closeness centrality
        if nx.is_connected(G):
            centrality['closeness'] = nx.closeness_centrality(G)
        else:
            # Calculate closeness on connected components
            closeness = {}
            for cc in nx.connected_components(G):
                cc_subgraph = G.subgraph(cc).copy()
                cc_closeness = nx.closeness_centrality(cc_subgraph)
                closeness.update(cc_closeness)
            centrality['closeness'] = closeness
        
        # Eigenvector centrality
        try:
            centrality['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            # May fail to converge for some networks
            centrality['eigenvector'] = None
            print("  Warning: Eigenvector centrality calculation failed")
    else:
        centrality['degree'] = {}
        centrality['betweenness'] = {}
        centrality['closeness'] = {}
        centrality['eigenvector'] = None
    
    return centrality


def largest_connected_component(G):
    """
    Find the largest connected component in the network
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    networkx.Graph
        The largest connected component as a subgraph
    int
        Size of the largest connected component
    """
    if not G.nodes():
        return G, 0
        
    connected_components = list(nx.connected_components(G))
    
    if not connected_components:
        return G, 0
    
    largest_cc = max(connected_components, key=len)
    largest_cc_subgraph = G.subgraph(largest_cc).copy()
    
    return largest_cc_subgraph, len(largest_cc)


def calculate_lcc_significance(G, seed_gene_ids, n_iterations=10000):
    """
    Calculate the significance of the largest connected component size
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_gene_ids : list
        List of node IDs for seed genes
    n_iterations : int
        Number of random samples to generate
        
    Returns:
    --------
    dict
        Dictionary containing significance statistics
    """
    # Number of seed genes
    n_seed_genes = len(seed_gene_ids)
    
    if n_seed_genes == 0:
        return {
            'p_value': 1.0,
            'z_score': 0.0,
            'observed_size': 0,
            'random_mean': 0,
            'random_std': 0,
            'random_sizes': []
        }
    
    # Observed LCC size
    seed_subgraph = G.subgraph(seed_gene_ids).copy()
    _, observed_size = largest_connected_component(seed_subgraph)
    
    # Get all nodes in the network
    all_nodes = list(G.nodes())
    
    # Calculate LCC size for random samples
    random_sizes = []
    for _ in tqdm(range(n_iterations), desc="Calculating LCC significance"):
        # Sample the same number of random nodes
        if len(all_nodes) >= n_seed_genes:
            random_nodes = random.sample(all_nodes, n_seed_genes)
        else:
            random_nodes = all_nodes.copy()
            
        random_subgraph = G.subgraph(random_nodes).copy()
        _, random_size = largest_connected_component(random_subgraph)
        random_sizes.append(random_size)
    
    # Calculate statistics
    random_mean = np.mean(random_sizes)
    random_std = np.std(random_sizes)
    
    # Calculate Z-score
    if random_std > 0:
        z_score = (observed_size - random_mean) / random_std
    else:
        z_score = 0
    
    # Calculate p-value (proportion of random LCCs >= observed LCC)
    p_value = sum(1 for size in random_sizes if size >= observed_size) / n_iterations
    
    return {
        'p_value': p_value,
        'z_score': z_score,
        'observed_size': observed_size,
        'random_mean': random_mean,
        'random_std': random_std,
        'random_sizes': random_sizes
    }


def create_distance_matrix(G):
    """
    Create a distance matrix for all nodes in the network
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Distance matrix for all nodes
    """
    # Get all shortest paths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Create distance matrix
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    
    # Initialize with maximum distances
    distance_matrix.fill(np.inf)
    
    # Fill with actual distances
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if node_i in shortest_paths and node_j in shortest_paths[node_i]:
                distance_matrix[i, j] = shortest_paths[node_i][node_j]
    
    # Replace inf with a large finite number for disconnected components
    max_distance = np.max(distance_matrix[distance_matrix < np.inf])
    distance_matrix[distance_matrix == np.inf] = max_distance * 2
    
    return pd.DataFrame(distance_matrix, index=nodes, columns=nodes)


def average_shortest_path_length_with_disconnected(G):
    """
    Calculate average shortest path length for potentially disconnected graph
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    float
        Average shortest path length
    """
    if not G.nodes():
        return 0
        
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    
    # Handle disconnected graph
    connected_components = list(nx.connected_components(G))
    avg_path_lengths = []
    node_counts = []
    
    for cc in connected_components:
        if len(cc) > 1:  # Only consider components with at least 2 nodes
            cc_subgraph = G.subgraph(cc).copy()
            avg_path_lengths.append(nx.average_shortest_path_length(cc_subgraph))
            node_counts.append(len(cc))
    
    if not avg_path_lengths:
        return 0
    
    # Weighted average by component size
    weighted_avg = sum(l * c for l, c in zip(avg_path_lengths, node_counts)) / sum(node_counts)
    return weighted_avg