# clustering.py
# Functions for clustering genes in PPI networks

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
import random
from tqdm import tqdm
from collections import defaultdict


def calculate_distance_matrix(G):
    """
    Calculate the shortest path distance matrix for a graph
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
        
    Returns:
    --------
    numpy.ndarray
        Distance matrix
    list
        List of node IDs corresponding to matrix rows/columns
    """
    # Get list of nodes
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create node to index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize distance matrix with infinity
    dist_matrix = np.full((n, n), np.inf)
    
    # Set diagonal to 0
    for i in range(n):
        dist_matrix[i, i] = 0
    
    # Calculate shortest paths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Fill distance matrix
    for u, paths in shortest_paths.items():
        u_idx = node_to_idx[u]
        for v, length in paths.items():
            v_idx = node_to_idx[v]
            dist_matrix[u_idx, v_idx] = length
    
    # Replace inf with a large finite value (e.g., n+1)
    dist_matrix[np.isinf(dist_matrix)] = n + 1
    
    return dist_matrix, nodes


def cluster_genes(G, n_clusters=None, method='kmeans'):
    """
    Cluster genes in a PPI network
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to cluster
    n_clusters : int or None
        Number of clusters (if None, will be determined automatically)
    method : str, default='kmeans'
        Clustering method: 'kmeans', 'spectral', 'dbscan', or 'hierarchical'
        
    Returns:
    --------
    dict
        Dictionary mapping node IDs to cluster labels
    dict
        Dictionary containing clustering metrics
    """
    # Calculate distance matrix
    dist_matrix, nodes = calculate_distance_matrix(G)
    
    # Use MDS to embed distances in 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    
    # Determine number of clusters if not provided
    if n_clusters is None:
        # Try different numbers of clusters
        max_clusters = min(len(nodes) - 1, 10)  # Max 10 clusters
        best_score = -1
        best_k = 2  # Default to 2 clusters
        
        for k in range(2, max_clusters + 1):
            # Try k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(coords)
            
            if len(set(cluster_labels)) < 2:
                continue
                
            # Calculate silhouette score
            score = silhouette_score(coords, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        n_clusters = best_k
    
    # Perform clustering
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering.fit_predict(coords)
    elif method == 'spectral':
        # Create similarity matrix from distance matrix
        similarity = np.exp(-dist_matrix / dist_matrix.max())
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        cluster_labels = clustering.fit_predict(similarity)
    elif method == 'dbscan':
        # Determine epsilon based on distance distribution
        eps = np.percentile(dist_matrix.flatten(), 10)  # 10th percentile of distances
        clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(dist_matrix)
        
        # Ensure cluster labels are non-negative
        if -1 in cluster_labels:
            # Assign noise points to a new cluster
            cluster_labels = [label if label >= 0 else max(cluster_labels) + 1 for label in cluster_labels]
    elif method == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        cluster_labels = clustering.fit_predict(dist_matrix)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Create mapping from node ID to cluster label
    clusters = {node: label for node, label in zip(nodes, cluster_labels)}
    
    # Calculate metrics
    metrics = {}
    
    try:
        # Silhouette score (for k-means and spectral)
        if method in ['kmeans', 'spectral', 'hierarchical']:
            metrics['silhouette_score'] = silhouette_score(coords, cluster_labels)
    except:
        metrics['silhouette_score'] = None
    
    # Cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().to_dict()
    metrics['cluster_sizes'] = cluster_sizes
    
    return clusters, metrics


def evaluate_cluster_significance(G, seed_genes, clusters, n_iterations=1000):
    """
    Evaluate the statistical significance of gene clusters
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    clusters : dict
        Dictionary mapping gene IDs to cluster labels
    n_iterations : int
        Number of random permutations to generate
        
    Returns:
    --------
    dict
        Dictionary containing significance metrics
    """
    # Count number of seed genes
    n_seed_genes = len(seed_genes)
    
    if n_seed_genes == 0:
        return {'p_value': 1.0, 'z_score': 0.0}
    
    # Get all nodes in the network
    all_nodes = list(G.nodes())
    
    # Calculate cluster sizes
    cluster_labels = list(clusters.values())
    cluster_sizes = pd.Series(cluster_labels).value_counts().to_dict()
    
    # Calculate the standard deviation of cluster sizes
    std_dev_observed = np.std(list(cluster_sizes.values()))
    
    # Generate random permutations and calculate their cluster size standard deviations
    random_std_devs = []
    for _ in tqdm(range(n_iterations), desc="Evaluating cluster significance"):
        # Sample random nodes
        if len(all_nodes) >= n_seed_genes:
            random_nodes = random.sample(all_nodes, n_seed_genes)
        else:
            random_nodes = all_nodes.copy()
            
        # Create subgraph
        random_subgraph = G.subgraph(random_nodes).copy()
        
        # Cluster the random subgraph
        random_clusters, _ = cluster_genes(random_subgraph)
        
        # Calculate standard deviation of cluster sizes
        random_labels = list(random_clusters.values())
        random_sizes = pd.Series(random_labels).value_counts().to_dict()
        random_std_dev = np.std(list(random_sizes.values()))
        
        random_std_devs.append(random_std_dev)
    
    # Calculate p-value: proportion of random samples with std_dev >= observed
    p_value = sum(1 for std_dev in random_std_devs if std_dev >= std_dev_observed) / n_iterations
    
    # Calculate z-score
    if len(random_std_devs) > 0:
        mean_random = np.mean(random_std_devs)
        std_random = np.std(random_std_devs)
        
        if std_random > 0:
            z_score = (std_dev_observed - mean_random) / std_random
        else:
            z_score = 0.0
    else:
        z_score = 0.0
    
    return {
        'p_value': p_value,
        'z_score': z_score,
        'observed_std_dev': std_dev_observed,
        'random_std_devs': random_std_devs
    }


def cluster_comparison(clusters1, clusters2):
    """
    Compare two clustering results
    
    Parameters:
    -----------
    clusters1 : dict
        First clustering result, mapping nodes to cluster labels
    clusters2 : dict
        Second clustering result, mapping nodes to cluster labels
        
    Returns:
    --------
    float
        Adjusted Rand Index (measure of similarity between clusterings)
    """
    from sklearn.metrics.cluster import adjusted_rand_score
    
    # Get common nodes
    common_nodes = set(clusters1.keys()) & set(clusters2.keys())
    
    if not common_nodes:
        return 0.0
    
    # Extract labels for common nodes
    labels1 = [clusters1[node] for node in common_nodes]
    labels2 = [clusters2[node] for node in common_nodes]
    
    # Calculate Adjusted Rand Index
    ari = adjusted_rand_score(labels1, labels2)
    
    return ari


def optimal_clustering(G, min_clusters=2, max_clusters=10):
    """
    Find the optimal number of clusters for a network
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
    min_clusters : int
        Minimum number of clusters to try
    max_clusters : int
        Maximum number of clusters to try
        
    Returns:
    --------
    int
        Optimal number of clusters
    dict
        Dictionary containing clustering metrics for each number of clusters
    """
    # Calculate distance matrix
    dist_matrix, nodes = calculate_distance_matrix(G)
    
    # Use MDS to embed distances in 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    
    # Try different numbers of clusters
    max_clusters = min(len(nodes) - 1, max_clusters)
    min_clusters = min(min_clusters, max_clusters)
    
    results = {}
    
    for k in range(min_clusters, max_clusters + 1):
        # Try k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)
        
        if len(set(cluster_labels)) < 2:
            continue
            
        # Calculate silhouette score
        try:
            score = silhouette_score(coords, cluster_labels)
            results[k] = {'silhouette_score': score}
        except:
            results[k] = {'silhouette_score': None}
    
    if not results:
        return min_clusters, {}
        
    # Find optimal number of clusters based on silhouette score
    optimal_k = max(results.items(), key=lambda x: x[1]['silhouette_score'] if x[1]['silhouette_score'] is not None else -1)[0]
    
    return optimal_k, results


def network_based_clustering(G, seed_genes, n_clusters=None):
    """
    Perform network-based clustering of seed genes
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    n_clusters : int or None
        Number of clusters (if None, will be determined automatically)
        
    Returns:
    --------
    dict
        Dictionary mapping node IDs to cluster labels
    dict
        Dictionary containing clustering metrics
    """
    # Create subgraph with seed genes
    seed_subgraph = G.subgraph(seed_genes).copy()
    
    # Find connected components
    connected_components = list(nx.connected_components(seed_subgraph))
    
    # If we have multiple connected components, we can use them as initial clusters
    if len(connected_components) > 1 and (n_clusters is None or n_clusters <= len(connected_components)):
        # Use connected components as clusters
        clusters = {}
        for i, component in enumerate(connected_components):
            for node in component:
                clusters[node] = i
        
        # Calculate metrics
        metrics = {
            'n_clusters': len(connected_components),
            'cluster_sizes': {i: len(component) for i, component in enumerate(connected_components)},
            'method': 'connected_components'
        }
        
        return clusters, metrics
    
    # Otherwise, use distance-based clustering
    if n_clusters is None:
        n_clusters, _ = optimal_clustering(seed_subgraph)
    
    # Perform clustering
    return cluster_genes(seed_subgraph, n_clusters=n_clusters)