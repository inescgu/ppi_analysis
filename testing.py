# testing.py
# Functions for testing PPI networks with simulated false positives and negatives

import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from random_walk import random_walk_with_restart


def test_false_negatives(G, seed_genes, n_iterations=10000, leave_out_fraction=0.1):
    """
    Test the network's ability to recover left-out seed genes
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    n_iterations : int
        Number of test iterations to run
    leave_out_fraction : float
        Fraction of seed genes to leave out in each iteration
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Ensure we have enough seed genes
    n_seed_genes = len(seed_genes)
    
    if n_seed_genes < 3:
        return {
            'recovery_rate': 0.0,
            'top_k_recovery_rates': {},
            'details': []
        }
    
    # Calculate the number of genes to leave out
    n_leave_out = max(1, int(leave_out_fraction * n_seed_genes))
    
    # Track recovery results
    recovery_results = []
    top_k_recoveries = defaultdict(int)
    detailed_results = []
    
    for i in tqdm(range(n_iterations), desc="Testing false negatives"):
        # Randomly select genes to leave out
        leave_out_genes = random.sample(seed_genes, n_leave_out)
        training_genes = [gene for gene in seed_genes if gene not in leave_out_genes]
        
        # Run random walk with restart on training genes
        priority_genes = random_walk_with_restart(G, training_genes)
        
        if not priority_genes:
            continue
            
        # Extract gene IDs and rankings
        priority_ids = [g[0] for g in priority_genes]
        
        # Check if left-out genes are in priority list
        recovery_status = []
        for gene in leave_out_genes:
            if gene in priority_ids:
                rank = priority_ids.index(gene) + 1
                recovery_status.append((gene, True, rank))
                
                # Track recovery in top-k
                for k in [1, 5, 10, 20, 50, 100]:
                    if rank <= k:
                        top_k_recoveries[k] += 1
            else:
                recovery_status.append((gene, False, None))
        
        # Calculate recovery rate for this iteration
        iteration_recovery_rate = sum(1 for _, recovered, _ in recovery_status if recovered) / len(leave_out_genes)
        recovery_results.append(iteration_recovery_rate)
        
        # Store detailed results
        detailed_results.append({
            'leave_out_genes': leave_out_genes,
            'recovery_status': recovery_status,
            'recovery_rate': iteration_recovery_rate
        })
    
    # Calculate overall recovery rate
    if not recovery_results:
        return {
            'recovery_rate': 0.0,
            'top_k_recovery_rates': {},
            'details': []
        }
        
    overall_recovery_rate = sum(recovery_results) / len(recovery_results)
    
    # Calculate top-k recovery rates
    n_total = n_iterations * n_leave_out
    top_k_recovery_rates = {k: count / n_total for k, count in top_k_recoveries.items()}
    
    return {
        'recovery_rate': overall_recovery_rate,
        'top_k_recovery_rates': top_k_recovery_rates,
        'details': detailed_results
    }


def test_false_negatives_edge_removal(G, seed_genes, n_iterations=10000, leave_out_fraction=0.1):
    """
    Test the network's ability to recover left-out edges between seed genes
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    n_iterations : int
        Number of test iterations to run
    leave_out_fraction : float
        Fraction of edges to leave out in each iteration
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Create subgraph with seed genes
    seed_subgraph = G.subgraph(seed_genes).copy()
    
    # Get all edges between seed genes
    seed_edges = list(seed_subgraph.edges())
    
    if not seed_edges:
        return {
            'recovery_rate': 0.0,
            'details': []
        }
    
    # Calculate the number of edges to leave out
    n_leave_out = max(1, int(leave_out_fraction * len(seed_edges)))
    
    # Track recovery results
    recovery_results = []
    detailed_results = []
    
    for i in tqdm(range(n_iterations), desc="Testing edge removal"):
        # Randomly select edges to leave out
        leave_out_edges = random.sample(seed_edges, n_leave_out)
        
        # Create a modified graph with edges removed
        modified_G = G.copy()
        for edge in leave_out_edges:
            modified_G.remove_edge(*edge)
        
        # Run random walk with restart on the modified graph
        priority_genes = random_walk_with_restart(modified_G, seed_genes)
        
        if not priority_genes:
            continue
            
        # Get the top gene pairs (potential edges)
        priority_pairs = []
        priority_ids = [g[0] for g in priority_genes]
        
        # Consider top 100 genes
        top_genes = priority_ids[:100]
        for i, gene1 in enumerate(top_genes):
            for gene2 in top_genes[i+1:]:
                if (gene1, gene2) in G.edges() or (gene2, gene1) in G.edges():
                    priority_pairs.append((gene1, gene2))
        
        # Check if left-out edges are in priority pairs
        recovery_status = []
        for edge in leave_out_edges:
            recovered = (edge in priority_pairs) or ((edge[1], edge[0]) in priority_pairs)
            recovery_status.append((edge, recovered))
        
        # Calculate recovery rate for this iteration
        iteration_recovery_rate = sum(1 for _, recovered in recovery_status if recovered) / len(leave_out_edges)
        recovery_results.append(iteration_recovery_rate)
        
        # Store detailed results
        detailed_results.append({
            'leave_out_edges': leave_out_edges,
            'recovery_status': recovery_status,
            'recovery_rate': iteration_recovery_rate
        })
    
    # Calculate overall recovery rate
    if not recovery_results:
        return {
            'recovery_rate': 0.0,
            'details': []
        }
        
    overall_recovery_rate = sum(recovery_results) / len(recovery_results)
    
    return {
        'recovery_rate': overall_recovery_rate,
        'details': detailed_results
    }


def test_false_positives(G, seed_genes, n_iterations=10000, add_fraction=0.1):
    """
    Test the network's ability to reject false positive genes
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    n_iterations : int
        Number of test iterations to run
    add_fraction : float
        Fraction of seed genes to add as false positives
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Ensure we have enough seed genes
    n_seed_genes = len(seed_genes)
    
    if n_seed_genes < 3:
        return {
            'rejection_rate': 0.0,
            'details': []
        }
    
    # Get all nodes that are not seed genes
    non_seed_genes = [node for node in G.nodes() if node not in seed_genes]
    
    if not non_seed_genes:
        return {
            'rejection_rate': 0.0,
            'details': []
        }
    
    # Calculate the number of false genes to add
    n_add = max(1, int(add_fraction * n_seed_genes))
    
    # Track rejection results
    rejection_results = []
    detailed_results = []
    
    for i in tqdm(range(n_iterations), desc="Testing false positives"):
        # Randomly select genes to add as false positives
        try:
            false_genes = random.sample(non_seed_genes, n_add)
        except ValueError:
            # If we can't sample enough non-seed genes, use what we have
            false_genes = non_seed_genes.copy()
            
        # Add false genes to create an augmented set
        augmented_genes = seed_genes + false_genes
        
        # Run random walk with restart on augmented genes
        priority_genes = random_walk_with_restart(G, augmented_genes)
        
        if not priority_genes:
            continue
            
        # Extract gene IDs and rankings
        priority_ids = [g[0] for g in priority_genes]
        
        # Get the bottom-ranked genes (expected to be false positives)
        bottom_k = n_add * 2  # Look at bottom 2x the number of added genes
        bottom_ranked = priority_ids[-bottom_k:] if len(priority_ids) > bottom_k else priority_ids
        
        # Check if false genes are in the bottom-ranked list
        rejection_status = []
        for gene in false_genes:
            if gene in bottom_ranked:
                rejection_status.append((gene, True))
            else:
                rejection_status.append((gene, False))
        
        # Calculate rejection rate for this iteration
        iteration_rejection_rate = sum(1 for _, rejected in rejection_status if rejected) / len(false_genes)
        rejection_results.append(iteration_rejection_rate)
        
        # Store detailed results
        detailed_results.append({
            'false_genes': false_genes,
            'rejection_status': rejection_status,
            'rejection_rate': iteration_rejection_rate
        })
    
    # Calculate overall rejection rate
    if not rejection_results:
        return {
            'rejection_rate': 0.0,
            'details': []
        }
        
    overall_rejection_rate = sum(rejection_results) / len(rejection_results)
    
    return {
        'rejection_rate': overall_rejection_rate,
        'details': detailed_results
    }


def test_false_positive_edges(G, seed_genes, n_iterations=100, add_edge_fraction=0.1):
    """
    Test the network's ability to reject false positive edges between seed genes
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    n_iterations : int
        Number of test iterations to run
    add_edge_fraction : float
        Fraction of potential edges to add as false positives
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Create subgraph with seed genes
    seed_subgraph = G.subgraph(seed_genes).copy()
    
    # Find potential false edges (node pairs that are not connected)
    existing_edges = set(seed_subgraph.edges())
    potential_false_edges = []
    
    for i, gene1 in enumerate(seed_genes):
        for gene2 in seed_genes[i+1:]:
            if gene1 != gene2 and (gene1, gene2) not in existing_edges and (gene2, gene1) not in existing_edges:
                potential_false_edges.append((gene1, gene2))
    
    if not potential_false_edges:
        return {
            'rejection_rate': 0.0,
            'details': []
        }
    
    # Calculate the number of false edges to add
    n_add = max(1, int(add_edge_fraction * len(existing_edges)))
    n_add = min(n_add, len(potential_false_edges))
    
    # Track rejection results
    rejection_results = []
    detailed_results = []
    
    for i in tqdm(range(n_iterations), desc="Testing false positive edges"):
        # Randomly select edges to add as false positives
        false_edges = random.sample(potential_false_edges, n_add)
        
        # Create a modified graph with false edges added
        modified_G = G.copy()
        for edge in false_edges:
            modified_G.add_edge(*edge, weight=0.5)  # Assign a default weight
        
        # Run random walk with restart on the modified graph
        priority_genes = random_walk_with_restart(modified_G, seed_genes)
        
        if not priority_genes:
            continue
            
        # Get the top gene pairs (potential edges)
        priority_pairs = []
        priority_ids = [g[0] for g in priority_genes]
        
        # Consider top 100 genes
        top_genes = priority_ids[:100]
        for i, gene1 in enumerate(top_genes):
            for gene2 in top_genes[i+1:]:
                if (gene1, gene2) in modified_G.edges() or (gene2, gene1) in modified_G.edges():
                    priority_pairs.append((gene1, gene2))
        
        # Check if false edges are NOT in the priority pairs (should be rejected)
        rejection_status = []
        for edge in false_edges:
            rejected = not ((edge in priority_pairs) or ((edge[1], edge[0]) in priority_pairs))
            rejection_status.append((edge, rejected))
        
        # Calculate rejection rate for this iteration
        iteration_rejection_rate = sum(1 for _, rejected in rejection_status if rejected) / len(false_edges)
        rejection_results.append(iteration_rejection_rate)
        
        # Store detailed results
        detailed_results.append({
            'false_edges': false_edges,
            'rejection_status': rejection_status,
            'rejection_rate': iteration_rejection_rate
        })
    
    # Calculate overall rejection rate
    if not rejection_results:
        return {
            'rejection_rate': 0.0,
            'details': []
        }
        
    overall_rejection_rate = sum(rejection_results) / len(rejection_results)
    
    return {
        'rejection_rate': overall_rejection_rate,
        'details': detailed_results
    }


def evaluate_network_performance(G, seed_genes, test_iterations=10000):
    """
    Comprehensive evaluation of a network's performance with seed genes
    
    Parameters:
    -----------
    G : networkx.Graph
        The full PPI network
    seed_genes : list
        List of seed gene IDs
    test_iterations : int
        Number of test iterations for each test
        
    Returns:
    --------
    dict
        Dictionary containing comprehensive evaluation results
    """
    results = {}
    
    # Test false negative recovery (node removal)
    print("Testing false negative recovery (node removal)...")
    fn_nodes = test_false_negatives(G, seed_genes, n_iterations=test_iterations)
    results['false_negative_nodes'] = fn_nodes
    
    # Test false negative recovery (edge removal)
    print("Testing false negative recovery (edge removal)...")
    fn_edges = test_false_negatives_edge_removal(G, seed_genes, n_iterations=test_iterations)
    results['false_negative_edges'] = fn_edges
    
    # Test false positive rejection (node addition)
    print("Testing false positive rejection (node addition)...")
    fp_nodes = test_false_positives(G, seed_genes, n_iterations=test_iterations)
    results['false_positive_nodes'] = fp_nodes
    
    # Test false positive rejection (edge addition)
    print("Testing false positive rejection (edge addition)...")
    fp_edges = test_false_positive_edges(G, seed_genes, n_iterations=test_iterations)
    results['false_positive_edges'] = fp_edges
    
    # Summarize results
    summary = {
        'fn_node_recovery_rate': fn_nodes['recovery_rate'],
        'fn_edge_recovery_rate': fn_edges['recovery_rate'],
        'fp_node_rejection_rate': fp_nodes['rejection_rate'],
        'fp_edge_rejection_rate': fp_edges['rejection_rate'],
        'overall_score': (fn_nodes['recovery_rate'] + fn_edges['recovery_rate'] + 
                          fp_nodes['rejection_rate'] + fp_edges['rejection_rate']) / 4
    }
    
    results['summary'] = summary
    
    return results


def compare_network_performance(network_results):
    """
    Compare the performance of multiple networks
    
    Parameters:
    -----------
    network_results : dict
        Dictionary mapping network names to their evaluation results
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table
    """
    comparison_data = []
    
    for network_name, results in network_results.items():
        if 'summary' in results:
            row = {
                'Network': network_name,
                'FN Node Recovery': results['summary']['fn_node_recovery_rate'],
                'FN Edge Recovery': results['summary']['fn_edge_recovery_rate'],
                'FP Node Rejection': results['summary']['fp_node_rejection_rate'],
                'FP Edge Rejection': results['summary']['fp_edge_rejection_rate'],
                'Overall Score': results['summary']['overall_score']
            }
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df.sort_values('Overall Score', ascending=False)