# PPI Network Analysis
# A collection of Python scripts for analyzing protein-protein interaction networks
# with seed genes on Google Colab

# =============================================
# main.py - Main script to orchestrate the analysis
# =============================================

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from goatools import obo_parser, go_enrichment
import time
import os
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from ppi_loaders import load_ppi_network
from network_analysis import (calculate_network_stats, 
                             calculate_centrality_measures,
                             largest_connected_component,
                             calculate_lcc_significance)
from go_analysis import (load_go_annotations, 
                        analyze_go_terms, 
                        compare_go_clusters)
from random_walk import random_walk_with_restart
from clustering import cluster_genes, evaluate_cluster_significance
from testing import (test_false_negatives,
                   test_false_positives)

# Function to run the entire pipeline
def run_analysis(seed_genes_file, output_dir="results"):
    """
    Main function to run the entire analysis pipeline
    
    Parameters:
    -----------
    seed_genes_file : str
        Path to the CSV file containing seed genes
    output_dir : str
        Directory to save results
    """
    print(f"Starting PPI Network Analysis with seed genes from {seed_genes_file}")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load seed genes
    seed_genes_df = pd.read_csv(seed_genes_file)
    print(f"Loaded {len(seed_genes_df)} seed genes")
    
    # Log basic info about seed genes
    sources_count = seed_genes_df['source'].value_counts()
    print("Seed gene sources:")
    for source, count in sources_count.items():
        print(f"  - {source}: {count} genes")
    
    # List of PPI sources to analyze
    ppi_sources = [
        "HIPPIE", "IntACT", "custom", "STRING", "HuRI", 
        "Bioplex", "Biogrid", "HAPPI", "DICS", 
        "IntNetDB", "ImitateDB", "DLIP"
    ]
    
    # Dictionary to store results for each PPI source
    results = {}
    
    # Process each PPI source
    for ppi_source in ppi_sources:
        print(f"\nProcessing PPI source: {ppi_source}")
        results[ppi_source] = analyze_ppi_source(ppi_source, seed_genes_df, output_dir)
    
    # Compare results across PPI sources
    compare_ppi_results(results, output_dir)
    
    # Save full results
    with open(f"{output_dir}/all_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Create summary dataframe
    summary_df = create_summary_dataframe(results)
    summary_df.to_csv(f"{output_dir}/summary_results.csv", index=True)
    
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
    
    return results, summary_df

def analyze_ppi_source(ppi_source, seed_genes_df, output_dir):
    """
    Analyze a specific PPI source
    
    Parameters:
    -----------
    ppi_source : str
        Name of the PPI source to analyze
    seed_genes_df : pandas.DataFrame
        DataFrame containing seed genes
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict
        Dictionary containing results for this PPI source
    """
    ppi_results = {}
    ppi_dir = f"{output_dir}/{ppi_source}"
    if not os.path.exists(ppi_dir):
        os.makedirs(ppi_dir)
    
    # Step 1: Load the PPI network
    G, gene_map = load_ppi_network(ppi_source)
    ppi_results["network_size"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    print(f"  Loaded PPI network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Extract seed genes that map to the PPI network
    seed_gene_set = set(seed_genes_df['gene'])
    mapped_genes = set(gene_map.keys()).intersection(seed_gene_set)
    mapped_gene_ids = [gene_map[gene] for gene in mapped_genes if gene in gene_map]
    
    # Create subgraph with only seed genes
    seed_subgraph = G.subgraph(mapped_gene_ids).copy()
    ppi_results["seed_subgraph_size"] = {"nodes": seed_subgraph.number_of_nodes(), "edges": seed_subgraph.number_of_edges()}
    print(f"  Created subgraph with {seed_subgraph.number_of_nodes()} seed genes and {seed_subgraph.number_of_edges()} edges")
    
    # Save the mapping of seed genes that were found and not found
    mapped_seed_genes = list(mapped_genes)
    unmapped_seed_genes = list(seed_gene_set - mapped_genes)
    ppi_results["mapped_genes"] = {"found": len(mapped_seed_genes), "not_found": len(unmapped_seed_genes)}
    
    # Save mapped and unmapped genes with their sources
    mapped_df = seed_genes_df[seed_genes_df['gene'].isin(mapped_seed_genes)]
    unmapped_df = seed_genes_df[seed_genes_df['gene'].isin(unmapped_seed_genes)]
    mapped_df.to_csv(f"{ppi_dir}/mapped_seed_genes.csv", index=False)
    unmapped_df.to_csv(f"{ppi_dir}/unmapped_seed_genes.csv", index=False)
    
    # Step 2: Network analysis on the seed gene subgraph
    # Calculate basic network statistics
    network_stats = calculate_network_stats(seed_subgraph)
    ppi_results["network_stats"] = network_stats
    
    # Calculate centrality measures
    centrality_measures = calculate_centrality_measures(seed_subgraph)
    ppi_results["centrality_measures"] = centrality_measures
    
    # Find and analyze the largest connected component
    lcc, lcc_size = largest_connected_component(seed_subgraph)
    lcc_significance = calculate_lcc_significance(G, mapped_gene_ids, n_iterations=1000)
    ppi_results["lcc"] = {
        "size": lcc_size,
        "significance": lcc_significance
    }
    print(f"  Largest connected component has {lcc_size} nodes with z-score: {lcc_significance['z_score']:.2f}")
    
    # Step 3: GO term analysis
    go_annotations = load_go_annotations()
    
    # Analyze GO terms for mapped genes
    mapped_go_analysis = analyze_go_terms(mapped_seed_genes, go_annotations)
    ppi_results["mapped_go_analysis"] = mapped_go_analysis
    
    # Analyze GO terms for unmapped genes
    unmapped_go_analysis = analyze_go_terms(unmapped_seed_genes, go_annotations)
    ppi_results["unmapped_go_analysis"] = unmapped_go_analysis
    
    # Step 4: Random walk with restart
    if seed_subgraph.number_of_nodes() > 0:
        priority_genes = random_walk_with_restart(G, mapped_gene_ids, n_iterations=1000)
        ppi_results["priority_genes"] = priority_genes
        
        # Save priority genes to file
        priority_df = pd.DataFrame({
            'gene_id': [g[0] for g in priority_genes],
            'gene_name': [g[1] for g in priority_genes],
            'score': [g[2] for g in priority_genes]
        })
        priority_df.to_csv(f"{ppi_dir}/priority_genes.csv", index=False)
        print(f"  Identified {len(priority_genes)} priority genes using random walk with restart")
    else:
        ppi_results["priority_genes"] = []
        print("  Could not perform random walk: no mapped seed genes in network")
    
    # Step 5: Clustering analysis
    if seed_subgraph.number_of_nodes() > 2:  # Need at least 3 nodes for meaningful clustering
        clusters, metrics = cluster_genes(seed_subgraph)
        cluster_significance = evaluate_cluster_significance(G, mapped_gene_ids, clusters)
        ppi_results["clustering"] = {
            "clusters": clusters,
            "metrics": metrics,
            "significance": cluster_significance
        }
        
        # Compare GO terms between clusters
        if len(set(clusters.values())) > 1:  # Need at least 2 clusters
            go_comparison = compare_go_clusters(mapped_seed_genes, clusters, go_annotations)
            ppi_results["cluster_go_comparison"] = go_comparison
        
        print(f"  Clustered genes into {len(set(clusters.values()))} clusters")
        print(f"  Clustering significance: {cluster_significance['p_value']:.3f}")
    else:
        ppi_results["clustering"] = None
        print("  Could not perform clustering: insufficient nodes")
    
    # Step 6: Evaluation tests
    if seed_subgraph.number_of_nodes() > 10:  # Need a reasonable number of nodes
        # Test with false negatives (leave-out tests)
        fn_results = test_false_negatives(G, mapped_gene_ids, n_iterations=100)
        ppi_results["false_negative_test"] = fn_results
        
        # Test with false positives (added nodes/edges)
        fp_results = test_false_positives(G, mapped_gene_ids, n_iterations=100)
        ppi_results["false_positive_test"] = fp_results
        
        print(f"  Completed evaluation tests:")
        print(f"    - False negative recovery: {fn_results['recovery_rate']:.2f}")
        print(f"    - False positive rejection: {fp_results['rejection_rate']:.2f}")
    else:
        ppi_results["false_negative_test"] = None
        ppi_results["false_positive_test"] = None
        print("  Could not perform evaluation tests: insufficient nodes")
    
    # Save results for this PPI source
    with open(f"{ppi_dir}/results.pkl", "wb") as f:
        pickle.dump(ppi_results, f)
    
    return ppi_results

def compare_ppi_results(results, output_dir):
    """
    Compare results across PPI sources
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each PPI source
    output_dir : str
        Directory to save comparison results
    """
    # Create directory for comparisons
    comparison_dir = f"{output_dir}/comparisons"
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Create comparison plots
    # 1. Network sizes
    network_sizes = {source: data["network_size"]["nodes"] for source, data in results.items()}
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(network_sizes.keys()), y=list(network_sizes.values()))
    plt.title("PPI Network Sizes (Nodes)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/network_sizes.png")
    
    # 2. Seed gene mapping rates
    mapping_rates = {source: data["mapped_genes"]["found"] / (data["mapped_genes"]["found"] + data["mapped_genes"]["not_found"]) 
                    for source, data in results.items()}
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(mapping_rates.keys()), y=list(mapping_rates.values()))
    plt.title("Seed Gene Mapping Rates")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/mapping_rates.png")
    
    # 3. LCC significance z-scores
    lcc_zscores = {source: data["lcc"]["significance"]["z_score"] 
                  for source, data in results.items() 
                  if "lcc" in data and data["lcc"]["significance"] is not None}
    if lcc_zscores:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(lcc_zscores.keys()), y=list(lcc_zscores.values()))
        plt.title("Largest Connected Component Significance (Z-score)")
        plt.xticks(rotation=45)
        plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/lcc_significance.png")
    
    # 4. False negative recovery rates
    fn_recovery = {source: data["false_negative_test"]["recovery_rate"] 
                  for source, data in results.items() 
                  if "false_negative_test" in data and data["false_negative_test"] is not None}
    if fn_recovery:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(fn_recovery.keys()), y=list(fn_recovery.values()))
        plt.title("False Negative Recovery Rates")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/fn_recovery_rates.png")
    
    # 5. False positive rejection rates
    fp_rejection = {source: data["false_positive_test"]["rejection_rate"] 
                   for source, data in results.items() 
                   if "false_positive_test" in data and data["false_positive_test"] is not None}
    if fp_rejection:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(fp_rejection.keys()), y=list(fp_rejection.values()))
        plt.title("False Positive Rejection Rates")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/fp_rejection_rates.png")
    
    print(f"Comparison plots saved to {comparison_dir}")

def create_summary_dataframe(results):
    """
    Create a summary dataframe with key metrics from all PPI sources
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each PPI source
    
    Returns:
    --------
    pandas.DataFrame
        Summary dataframe
    """
    summary_data = []
    
    for source, data in results.items():
        row = {
            "PPI_Source": source,
            "Network_Nodes": data["network_size"]["nodes"],
            "Network_Edges": data["network_size"]["edges"],
            "Seed_Genes_Mapped": data["mapped_genes"]["found"],
            "Seed_Genes_Unmapped": data["mapped_genes"]["not_found"],
            "Mapping_Rate": data["mapped_genes"]["found"] / (data["mapped_genes"]["found"] + data["mapped_genes"]["not_found"]),
            "Subgraph_Nodes": data["seed_subgraph_size"]["nodes"],
            "Subgraph_Edges": data["seed_subgraph_size"]["edges"],
            "Avg_Distance": data["network_stats"]["avg_distance"] if "avg_distance" in data["network_stats"] else None,
            "LCC_Size": data["lcc"]["size"] if "lcc" in data else None,
            "LCC_Z_Score": data["lcc"]["significance"]["z_score"] if "lcc" in data and data["lcc"]["significance"] is not None else None,
        }
        
        # Add centrality measures summary stats
        if "centrality_measures" in data:
            for measure, values in data["centrality_measures"].items():
                if values:
                    row[f"{measure}_mean"] = np.mean(list(values.values()))
                    row[f"{measure}_median"] = np.median(list(values.values()))
        
        # Add evaluation metrics
        if "false_negative_test" in data and data["false_negative_test"] is not None:
            row["FN_Recovery_Rate"] = data["false_negative_test"]["recovery_rate"]
        else:
            row["FN_Recovery_Rate"] = None
            
        if "false_positive_test" in data and data["false_positive_test"] is not None:
            row["FP_Rejection_Rate"] = data["false_positive_test"]["rejection_rate"]
        else:
            row["FP_Rejection_Rate"] = None
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

if __name__ == "__main__":
    # This script is meant to be run in Google Colab
    # Example usage:
    # results, summary_df = run_analysis("seed_genes.csv")
    pass