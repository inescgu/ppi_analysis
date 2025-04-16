# go_analysis.py
# Functions for Gene Ontology (GO) term analysis

import pandas as pd
import numpy as np
from scipy import stats
import requests
import io
import tempfile
import os
from tqdm import tqdm
from collections import Counter, defaultdict


def load_go_annotations(species="human"):
    """
    Load Gene Ontology annotations for the specified species
    
    Parameters:
    -----------
    species : str, default="human"
        Species for which to load GO annotations
        
    Returns:
    --------
    dict
        Dictionary mapping gene symbols to lists of GO terms
    """
    print("Loading GO annotations...")
    
    # URL for human GO annotations (GAF format)
    if species.lower() == "human":
        url = "http://geneontology.org/gene-associations/goa_human.gaf.gz"
    else:
        raise ValueError(f"GO annotations for species '{species}' not supported")
    
    # Create a temporary file to store the compressed data
    with tempfile.NamedTemporaryFile(suffix='.gaf.gz', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Download the data
    try:
        print(f"Downloading GO annotations from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        os.unlink(temp_path)
        raise e
    
    # Read the gzipped file
    try:
        # Skip header lines that start with '!'
        go_data = pd.read_csv(temp_path, sep='\t', compression='gzip', 
                             comment='!', header=None)
    except Exception as e:
        os.unlink(temp_path)
        raise e
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
    # Columns in GAF 2.2 format:
    # 0: DB, 1: DB Object ID, 2: DB Object Symbol, 3: Qualifier, 4: GO ID,
    # 5: DB Reference, 6: Evidence Code, 7: With/From, 8: Aspect, 9: DB Object Name,
    # 10: DB Object Synonym, 11: DB Object Type, 12: Taxon, 13: Date, 14: Assigned By,
    # 15: Annotation Extension, 16: Gene Product Form ID
    
    # Keep only relevant columns
    go_data = go_data[[2, 4, 8]]  # Symbol, GO ID, Aspect
    go_data.columns = ['gene', 'go_id', 'aspect']
    
    # Create a dictionary mapping genes to GO terms
    gene_to_go = defaultdict(list)
    for _, row in tqdm(go_data.iterrows(), total=len(go_data), desc="Processing GO annotations"):
        gene_to_go[row['gene']].append({
            'go_id': row['go_id'],
            'aspect': row['aspect']  # P: Biological Process, F: Molecular Function, C: Cellular Component
        })
    
    print(f"Loaded GO annotations for {len(gene_to_go)} genes")
    
    return gene_to_go


def analyze_go_terms(gene_list, go_annotations):
    """
    Analyze GO terms for a list of genes
    
    Parameters:
    -----------
    gene_list : list
        List of gene symbols to analyze
    go_annotations : dict
        Dictionary mapping gene symbols to lists of GO terms
        
    Returns:
    --------
    dict
        Dictionary containing GO term analysis results
    """
    # Count genes with annotations
    genes_with_annotations = [gene for gene in gene_list if gene in go_annotations]
    
    if not genes_with_annotations:
        return {
            'genes_with_annotations': 0,
            'genes_without_annotations': len(gene_list),
            'go_terms': {},
            'aspect_counts': {},
            'top_terms': {}
        }
    
    # Collect all GO terms for the genes
    all_go_terms = []
    aspect_counts = Counter()
    
    for gene in genes_with_annotations:
        for term in go_annotations[gene]:
            all_go_terms.append(term['go_id'])
            aspect_counts[term['aspect']] += 1
    
    # Count occurrences of each GO term
    go_term_counts = Counter(all_go_terms)
    
    # Get top GO terms overall
    top_terms = go_term_counts.most_common(20)
    
    # Get top GO terms by aspect
    top_by_aspect = {}
    for gene in genes_with_annotations:
        for term in go_annotations[gene]:
            aspect = term['aspect']
            if aspect not in top_by_aspect:
                top_by_aspect[aspect] = Counter()
            top_by_aspect[aspect][term['go_id']] += 1
    
    top_terms_by_aspect = {}
    for aspect, terms in top_by_aspect.items():
        top_terms_by_aspect[aspect] = terms.most_common(10)
    
    # Return results
    return {
        'genes_with_annotations': len(genes_with_annotations),
        'genes_without_annotations': len(gene_list) - len(genes_with_annotations),
        'go_terms': dict(go_term_counts),
        'aspect_counts': dict(aspect_counts),
        'top_terms': dict(top_terms),
        'top_by_aspect': top_terms_by_aspect
    }


def compare_go_clusters(gene_list, clusters, go_annotations):
    """
    Compare GO term enrichment between clusters
    
    Parameters:
    -----------
    gene_list : list
        List of gene symbols
    clusters : dict
        Dictionary mapping gene symbols to cluster IDs
    go_annotations : dict
        Dictionary mapping gene symbols to lists of GO terms
        
    Returns:
    --------
    dict
        Dictionary containing results of GO term comparison between clusters
    """
    # Get unique cluster IDs
    cluster_ids = set(clusters.values())
    
    # Group genes by cluster
    genes_by_cluster = defaultdict(list)
    for gene in gene_list:
        if gene in clusters:
            genes_by_cluster[clusters[gene]].append(gene)
    
    # Analyze GO terms for each cluster
    cluster_go_analysis = {}
    for cluster_id in cluster_ids:
        if cluster_id in genes_by_cluster:
            cluster_genes = genes_by_cluster[cluster_id]
            cluster_go_analysis[cluster_id] = analyze_go_terms(cluster_genes, go_annotations)
    
    # Compare clusters using chi-squared tests for top GO terms
    comparison_results = {}
    
    # Need at least 2 clusters for comparison
    if len(cluster_ids) < 2:
        return {
            'cluster_analysis': cluster_go_analysis,
            'comparisons': comparison_results
        }
    
    # Get all GO terms found in any cluster
    all_terms = set()
    for cluster_id, analysis in cluster_go_analysis.items():
        all_terms.update(analysis['go_terms'].keys())
    
    # Perform chi-squared test for each GO term
    for term in all_terms:
        # Create contingency table
        contingency = []
        for cluster_id in cluster_ids:
            if cluster_id in cluster_go_analysis:
                # Count of genes with this term in this cluster
                term_count = cluster_go_analysis[cluster_id]['go_terms'].get(term, 0)
                
                # Count of genes without this term in this cluster
                no_term_count = (
                    cluster_go_analysis[cluster_id]['genes_with_annotations'] - term_count
                )
                
                contingency.append([term_count, no_term_count])
        
        # Only perform test if we have a proper contingency table
        if len(contingency) >= 2 and all(sum(row) > 0 for row in contingency):
            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                comparison_results[term] = {
                    'chi2': chi2,
                    'p_value': p,
                    'dof': dof
                }
            except:
                # Skip if there's an error in the chi-squared calculation
                pass
    
    # Sort results by p-value
    sorted_comparisons = {
        term: results for term, results in 
        sorted(comparison_results.items(), key=lambda x: x[1]['p_value'])
    }
    
    return {
        'cluster_analysis': cluster_go_analysis,
        'comparisons': sorted_comparisons
    }


def enrich_go_terms(gene_list, background_genes, go_annotations, p_threshold=0.05):
    """
    Perform GO term enrichment analysis
    
    Parameters:
    -----------
    gene_list : list
        List of gene symbols to analyze
    background_genes : list
        List of background gene symbols to compare against
    go_annotations : dict
        Dictionary mapping gene symbols to lists of GO terms
    p_threshold : float
        P-value threshold for significance
        
    Returns:
    --------
    dict
        Dictionary containing GO term enrichment results
    """
    # Filter genes to those with annotations
    genes_with_annotations = [gene for gene in gene_list if gene in go_annotations]
    background_with_annotations = [gene for gene in background_genes if gene in go_annotations]
    
    if not genes_with_annotations or not background_with_annotations:
        return {
            'enriched_terms': {},
            'genes_tested': len(genes_with_annotations),
            'background_size': len(background_with_annotations)
        }
    
    # Count GO term occurrences in gene list
    term_counts = Counter()
    for gene in genes_with_annotations:
        for term in go_annotations[gene]:
            term_counts[term['go_id']] += 1
    
    # Count GO term occurrences in background
    background_counts = Counter()
    for gene in background_with_annotations:
        for term in go_annotations[gene]:
            background_counts[term['go_id']] += 1
    
    # Perform Fisher's exact test for each term
    enrichment_results = {}
    
    for term, count in term_counts.items():
        if term in background_counts:
            # Create contingency table:
            # [[gene_list_with_term, gene_list_without_term],
            #  [background_with_term, background_without_term]]
            gene_list_with_term = count
            gene_list_without_term = len(genes_with_annotations) - count
            background_with_term = background_counts[term] - count
            background_without_term = (
                len(background_with_annotations) - len(genes_with_annotations) - background_with_term
            )
            
            # Ensure no negative values in contingency table
            if background_with_term < 0:
                background_with_term = 0
            if background_without_term < 0:
                background_without_term = 0
            
            contingency = [
                [gene_list_with_term, gene_list_without_term],
                [background_with_term, background_without_term]
            ]
            
            try:
                odds_ratio, p_value = stats.fisher_exact(contingency)
                
                if p_value <= p_threshold:
                    enrichment_results[term] = {
                        'p_value': p_value,
                        'odds_ratio': odds_ratio,
                        'count_in_gene_list': count,
                        'count_in_background': background_counts[term]
                    }
            except:
                # Skip if there's an error in the Fisher's exact test
                pass
    
    # Sort results by p-value
    sorted_results = {
        term: results for term, results in 
        sorted(enrichment_results.items(), key=lambda x: x[1]['p_value'])
    }
    
    return {
        'enriched_terms': sorted_results,
        'genes_tested': len(genes_with_annotations),
        'background_size': len(background_with_annotations)
    }