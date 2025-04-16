# ppi_loaders.py
# Human-specific PPI network loaders with improved error handling

import pandas as pd
import networkx as nx
import requests
import io
import os
import zipfile
import tempfile
import shutil
from tqdm import tqdm
import sys
import time


def load_ppi_network(ppi_source):
    """
    Load a human-specific PPI network from the specified source
    
    Parameters:
    -----------
    ppi_source : str
        Name of the PPI source to load
        
    Returns:
    --------
    networkx.Graph
        The PPI network as a Graph
    dict
        Mapping of gene symbols to node identifiers
    """
    loaders = {
        "HIPPIE": load_hippie,
        "IntACT": load_intact,
        "custom": load_custom_ppi,
        "STRING": load_string,
        "HuRI": load_huri,
        "Bioplex": load_bioplex,
        "Biogrid": load_biogrid,
        "HAPPI": load_happi,
        "DICS": load_dics,
        "IntNetDB": load_intnetdb,
        "ImitateDB": load_imitatedb,
        "DLIP": load_dlip,
        "Test": create_test_ppi_network
    }
    
    if ppi_source not in loaders:
        raise ValueError(f"Unknown PPI source: {ppi_source}")
    
    try:
        G, gene_map = loaders[ppi_source]()
        print(f"Loaded PPI network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, gene_map
    except Exception as e:
        print(f"Error in {ppi_source} loader: {str(e)}")
        # Fall back to test network
        G, gene_map = create_test_ppi_network(ppi_source)
        print(f"Loaded test network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, gene_map


def download_file(url, save_path=None, timeout=300):
    """
    Download a file from a URL with progress bar and timeout
    
    Parameters:
    -----------
    url : str
        URL to download
    save_path : str or None
        Path to save the file, if None returns the content
    timeout : int
        Timeout in seconds
        
    Returns:
    --------
    str or bytes
        File content if save_path is None
    """
    print(f"Downloading from {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        if save_path:
            # Save to file with progress bar
            with open(save_path, 'wb') as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
            return save_path
        else:
            # Return content with progress bar
            content = io.BytesIO()
            with tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    content.write(data)
                    bar.update(len(data))
            return content.getvalue()
    except Exception as e:
        print(f"Error downloading from {url}: {str(e)}")
        if save_path:
            # Create an empty file to avoid further errors
            with open(save_path, 'wb') as f:
                pass
            return save_path
        else:
            return b''  # Return empty bytes if not saving to file


def safe_rmdir(path):
    """
    Safely remove a directory and all its contents
    
    Parameters:
    -----------
    path : str
        Path to the directory to remove
    """
    try:
        # Sleep briefly to allow file handles to be released
        time.sleep(1)
        shutil.rmtree(path)
    except Exception as e:
        print(f"Warning: Could not remove directory {path}: {str(e)}")
        # Try to remove files individually
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except:
                        pass
            os.rmdir(path)
        except:
            pass


def create_test_ppi_network(name="Test"):
    """
    Create a simple test PPI network with human disease genes
    
    Parameters:
    -----------
    name : str
        Name of the test network
        
    Returns:
    --------
    networkx.Graph
        A simple test PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print(f"Creating test PPI network: {name}")
    
    # Create a graph
    G = nx.Graph()
    
    # Add some nodes (human disease gene symbols)
    genes = ['BRCA1', 'BRCA2', 'TP53', 'ATM', 'CHEK2', 'PALB2', 'PTEN', 'STK11', 
             'CDH1', 'RAD51C', 'RAD51D', 'BARD1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM', 
             'MLH1', 'NBN', 'NF1', 'CDKN2A']
    
    # Add edges (interactions)
    interactions = [
        ('BRCA1', 'BRCA2', 0.9), ('BRCA1', 'TP53', 0.8), ('BRCA2', 'PALB2', 0.9),
        ('TP53', 'ATM', 0.7), ('ATM', 'CHEK2', 0.8), ('PALB2', 'BRCA2', 0.9),
        ('PTEN', 'TP53', 0.6), ('STK11', 'PTEN', 0.5), ('BRCA1', 'BARD1', 0.9),
        ('MSH2', 'MSH6', 0.9), ('MSH2', 'MLH1', 0.8), ('MLH1', 'PMS2', 0.9),
        ('EPCAM', 'MSH2', 0.7), ('NBN', 'ATM', 0.6), ('CDKN2A', 'TP53', 0.6),
        ('NF1', 'PTEN', 0.5), ('RAD51C', 'BRCA2', 0.7), ('RAD51D', 'RAD51C', 0.8)
    ]
    
    # Add nodes and edges to the graph
    for gene in genes:
        G.add_node(gene)
    
    for gene1, gene2, weight in interactions:
        G.add_edge(gene1, gene2, weight=weight)
    
    # Create a simple gene map (in this case, node ID = gene symbol)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"Test network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


def load_hippie():
    """
    Load the HIPPIE PPI network (human-specific)
    
    Returns:
    --------
    networkx.Graph
        The HIPPIE PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading HIPPIE PPI network...")
    
    # URL to the HIPPIE dataset (already human-specific)
    url = "http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/HIPPIE-current.mitab.txt"
    
    # Create a temporary file to store the data
    temp_dir = tempfile.mkdtemp(prefix="hippie_")
    temp_path = os.path.join(temp_dir, "hippie.txt")
    
    try:
        # Download the data
        download_file(url, temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("Failed to download HIPPIE data")
            return create_test_ppi_network("HIPPIE")
        
        # Read the MITAB format file, skipping header rows
        try:
            df = pd.read_csv(temp_path, sep='\t', comment='#', header=None)
            # If first row contains column headers, skip it
            if "ID" in str(df.iloc[0, 0]) and "interactor" in str(df.iloc[0, 0]):
                df = df.iloc[1:]
            
            # Ensure at least 15 columns (for confidence in index 14)
            if df.shape[1] < 15:
                print(f"HIPPIE data has insufficient columns: {df.shape[1]}")
                return create_test_ppi_network("HIPPIE")
            
            # Extract required columns
            df = df.iloc[:, [0, 1, 2, 3, 14]]
            df.columns = ['id_a', 'id_b', 'alt_id_a', 'alt_id_b', 'confidence']
            
        except Exception as e:
            print(f"Error reading HIPPIE data: {str(e)}")
            return create_test_ppi_network("HIPPIE")
        
        # Extract gene symbols from alternative IDs
        def extract_gene_symbol(alt_id):
            if not isinstance(alt_id, str):
                return None
            
            try:
                for item in alt_id.split('|'):
                    # Look for gene name in parentheses
                    if 'uniprotkb:' in item and '(' in item and ')' in item:
                        start = item.find('(') + 1
                        end = item.find(')', start)
                        if start > 0 and end > start:
                            return item[start:end]
            except:
                return None
            return None
        
        df['gene_a'] = df['alt_id_a'].apply(extract_gene_symbol)
        df['gene_b'] = df['alt_id_b'].apply(extract_gene_symbol)
        
        # Filter interactions with confidence score >= 0.63 (medium confidence)
        try:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            df = df[df['confidence'] >= 0.63]
        except Exception as e:
            print(f"Error converting confidence scores: {str(e)}")
            # Keep all if conversion fails
        
        # Filter out interactions where gene symbols couldn't be extracted
        df = df.dropna(subset=['gene_a', 'gene_b'])
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                confidence = float(row['confidence']) if not pd.isna(row['confidence']) else 0.5
                G.add_edge(row['gene_a'], row['gene_b'], weight=confidence)
            except:
                # Skip problematic edges
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"HIPPIE network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_intact():
    """
    Load the IntAct PPI network (filtered for human interactions)
    
    Returns:
    --------
    networkx.Graph
        The IntAct PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading IntAct PPI network...")
    
    # URL to the IntAct dataset
    url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip"
    
    # Create a temporary directory to extract the files
    temp_dir = tempfile.mkdtemp(prefix="intact_")
    zip_path = os.path.join(temp_dir, "intact.zip")
    
    try:
        # Download the data
        download_file(url, zip_path)
        
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            print("Failed to download IntAct data")
            return create_test_ppi_network("IntACT")
        
        # Extract the zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            print(f"Error extracting IntAct zip file: {str(e)}")
            return create_test_ppi_network("IntACT")
        
        # Path to the extracted file
        intact_file = os.path.join(temp_dir, "intact.txt")
        
        if not os.path.exists(intact_file):
            print(f"Extracted file not found at {intact_file}")
            # Find any .txt file in the directory
            txt_files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]
            if txt_files:
                intact_file = os.path.join(temp_dir, txt_files[0])
                print(f"Using {intact_file} instead")
            else:
                print("No .txt files found in extracted content")
                return create_test_ppi_network("IntACT")
        
        # Read the first few lines to determine column names
        column_names = []
        try:
            with open(intact_file, 'r') as f:
                header = f.readline().strip()
                if '\t' in header:
                    column_names = header.split('\t')
        except:
            print("Could not read IntAct header")
        
        # Read the file with pandas
        try:
            # Use column names if found, otherwise use numeric indices
            if column_names:
                df = pd.read_csv(intact_file, sep='\t', names=column_names, comment='#', skiprows=1, low_memory=False)
            else:
                df = pd.read_csv(intact_file, sep='\t', comment='#', low_memory=False)
            
            # Look for taxid columns to filter for human interactions
            taxid_cols = []
            for i, col in enumerate(df.columns):
                if 'taxid' in str(col).lower() and 'interactor' in str(col).lower():
                    taxid_cols.append(i)
            
            # If we found taxid columns
            if len(taxid_cols) >= 2:
                # Keep only rows where both interactors are human (taxid 9606)
                human_mask = df.iloc[:, taxid_cols].apply(
                    lambda x: all('9606' in str(val).lower() or 'human' in str(val).lower() for val in x),
                    axis=1
                )
                df = df[human_mask]
            
            # Identify alias/gene name columns
            alias_cols = []
            for i, col in enumerate(df.columns):
                if 'alias' in str(col).lower() and 'interactor' in str(col).lower():
                    alias_cols.append(i)
            
            if len(alias_cols) < 2:
                print("Could not find alias columns in IntAct data")
                return create_test_ppi_network("IntACT")
            
            # Extract gene symbols from aliases
            def extract_gene_symbol(aliases):
                if not isinstance(aliases, str):
                    return None
                
                try:
                    for alias in aliases.split('|'):
                        if 'gene name:' in alias.lower():
                            # Extract the gene name
                            name = alias.split('gene name:')[1].split('(')[0].strip()
                            return name
                except:
                    return None
                return None
            
            # Apply gene symbol extraction to alias columns
            df['gene_a'] = df.iloc[:, alias_cols[0]].apply(extract_gene_symbol)
            df['gene_b'] = df.iloc[:, alias_cols[1]].apply(extract_gene_symbol)
            
            # Filter out interactions where gene symbols couldn't be extracted
            df = df.dropna(subset=['gene_a', 'gene_b'])
            
            # Extract confidence scores if available
            conf_col = None
            for i, col in enumerate(df.columns):
                if 'confidence' in str(col).lower() and 'value' in str(col).lower():
                    conf_col = i
                    break
            
            if conf_col is not None:
                def extract_confidence(conf_str):
                    try:
                        if isinstance(conf_str, str) and 'intact-miscore:' in conf_str:
                            return float(conf_str.split('intact-miscore:')[1])
                        return 0.5  # default confidence
                    except:
                        return 0.5
                
                df['confidence'] = df.iloc[:, conf_col].apply(extract_confidence)
            else:
                df['confidence'] = 0.5  # default confidence
            
            # Create a graph
            G = nx.Graph()
            
            # Add edges to the graph
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
            
            # Create a mapping of gene symbols to node IDs (in this case, they're the same)
            gene_map = {gene: gene for gene in G.nodes()}
            
            print(f"IntAct network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            return G, gene_map
            
        except Exception as e:
            print(f"Error processing IntAct data: {str(e)}")
            return create_test_ppi_network("IntACT")
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_custom_ppi():
    """
    Load a custom PPI network from a user-uploaded file
    
    Returns:
    --------
    networkx.Graph
        The custom PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading custom PPI network...")
    print("Please upload a CSV file with columns: 'gene_a', 'gene_b', and optionally 'weight'")
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Creating test network.")
            return create_test_ppi_network("Custom")
        
        filename = list(uploaded.keys())[0]
        
        # Read the uploaded CSV file
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {', '.join(required_cols)}")
            return create_test_ppi_network("Custom")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        if 'weight' in df.columns:
            for _, row in df.iterrows():
                try:
                    weight = float(row['weight'])
                    G.add_edge(row['gene_a'], row['gene_b'], weight=weight)
                except:
                    G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        else:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"Custom PPI network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading custom PPI: {str(e)}")
        return create_test_ppi_network("Custom")


def load_string():
    """
    Load the STRING PPI network (human-specific)
    
    Returns:
    --------
    networkx.Graph
        The STRING PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading STRING PPI network...")
    
    # URL to the STRING dataset (filtered for human interactions - 9606 is human taxid)
    url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    
    # Create a temporary file to store the compressed data
    temp_dir = tempfile.mkdtemp(prefix="string_")
    temp_path = os.path.join(temp_dir, "string.txt.gz")
    
    try:
        # Download the data
        download_file(url, temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("Failed to download STRING data")
            return create_test_ppi_network("STRING")
        
        # Read the gzipped file
        try:
            # Try to read with headers
            df = pd.read_csv(temp_path, sep=' ', compression='gzip')
            
            # Check if necessary columns exist
            if 'protein1' not in df.columns or 'protein2' not in df.columns:
                # Try without header
                df = pd.read_csv(temp_path, sep=' ', compression='gzip', header=None)
                if df.shape[1] >= 3:  # Need at least protein1, protein2, score
                    df.columns = ['protein1', 'protein2'] + [f'score{i}' for i in range(df.shape[1]-2)]
                    
                    # Find the combined score column (usually the last one)
                    df['combined_score'] = df.iloc[:, -1]
                else:
                    print("STRING data format not recognized")
                    return create_test_ppi_network("STRING")
        except Exception as e:
            print(f"Error reading STRING data: {str(e)}")
            return create_test_ppi_network("STRING")
        
        # Download the mapping file for protein IDs to gene symbols
        mapping_url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
        
        # Create a temporary file for the mapping
        temp_map_path = os.path.join(temp_dir, "string_mapping.tsv.gz")
        
        # Download the mapping data
        download_file(mapping_url, temp_map_path)
        
        if not os.path.exists(temp_map_path) or os.path.getsize(temp_map_path) == 0:
            print("Failed to download STRING mapping data")
            return create_test_ppi_network("STRING")
        
        # Read the mapping file
        try:
            # First try with standard column names
            mapping_df = pd.read_csv(temp_map_path, sep='\t', compression='gzip')
            
            # Check for necessary columns or alternative names
            protein_id_col = None
            gene_name_col = None
            
            for col in mapping_df.columns:
                if 'protein' in col.lower() and ('id' in col.lower() or 'string' in col.lower()):
                    protein_id_col = col
                elif any(name in col.lower() for name in ['preferred', 'gene', 'name', 'symbol']):
                    gene_name_col = col
            
            if protein_id_col is None or gene_name_col is None:
                print(f"Required columns not found in STRING mapping. Available: {mapping_df.columns.tolist()}")
                return create_test_ppi_network("STRING")
            
            # Create a mapping dictionary
            protein_to_gene = dict(zip(mapping_df[protein_id_col], mapping_df[gene_name_col]))
            
        except Exception as e:
            print(f"Error processing STRING mapping: {str(e)}")
            return create_test_ppi_network("STRING")
        
        # Filter for medium-high confidence interactions (score >= 700)
        try:
            df['combined_score'] = pd.to_numeric(df['combined_score'], errors='coerce')
            df = df[df['combined_score'] >= 700]
        except:
            # If conversion fails, try to keep going
            pass
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                # Extract protein IDs - handling different formats
                p1 = str(row['protein1'])
                p2 = str(row['protein2'])
                
                # Get just the protein ID part (remove species prefix if present)
                if '.' in p1:
                    protein_a = p1.split('.')[1]
                else:
                    protein_a = p1
                    
                if '.' in p2:
                    protein_b = p2.split('.')[1]
                else:
                    protein_b = p2
                
                # Map to gene symbols
                gene_a = protein_to_gene.get(protein_a)
                gene_b = protein_to_gene.get(protein_b)
                
                if gene_a and gene_b:
                    try:
                        weight = float(row['combined_score']) / 1000.0  # Normalize to [0,1]
                    except:
                        weight = 0.7  # Default for high confidence
                        
                    G.add_edge(gene_a, gene_b, weight=weight)
            except Exception as e:
                # Skip problematic rows
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"STRING network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_huri():
    """
    Load the Human Reference Interactome (HuRI) network
    
    Returns:
    --------
    networkx.Graph
        The HuRI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading HuRI PPI network...")
    
    # URL to the HuRI dataset
    url = "http://www.interactome-atlas.org/data/HuRI.tsv"
    
    # Create a temporary file to store the data
    temp_dir = tempfile.mkdtemp(prefix="huri_")
    temp_path = os.path.join(temp_dir, "huri.tsv")
    
    try:
        # Download the data
        download_file(url, temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("Failed to download HuRI data")
            return create_test_ppi_network("HuRI")
        
        # Determine the file format by reading the first few lines
        try:
            with open(temp_path, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip() if first_line else ""
            
            # Check if first line looks like a header
            is_header_line = False
            if first_line and any(keyword in first_line.lower() for keyword in ['gene', 'protein', 'symbol', 'id']):
                is_header_line = True
            
            # Read the TSV file
            if is_header_line:
                df = pd.read_csv(temp_path, sep='\t')
            else:
                df = pd.read_csv(temp_path, sep='\t', header=None)
                # Assign default column names
                if df.shape[1] >= 2:
                    df.columns = ['Gene_A', 'Gene_B'] + [f'Col{i}' for i in range(2, df.shape[1])]
            
            # Look for gene name columns
            gene_a_col = None
            gene_b_col = None
            
            for col in df.columns:
                col_str = str(col).lower()
                if ('gene' in col_str or 'symbol' in col_str) and ('a' in col_str or '1' in col_str or 'first' in col_str):
                    gene_a_col = col
                elif ('gene' in col_str or 'symbol' in col_str) and ('b' in col_str or '2' in col_str or 'second' in col_str):
                    gene_b_col = col
            
            # If standard naming failed, assume first two columns are gene IDs
            if gene_a_col is None or gene_b_col is None:
                if df.shape[1] >= 2:
                    gene_a_col = df.columns[0]
                    gene_b_col = df.columns[1]
                else:
                    print("HuRI data does not have enough columns")
                    return create_test_ppi_network("HuRI")
            
        except Exception as e:
            print(f"Error reading HuRI data: {str(e)}")
            return create_test_ppi_network("HuRI")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row[gene_a_col]).strip()
                gene_b = str(row[gene_b_col]).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                    
                G.add_edge(gene_a, gene_b, weight=1.0)  # Default weight of 1.0
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"HuRI network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_bioplex():
    """
    Load the BioPlex PPI network (human-specific)
    
    Returns:
    --------
    networkx.Graph
        The BioPlex network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading BioPlex PPI network...")
    
    # URL to the BioPlex dataset (BioPlex 3.0)
    url = "https://bioplex.hms.harvard.edu/data/BioPlex_interactionList_v3.tsv"
    
    # Create a temporary file to store the data
    temp_dir = tempfile.mkdtemp(prefix="bioplex_")
    temp_path = os.path.join(temp_dir, "bioplex.tsv")
    
    try:
        # Download the data
        download_file(url, temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("Failed to download BioPlex data")
            return create_test_ppi_network("Bioplex")
        
        # Read the TSV file
        try:
            # Try to read with headers
            df = pd.read_csv(temp_path, sep='\t')
            
            # Check for gene symbol columns
            symbol_cols = [col for col in df.columns if 'symbol' in col.lower()]
            
            if len(symbol_cols) >= 2:
                # Use the detected symbol columns
                gene_a_col = symbol_cols[0]
                gene_b_col = symbol_cols[1]
            else:
                # Try common column names
                potential_cols = [
                    ('SymbolA', 'SymbolB'),
                    ('GeneA', 'GeneB'),
                    ('Gene_A', 'Gene_B'),
                    ('Protein1', 'Protein2')
                ]
                
                found = False
                for col_a, col_b in potential_cols:
                    if col_a in df.columns and col_b in df.columns:
                        gene_a_col = col_a
                        gene_b_col = col_b
                        found = True
                        break
                        
                if not found:
                    # Try to use the first two columns
                    if df.shape[1] >= 2:
                        gene_a_col = df.columns[0]
                        gene_b_col = df.columns[1]
                    else:
                        print("BioPlex data format not recognized")
                        return create_test_ppi_network("Bioplex")
            
            # Check for confidence column
            conf_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['confidence', 'score', 'prob', 'wrong']):
                    conf_col = col
                    break
            
        except Exception as e:
            print(f"Error reading BioPlex data: {str(e)}")
            return create_test_ppi_network("Bioplex")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph using gene symbols
        for _, row in df.iterrows():
            try:
                gene_a = str(row[gene_a_col]).strip()
                gene_b = str(row[gene_b_col]).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                
                # Use confidence if available, otherwise default
                if conf_col:
                    try:
                        confidence = float(row[conf_col])
                        if 'wrong' in conf_col.lower():
                            # If this is p(Wrong) or similar, convert to confidence
                            confidence = 1.0 - confidence
                    except:
                        confidence = 0.5
                else:
                    confidence = 0.5
                    
                G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"BioPlex network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_biogrid():
    """
    Load the BioGRID PPI network (filtered for human interactions)
    
    Returns:
    --------
    networkx.Graph
        The BioGRID network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading BioGRID PPI network...")
    
    # URL to the BioGRID dataset (filtered for human interactions)
    url = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab2.zip"
    
    # Create a temporary directory to extract the files
    temp_dir = tempfile.mkdtemp(prefix="biogrid_")
    zip_path = os.path.join(temp_dir, "biogrid.zip")
    
    try:
        # Download the data
        download_file(url, zip_path)
        
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            print("Failed to download BioGRID data")
            return create_test_ppi_network("Biogrid")
        
        # Extract the zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # Find the human interactions file
            human_file = None
            for file in os.listdir(temp_dir):
                if "HUMAN" in file and file.endswith(".tab2.txt"):
                    human_file = os.path.join(temp_dir, file)
                    break
            
            if not human_file:
                # Try to find any organism file and then filter for human
                tab_files = [f for f in os.listdir(temp_dir) if f.endswith(".tab2.txt")]
                if tab_files:
                    human_file = os.path.join(temp_dir, tab_files[0])
                    print(f"Human-specific file not found. Using {human_file} instead and will filter for human interactions.")
                else:
                    print("No .tab2.txt files found in BioGRID archive")
                    return create_test_ppi_network("Biogrid")
                
        except Exception as e:
            print(f"Error extracting BioGRID data: {str(e)}")
            return create_test_ppi_network("Biogrid")
        
        # Read the file with pandas
        try:
            # Try reading with automatic header detection
            df = pd.read_csv(human_file, sep='\t', low_memory=False)
            
            # Process column names to find what we need
            gene_a_col = None
            gene_b_col = None
            species_a_col = None
            species_b_col = None
            exp_system_col = None
            
            for col in df.columns:
                col_str = str(col).lower()
                if 'symbol' in col_str and ('a' in col_str or 'interactor a' in col_str):
                    gene_a_col = col
                elif 'symbol' in col_str and ('b' in col_str or 'interactor b' in col_str):
                    gene_b_col = col
                elif 'organism' in col_str and ('a' in col_str or 'interactor a' in col_str):
                    species_a_col = col
                elif 'organism' in col_str and ('b' in col_str or 'interactor b' in col_str):
                    species_b_col = col
                elif 'experimental' in col_str and 'system' in col_str:
                    exp_system_col = col
            
            # If we couldn't find the columns, try more generic patterns
            if gene_a_col is None or gene_b_col is None:
                for col in df.columns:
                    col_str = str(col).lower()
                    if ('official' in col_str or 'gene' in col_str) and ('a' in col_str or '1' in col_str):
                        gene_a_col = col
                    elif ('official' in col_str or 'gene' in col_str) and ('b' in col_str or '2' in col_str):
                        gene_b_col = col
            
            # If we still can't find them, use positional columns
            if gene_a_col is None or gene_b_col is None:
                print("Could not identify gene columns in BioGRID data")
                return create_test_ppi_network("Biogrid")
            
            # Filter for human interactions if species columns are available
            if species_a_col and species_b_col:
                human_mask = df[species_a_col].str.contains('sapiens|human', case=False, na=False) & \
                             df[species_b_col].str.contains('sapiens|human', case=False, na=False)
                df = df[human_mask]
                
        except Exception as e:
            print(f"Error reading BioGRID data: {str(e)}")
            return create_test_ppi_network("Biogrid")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row[gene_a_col]).strip()
                gene_b = str(row[gene_b_col]).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                
                # Assign confidence based on experimental system
                if exp_system_col and pd.notna(row[exp_system_col]):
                    system = str(row[exp_system_col]).lower()
                    
                    # Higher confidence for physical interactions, lower for genetic interactions
                    if any(method in system for method in ['two-hybrid', 'affinity', 'capture', 'co-crystal', 
                                                          'reconstituted', 'pca']):
                        confidence = 0.8
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.5
                
                # Add the edge if it doesn't exist or update with higher confidence
                if G.has_edge(gene_a, gene_b):
                    # Use the maximum confidence if there are multiple lines of evidence
                    G[gene_a][gene_b]['weight'] = max(G[gene_a][gene_b].get('weight', 0), confidence)
                else:
                    G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"BioGRID network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_happi():
    """
    Load the Human Annotated and Predicted Protein Interactions (HAPPI) network
    
    Returns:
    --------
    networkx.Graph
        The HAPPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading HAPPI PPI network...")
    
    # HAPPI requires registration and API key, so we'll simulate access via a custom upload
    print("HAPPI database requires registration. Please upload a HAPPI network file with columns:")
    print("'gene_a', 'gene_b', 'confidence' (optional)")
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Creating test network.")
            return create_test_ppi_network("HAPPI")
        
        filename = list(uploaded.keys())[0]
        
        # Read the uploaded CSV file
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {', '.join(required_cols)}")
            return create_test_ppi_network("HAPPI")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row['gene_a']).strip()
                gene_b = str(row['gene_b']).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                    
                if 'confidence' in df.columns and pd.notna(row['confidence']):
                    try:
                        confidence = float(row['confidence'])
                    except:
                        confidence = 1.0
                else:
                    confidence = 1.0
                    
                G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"HAPPI network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading HAPPI network: {str(e)}")
        return create_test_ppi_network("HAPPI")


def load_dics():
    """
    Load the Database of Interacting Cytokines and Signals (DICS) network
    
    Returns:
    --------
    networkx.Graph
        The DICS network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading DICS PPI network...")
    
    # DICS might not have a direct download URL, so we'll simulate access via a custom upload
    print("Please upload a DICS network file with columns:")
    print("'gene_a', 'gene_b', 'confidence' (optional)")
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Creating test network.")
            return create_test_ppi_network("DICS")
        
        filename = list(uploaded.keys())[0]
        
        # Read the uploaded CSV file
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {', '.join(required_cols)}")
            return create_test_ppi_network("DICS")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row['gene_a']).strip()
                gene_b = str(row['gene_b']).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                    
                if 'confidence' in df.columns and pd.notna(row['confidence']):
                    try:
                        confidence = float(row['confidence'])
                    except:
                        confidence = 1.0
                else:
                    confidence = 1.0
                    
                G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"DICS network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading DICS network: {str(e)}")
        return create_test_ppi_network("DICS")


def load_intnetdb():
    """
    Load the Integrated Network Database (IntNetDB) network
    
    Returns:
    --------
    networkx.Graph
        The IntNetDB network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading IntNetDB PPI network...")
    
    # IntNetDB might not have a direct download URL, so we'll simulate access via a custom upload
    print("Please upload an IntNetDB network file with columns:")
    print("'gene_a', 'gene_b', 'confidence' (optional)")
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Creating test network.")
            return create_test_ppi_network("IntNetDB")
        
        filename = list(uploaded.keys())[0]
        
        # Read the uploaded CSV file
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {', '.join(required_cols)}")
            return create_test_ppi_network("IntNetDB")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row['gene_a']).strip()
                gene_b = str(row['gene_b']).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                    
                if 'confidence' in df.columns and pd.notna(row['confidence']):
                    try:
                        confidence = float(row['confidence'])
                    except:
                        confidence = 1.0
                else:
                    confidence = 1.0
                    
                G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"IntNetDB network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading IntNetDB network: {str(e)}")
        return create_test_ppi_network("IntNetDB")


def load_imitatedb():
    """
    Load the ImitateDB network
    
    Returns:
    --------
    networkx.Graph
        The ImitateDB network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading ImitateDB PPI network...")
    
    # ImitateDB might not have a direct download URL, so we'll simulate access via a custom upload
    print("Please upload an ImitateDB network file with columns:")
    print("'gene_a', 'gene_b', 'confidence' (optional)")
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Creating test network.")
            return create_test_ppi_network("ImitateDB")
        
        filename = list(uploaded.keys())[0]
        
        # Read the uploaded CSV file
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {', '.join(required_cols)}")
            return create_test_ppi_network("ImitateDB")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row['gene_a']).strip()
                gene_b = str(row['gene_b']).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                    
                if 'confidence' in df.columns and pd.notna(row['confidence']):
                    try:
                        confidence = float(row['confidence'])
                    except:
                        confidence = 1.0
                else:
                    confidence = 1.0
                    
                G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"ImitateDB network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading ImitateDB network: {str(e)}")
        return create_test_ppi_network("ImitateDB")


def load_dlip():
    """
    Load the DLIP (Deep Learning-based Protein Interaction Prediction) network
    
    Returns:
    --------
    networkx.Graph
        The DLIP network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading DLIP PPI network...")
    
    # DLIP might not have a direct download URL, so we'll simulate access via a custom upload
    print("Please upload a DLIP network file with columns:")
    print("'gene_a', 'gene_b', 'confidence' (optional)")
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Creating test network.")
            return create_test_ppi_network("DLIP")
        
        filename = list(uploaded.keys())[0]
        
        # Read the uploaded CSV file
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {', '.join(required_cols)}")
            return create_test_ppi_network("DLIP")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = str(row['gene_a']).strip()
                gene_b = str(row['gene_b']).strip()
                
                # Skip if not valid gene symbols
                if not gene_a or not gene_b or gene_a == 'nan' or gene_b == 'nan':
                    continue
                    
                if 'confidence' in df.columns and pd.notna(row['confidence']):
                    try:
                        confidence = float(row['confidence'])
                    except:
                        confidence = 1.0
                else:
                    confidence = 1.0
                    
                G.add_edge(gene_a, gene_b, weight=confidence)
            except:
                continue
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"DLIP network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading DLIP network: {str(e)}")
        return create_test_ppi_network("DLIP")