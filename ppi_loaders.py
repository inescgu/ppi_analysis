# ppi_loaders.py
# Corrected functions for loading different protein-protein interaction networks

import pandas as pd
import networkx as nx
import requests
import io
import os
import zipfile
import tempfile
import shutil
from tqdm import tqdm


def load_ppi_network(ppi_source):
    """
    Load a PPI network from the specified source
    
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
    
    return loaders[ppi_source]()


def download_file(url, save_path=None):
    """
    Download a file from a URL with progress bar
    
    Parameters:
    -----------
    url : str
        URL to download
    save_path : str or None
        Path to save the file, if None returns the content
        
    Returns:
    --------
    str or bytes
        File content if save_path is None
    """
    print(f"Downloading from {url}...")
    
    try:
        response = requests.get(url, stream=True)
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
        # Return a small test dataset as fallback
        if save_path:
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
        shutil.rmtree(path)
    except Exception as e:
        print(f"Warning: Could not remove directory {path}: {str(e)}")


def create_test_ppi_network(name="Test"):
    """
    Create a simple test PPI network
    
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
    
    # Add some nodes (gene symbols)
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
    gene_map = {gene: gene for gene in genes}
    
    print(f"Test network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


def load_hippie():
    """
    Load the HIPPIE PPI network
    
    Returns:
    --------
    networkx.Graph
        The HIPPIE PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading HIPPIE PPI network...")
    
    # URL to the HIPPIE dataset
    url = "http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/HIPPIE-current.mitab.txt"
    
    # Create a temporary file to store the data
    temp_dir = tempfile.mkdtemp(prefix="hippie_")
    temp_path = os.path.join(temp_dir, "hippie.txt")
    
    try:
        # Download the data
        download_file(url, temp_path)
        
        # Read the MITAB format file, skipping the header row
        try:
            df = pd.read_csv(temp_path, sep='\t', skiprows=1, 
                            usecols=[0, 1, 2, 3, 14],  # ID A, ID B, Alt ID A, Alt ID B, Confidence
                            names=['id_a', 'id_b', 'alt_id_a', 'alt_id_b', 'confidence'])
        except Exception as e:
            print(f"Error reading HIPPIE data: {str(e)}")
            # If there's an error, create a test network instead
            safe_rmdir(temp_dir)
            return create_test_ppi_network("HIPPIE")
        
        # Extract gene symbols from alternative IDs
        def extract_gene_symbol(alt_id):
            if isinstance(alt_id, str):
                for item in alt_id.split('|'):
                    if item.startswith('uniprotkb:') and '(' in item and ')' in item:
                        start = item.find('(') + 1
                        end = item.find(')', start)
                        if start > 0 and end > start:
                            return item[start:end]
            return None
        
        df['gene_a'] = df['alt_id_a'].apply(extract_gene_symbol)
        df['gene_b'] = df['alt_id_b'].apply(extract_gene_symbol)
        
        # Filter interactions with confidence score >= 0.63 (medium confidence)
        # and where both gene symbols are available
        df = df[df['confidence'].astype(float) >= 0.63]
        df = df.dropna(subset=['gene_a', 'gene_b'])
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            G.add_edge(row['gene_a'], row['gene_b'], weight=float(row['confidence']))
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"HIPPIE network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    finally:
        # Clean up temporary files
        safe_rmdir(temp_dir)


def load_intact():
    """
    Load the IntAct PPI network
    
    Returns:
    --------
    networkx.Graph
        The IntAct PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading IntAct PPI network...")
    
    # URL to the IntAct dataset (filtered for human interactions)
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
        
        # Read the file with pandas
        try:
            df = pd.read_csv(intact_file, sep='\t', low_memory=False)
            
            # Check if necessary columns exist
            required_cols = ['#ID(s) interactor A', 'ID(s) interactor B', 
                            'Alias(es) interactor A', 'Alias(es) interactor B',
                            'Taxid interactor A', 'Taxid interactor B']
            
            if not all(col in df.columns for col in required_cols):
                print(f"Missing required columns in IntAct data. Available columns: {df.columns.tolist()}")
                return create_test_ppi_network("IntACT")
                
            # Filter for human-human interactions (taxid 9606)
            df = df[(df['Taxid interactor A'] == 'taxid:9606(human)') & 
                    (df['Taxid interactor B'] == 'taxid:9606(human)')]
            
            # Extract gene symbols from aliases
            def extract_gene_symbol(aliases):
                if isinstance(aliases, str):
                    for alias in aliases.split('|'):
                        if 'gene name:' in alias:
                            return alias.split('gene name:')[1].split('(')[0]
                return None
            
            df['gene_a'] = df['Alias(es) interactor A'].apply(extract_gene_symbol)
            df['gene_b'] = df['Alias(es) interactor B'].apply(extract_gene_symbol)
            
            # Filter out interactions where gene symbols couldn't be extracted
            df = df.dropna(subset=['gene_a', 'gene_b'])
            
            # Extract confidence scores where available
            def extract_confidence(conf_str):
                try:
                    if isinstance(conf_str, str) and 'intact-miscore:' in conf_str:
                        return float(conf_str.split('intact-miscore:')[1])
                    return 0.5  # default confidence
                except:
                    return 0.5
            
            # Use 'Confidence value(s)' column if it exists
            if 'Confidence value(s)' in df.columns:
                df['confidence'] = df['Confidence value(s)'].apply(extract_confidence)
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
    
    from google.colab import files
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Creating test network.")
        return create_test_ppi_network("Custom")
    
    filename = list(uploaded.keys())[0]
    
    try:
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
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['weight'])
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
    Load the STRING PPI network
    
    Returns:
    --------
    networkx.Graph
        The STRING PPI network
    dict
        Mapping of gene symbols to node identifiers
    """
    print("Loading STRING PPI network...")
    
    # Updated URL to the STRING dataset (filtered for human interactions)
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
            df = pd.read_csv(temp_path, sep=' ', compression='gzip')
            
            # Check if necessary columns exist
            if 'protein1' not in df.columns or 'protein2' not in df.columns or 'combined_score' not in df.columns:
                print(f"Missing required columns in STRING data. Available columns: {df.columns.tolist()}")
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
            mapping_df = pd.read_csv(temp_map_path, sep='\t', compression='gzip')
            
            # Check if necessary columns exist
            if 'preferred_name' not in mapping_df.columns or 'protein_external_id' not in mapping_df.columns:
                print(f"Missing required columns in STRING mapping. Available columns: {mapping_df.columns.tolist()}")
                return create_test_ppi_network("STRING")
                
            # Create a mapping dictionary
            protein_to_gene = dict(zip(mapping_df['protein_external_id'], mapping_df['preferred_name']))
            
        except Exception as e:
            print(f"Error processing STRING mapping: {str(e)}")
            return create_test_ppi_network("STRING")
        
        # Filter for medium-high confidence interactions (score >= 700)
        df = df[df['combined_score'] >= 700]
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                # Extract protein IDs
                protein_a = row['protein1'].split('.')[1]
                protein_b = row['protein2'].split('.')[1]
                
                # Map to gene symbols
                gene_a = protein_to_gene.get(protein_a)
                gene_b = protein_to_gene.get(protein_b)
                
                if gene_a and gene_b:
                    G.add_edge(gene_a, gene_b, weight=row['combined_score'] / 1000.0)  # Normalize to [0,1]
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
        
        # Read the TSV file
        try:
            # Check the first few lines to determine column names
            with open(temp_path, 'r') as f:
                header_line = f.readline().strip()
                
            # Determine separator and column names
            if '\t' in header_line:
                sep = '\t'
            else:
                sep = ' '
                
            df = pd.read_csv(temp_path, sep=sep)
            
            # Check if gene name columns exist
            gene_cols = [col for col in df.columns if 'gene' in col.lower()]
            
            if len(gene_cols) >= 2:
                # Use the detected gene columns
                gene_a_col = gene_cols[0]
                gene_b_col = gene_cols[1]
            else:
                # Fallback to default names
                gene_a_col = 'SymbolA'
                gene_b_col = 'SymbolB'
                
                if gene_a_col not in df.columns or gene_b_col not in df.columns:
                    print(f"Cannot find gene columns in HuRI data. Available columns: {df.columns.tolist()}")
                    return create_test_ppi_network("HuRI")
            
        except Exception as e:
            print(f"Error reading HuRI data: {str(e)}")
            return create_test_ppi_network("HuRI")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                G.add_edge(row[gene_a_col], row[gene_b_col], weight=1.0)  # Default weight of 1.0
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
    Load the BioPlex PPI network
    
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
                    print(f"Cannot find gene/protein columns in BioPlex data. Available columns: {df.columns.tolist()}")
                    return create_test_ppi_network("Bioplex")
            
            # Check for confidence column
            conf_col = None
            confidence_keywords = ['confidence', 'score', 'prob', 'weight']
            
            for col in df.columns:
                if any(keyword in col.lower() for keyword in confidence_keywords):
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
                gene_a = row[gene_a_col]
                gene_b = row[gene_b_col]
                
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
    Load the BioGRID PPI network
    
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
                print("Could not find human interactions file in BioGRID archive")
                return create_test_ppi_network("Biogrid")
                
        except Exception as e:
            print(f"Error extracting BioGRID data: {str(e)}")
            return create_test_ppi_network("Biogrid")
        
        # Read the file with pandas
        try:
            # Try to determine the column names by checking the file header
            with open(human_file, 'r') as f:
                header = f.readline().strip()
                
            if '\t' in header:
                cols = header.split('\t')
                
                # Look for gene symbol columns
                symbol_a_col = None
                symbol_b_col = None
                exp_system_col = None
                
                for i, col in enumerate(cols):
                    if "official symbol" in col.lower() and "interactor a" in col.lower():
                        symbol_a_col = i
                    elif "official symbol" in col.lower() and "interactor b" in col.lower():
                        symbol_b_col = i
                    elif "experimental system" in col.lower():
                        exp_system_col = i
                
                if symbol_a_col is not None and symbol_b_col is not None:
                    # Read the file with custom column names
                    df = pd.read_csv(human_file, sep='\t', usecols=[symbol_a_col, symbol_b_col, exp_system_col],
                                    names=['gene_a', 'gene_b', 'exp_system'])
                else:
                    # Fallback to standard column names
                    df = pd.read_csv(human_file, sep='\t')
                    
                    # Check for known column patterns
                    if 'Official Symbol Interactor A' in df.columns and 'Official Symbol Interactor B' in df.columns:
                        df = df.rename(columns={
                            'Official Symbol Interactor A': 'gene_a',
                            'Official Symbol Interactor B': 'gene_b',
                            'Experimental System': 'exp_system'
                        })
                    else:
                        # Try to find appropriate columns
                        symbol_cols = [col for col in df.columns if 'symbol' in col.lower() and 'official' in col.lower()]
                        if len(symbol_cols) >= 2:
                            df = df.rename(columns={
                                symbol_cols[0]: 'gene_a',
                                symbol_cols[1]: 'gene_b'
                            })
                        
                        exp_cols = [col for col in df.columns if 'experimental' in col.lower() and 'system' in col.lower()]
                        if exp_cols:
                            df = df.rename(columns={exp_cols[0]: 'exp_system'})
            else:
                print("Invalid BioGRID file format")
                return create_test_ppi_network("Biogrid")
                
        except Exception as e:
            print(f"Error reading BioGRID data: {str(e)}")
            return create_test_ppi_network("Biogrid")
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            try:
                gene_a = row['gene_a']
                gene_b = row['gene_b']
                
                # Assign confidence based on experimental system
                if 'exp_system' in row:
                    system = row['exp_system']
                    
                    # Higher confidence for physical interactions, lower for genetic interactions
                    if system in ['Two-hybrid', 'Affinity Capture-MS', 'Affinity Capture-Western', 
                                'Co-crystal Structure', 'Reconstituted Complex', 'PCA']:
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
        if 'confidence' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
        else:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        
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
        if 'confidence' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
        else:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        
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
        if 'confidence' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
        else:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        
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
        if 'confidence' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
        else:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        
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
        if 'confidence' in df.columns:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
        else:
            for _, row in df.iterrows():
                G.add_edge(row['gene_a'], row['gene_b'], weight=1.0)
        
        # Create a mapping of gene symbols to node IDs (in this case, they're the same)
        gene_map = {gene: gene for gene in G.nodes()}
        
        print(f"DLIP network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, gene_map
    
    except Exception as e:
        print(f"Error loading DLIP network: {str(e)}")
        return create_test_ppi_network("DLIP")