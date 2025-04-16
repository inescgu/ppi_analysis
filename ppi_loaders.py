# ppi_loaders.py
# Functions for loading different protein-protein interaction networks

import pandas as pd
import networkx as nx
import requests
import io
import os
import zipfile
import tempfile
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
        "DLIP": load_dlip
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
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Download the data
    download_file(url, temp_path)
    
    # Read the MITAB format file
    try:
        df = pd.read_csv(temp_path, sep='\t', header=None, 
                         usecols=[0, 1, 2, 3, 14],  # ID A, ID B, Alt ID A, Alt ID B, Confidence
                         names=['id_a', 'id_b', 'alt_id_a', 'alt_id_b', 'confidence'])
    except Exception as e:
        os.unlink(temp_path)
        raise e
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
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
    df = df[df['confidence'].astype(float) >= 0.63]
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for _, row in df.iterrows():
        if row['gene_a'] and row['gene_b']:
            G.add_edge(row['gene_a'], row['gene_b'], weight=float(row['confidence']))
    
    # Create a mapping of gene symbols to node IDs (in this case, they're the same)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"HIPPIE network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


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
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "intact.zip")
    
    # Download the data
    download_file(url, zip_path)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Path to the extracted file
    intact_file = os.path.join(temp_dir, "intact.txt")
    
    # Read the file with pandas
    try:
        df = pd.read_csv(intact_file, sep='\t', header=0,
                        usecols=['#ID(s) interactor A', 'ID(s) interactor B', 
                                'Alias(es) interactor A', 'Alias(es) interactor B',
                                'Taxid interactor A', 'Taxid interactor B',
                                'Confidence value(s)'])
    except Exception as e:
        # Clean up temporary files
        os.unlink(zip_path)
        os.unlink(intact_file)
        os.rmdir(temp_dir)
        raise e
    
    # Clean up temporary files
    os.unlink(zip_path)
    os.unlink(intact_file)
    os.rmdir(temp_dir)
    
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
    
    df['confidence'] = df['Confidence value(s)'].apply(extract_confidence)
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for _, row in df.iterrows():
        G.add_edge(row['gene_a'], row['gene_b'], weight=row['confidence'])
    
    # Create a mapping of gene symbols to node IDs (in this case, they're the same)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"IntAct network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


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
        print("No file uploaded. Creating empty network.")
        G = nx.Graph()
        return G, {}
    
    filename = list(uploaded.keys())[0]
    
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Check if required columns exist
    required_cols = ['gene_a', 'gene_b']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    
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
    
    # URL to the STRING dataset (filtered for human interactions)
    url = "https://string-db.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
    
    # Create a temporary file to store the compressed data
    with tempfile.NamedTemporaryFile(suffix='.txt.gz', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Download the data
    download_file(url, temp_path)
    
    # Read the gzipped file
    try:
        df = pd.read_csv(temp_path, sep=' ', compression='gzip')
    except Exception as e:
        os.unlink(temp_path)
        raise e
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
    # Download the mapping file for protein IDs to gene symbols
    mapping_url = "https://string-db.org/mapping_files/STRING_display_names/human.name_2_string.tsv.gz"
    
    # Create a temporary file for the mapping
    with tempfile.NamedTemporaryFile(suffix='.tsv.gz', delete=False) as temp_map_file:
        temp_map_path = temp_map_file.name
    
    # Download the mapping data
    download_file(mapping_url, temp_map_path)
    
    # Read the mapping file
    try:
        mapping_df = pd.read_csv(temp_map_path, sep='\t', compression='gzip', 
                                names=['gene', 'protein_id'])
    except Exception as e:
        os.unlink(temp_map_path)
        raise e
    
    # Clean up the temporary mapping file
    os.unlink(temp_map_path)
    
    # Create a mapping dictionary
    protein_to_gene = dict(zip(mapping_df['protein_id'], mapping_df['gene']))
    
    # Filter for medium-high confidence interactions (score >= 700)
    df = df[df['combined_score'] >= 700]
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for _, row in df.iterrows():
        protein_a = row['protein1'].split('.')[1]
        protein_b = row['protein2'].split('.')[1]
        
        gene_a = protein_to_gene.get(protein_a)
        gene_b = protein_to_gene.get(protein_b)
        
        if gene_a and gene_b:
            G.add_edge(gene_a, gene_b, weight=row['combined_score'] / 1000.0)  # Normalize to [0,1]
    
    # Create a mapping of gene symbols to node IDs (in this case, they're the same)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"STRING network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


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
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Download the data
    download_file(url, temp_path)
    
    # Read the TSV file
    try:
        df = pd.read_csv(temp_path, sep='\t')
    except Exception as e:
        os.unlink(temp_path)
        raise e
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for _, row in df.iterrows():
        G.add_edge(row['Gene_A'], row['Gene_B'], weight=1.0)  # Default weight of 1.0
    
    # Create a mapping of gene symbols to node IDs (in this case, they're the same)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"HuRI network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


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
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Download the data
    download_file(url, temp_path)
    
    # Read the TSV file
    try:
        df = pd.read_csv(temp_path, sep='\t')
    except Exception as e:
        os.unlink(temp_path)
        raise e
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph using gene symbols
    for _, row in df.iterrows():
        gene_a = row['SymbolA']
        gene_b = row['SymbolB']
        confidence = row['p(Wrong)']
        # Convert p(Wrong) to confidence [0,1] where 1 is most confident
        confidence_score = 1.0 - confidence
        
        G.add_edge(gene_a, gene_b, weight=confidence_score)
    
    # Create a mapping of gene symbols to node IDs (in this case, they're the same)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"BioPlex network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


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
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "biogrid.zip")
    
    # Download the data
    download_file(url, zip_path)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the human interactions file
    human_file = None
    for file in os.listdir(temp_dir):
        if "HUMAN" in file and file.endswith(".tab2.txt"):
            human_file = os.path.join(temp_dir, file)
            break
    
    if not human_file:
        # Clean up temporary files
        os.unlink(zip_path)
        os.rmdir(temp_dir)
        raise FileNotFoundError("Could not find human interactions file in BioGRID archive")
    
    # Read the file with pandas
    try:
        df = pd.read_csv(human_file, sep='\t', usecols=[
            'Official Symbol Interactor A', 'Official Symbol Interactor B',
            'Experimental System'
        ])
    except Exception as e:
        # Clean up temporary files
        os.unlink(zip_path)
        os.unlink(human_file)
        os.rmdir(temp_dir)
        raise e
    
    # Clean up temporary files
    os.unlink(zip_path)
    os.unlink(human_file)
    os.rmdir(temp_dir)
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges to the graph
    for _, row in df.iterrows():
        gene_a = row['Official Symbol Interactor A']
        gene_b = row['Official Symbol Interactor B']
        
        # Assign confidence based on experimental system
        system = row['Experimental System']
        
        # Higher confidence for physical interactions, lower for genetic interactions
        if system in ['Two-hybrid', 'Affinity Capture-MS', 'Affinity Capture-Western', 
                      'Co-crystal Structure', 'Reconstituted Complex', 'PCA']:
            confidence = 0.8
        else:
            confidence = 0.5
        
        # Add the edge if it doesn't exist or update with higher confidence
        if G.has_edge(gene_a, gene_b):
            # Use the maximum confidence if there are multiple lines of evidence
            G[gene_a][gene_b]['weight'] = max(G[gene_a][gene_b]['weight'], confidence)
        else:
            G.add_edge(gene_a, gene_b, weight=confidence)
    
    # Create a mapping of gene symbols to node IDs (in this case, they're the same)
    gene_map = {gene: gene for gene in G.nodes()}
    
    print(f"BioGRID network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, gene_map


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
    
    from google.colab import files
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Creating empty network.")
        G = nx.Graph()
        return G, {}
    
    filename = list(uploaded.keys())[0]
    
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Check if required columns exist
    required_cols = ['gene_a', 'gene_b']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    
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
    
    from google.colab import files
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Creating empty network.")
        G = nx.Graph()
        return G, {}
    
    filename = list(uploaded.keys())[0]
    
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Check if required columns exist
    required_cols = ['gene_a', 'gene_b']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    
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
    
    from google.colab import files
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Creating empty network.")
        G = nx.Graph()
        return G, {}
    
    filename = list(uploaded.keys())[0]
    
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Check if required columns exist
    required_cols = ['gene_a', 'gene_b']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    
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
    
    from google.colab import files
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Creating empty network.")
        G = nx.Graph()
        return G, {}
    
    filename = list(uploaded.keys())[0]
    
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Check if required columns exist
    required_cols = ['gene_a', 'gene_b']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    
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
    
    from google.colab import files
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Creating empty network.")
        G = nx.Graph()
        return G, {}
    
    filename = list(uploaded.keys())[0]
    
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Check if required columns exist
    required_cols = ['gene_a', 'gene_b']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
    
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