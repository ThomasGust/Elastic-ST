import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial import KDTree


class SpatialTranscriptomicsData:
    def __init__(self, G:np.array, P:np.array, T, gene_names:list[str], cell_types:list[str]):
        """
        Common data type for all spatial transcriptomics datasets. Any dataset must be converted to this format before being used in any analysis for Elastic-ST.
        All data objects require an expression matrix (shape n_cells x n_genes), a position matrix (shape n_cells x 2), cell types (shape n_cells,), gene names (shape n_genes,), and cell type names (shape n_cell_types,).
        Make sure all of these line up properly with each other or you will have headaches later on.

        Parameters:
            G (array): Gene expression matrix (n_cells x n_genes).
            P (array): Cell positions (n_cells x 2).
            T (array): Cell types (n_cells,). Can be strings or integers.
            gene_names (list): Gene names (n_genes,).
            cell_types (list): Cell type names (n_cell_types,).
        """

        #Basic assertions to check sanity:
        assert G.shape[0] == P.shape[0], "Gene expression and position matrices must have the same number of cells."
        assert G.shape[1] == len(gene_names), "Gene expression matrix and gene names must have the same number of genes."
        assert len(T) == G.shape[0], "Cell type vector must have the same number of cells as the gene expression matrix."
        assert len(cell_types) == len(set(T)), "Cell types and cell type vector must have the same number of unique cell types."

        self.G = G
        self.P = P
        self.T = T
        self.gene_names = gene_names
        self.cell_types = cell_types

        self.map_dictionaries() #Also call in setters to gene names and cell types. These dictionaries are needed for any kind of interpetation.

        # If it is not already, convert cell types to integers for standardization across modules.
        if isinstance(self.T[0], str):
            self.T = [self.cell_type2idx[cell_type] for cell_type in self.T]
    
    #These 3 helper functions allow easy access to cell type specific information
    def get_cell_type_indices(self, cell_type:str) -> np.array:
        """
        Gets the indices of cells of a specific type with respect to all cells in the dataset.

        Parameters:
            cell_type (str): The cell type to get the indices for.
        Returns:
            np.array: The indices of cells of the specified type. 
        """
        return np.where(self.T == self.cell_type2idx[cell_type])[0]
    
    def get_expression_by_cell_type(self, cell_type:str) -> np.array:
        """
        Gets the expression matrix of all cells of a specific type.

        Parameters:
            cell_type (str): The cell type to get the expression matrix for.
        Returns:
            np.array: The expression matrix of cells of the specified type.
        """
        indices = self.get_cell_type_indices(cell_type)
        return self.G[indices]
    
    def get_position_by_cell_type(self, cell_type:str) -> np.array:
        """
        Gets the position matrix of all cells of a specific type.

        Parameters:
            cell_type (str): The cell type to get the position matrix for.
        Returns:
            np.array: The position matrix of cells of the specified type.
        """
        indices = self.get_cell_type_indices(cell_type)
        return self.P[indices]
    
    def get_neighbors(self, radius:float) -> list[list[int]]:
        """
        Finds the neighbors of each cell within a certain radius.

        Parameters:
            radius (float): The radius within which to find neighbors.
        Returns:
            list: A list of lists, where each sublist contains the indices of the neighbors
        """
        kdtree = KDTree(self.P)
        neighbors = kdtree.query_ball_point(self.P, radius)
        return neighbors
    
    def map_dictionaries(self) -> None:
        """
        A utility function to map gene names and cell types to indices and vice versa. This is needed for easy access to information later on.
        """
        self.gene2idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.idx2gene = {idx: gene for idx, gene in enumerate(self.gene_names)}

        self.cell_type2idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}
        self.idx2cell_type = {idx: cell_type for idx, cell_type in enumerate(self.cell_types)}

    def set_gene_names(self, gene_names:list[str]) -> None:
        """
        Changes the gene names in the dataset. Useful for filtering genes or grouping them into metagenes.
        """
        self.gene_names = gene_names
        self.map_dictionaries()
    
    def set_cell_types(self, cell_types:list[str]) -> None:
        """
        Changes the cell types in the dataset. Useful for filtering cell types.
        """
        self.cell_types = cell_types
        self.map_dictionaries()

    def filter_genes(self, genes:list[str]) -> None:
        """
        Only the genes in the list will be kept in the dataset. All other genes are discarded.

        Parameters:
            genes (list): List of genes to keep in the dataset
        Returns:
            None
        """
        gene_indices = [self.gene2idx[gene] for gene in genes]

        self.G = self.G[:, gene_indices]
        self.set_gene_names(genes)
    
    def filter_cell_types(self, cell_types:list[str]) -> None:
        """
        Only the cell types in the list will be kept in the dataset. All other cell types are discarded.

        Parameters:
            cell_types (list): List of cell types to keep in the dataset
        Returns:
            None
        """
        cell_type_indices = [self.cell_type2idx[cell_type] for cell_type in cell_types]

        keep_indices = np.isin(self.T, cell_type_indices)

        self.G = self.G[keep_indices]
        self.P = self.P[keep_indices]
        self.T = self.T[keep_indices] #This is kind of stupid but should be done for consistency.
        self.set_cell_types(cell_types)

    def group_genes(self, gene_groups:dict[str, list[str]], by:callable=np.sum) -> None:
        """
        Function to group genes together into metagenes. Many will find this function useful if wishing to study their data at a higher level of abstraction above gene by gene analysis.

        Parameters:
            gene_groups (dict): Dictionary of gene groups. Format: {metagene_name: [gene1, gene2, ...]}.
            by (callable): Function to aggregate genes within each metagene. Default is np.sum.
        """
        metagenes = [tuple([self.gene2idx[gene] for gene in genes]) for genes in gene_groups.values()] #Finds the indices of the genes in each metagene

        new_G = np.zeros((self.G.shape[0], len(metagenes)))

        for i, gene_indices in enumerate(metagenes):
            expression = self.G[:, gene_indices]
            new_G[:, i] = by(expression, axis=1)
        
        self.G = new_G

        #Also update the names so we preserve interpretability later on.
        metagene_names = list(gene_groups.keys())
        self.set_gene_names(metagene_names)
    
    def variance_filter(self, threshold:float=0.1) -> list[str]:
        """
        Removes any genes with low enough variance to not be considered important. This is a common preprocessing step for many analyses.

        Parameters:
            threshold (float): The variance threshold below which genes are considered unimportant and removed.
        Returns:
            list: The names of the genes that were kept after filtering.
        """
        selector = VarianceThreshold(threshold=threshold)
        self.G = selector.fit_transform(self.G)
        self.gene_names = [self.gene_names[i] for i in np.where(selector.get_support())[0]]
        self.map_dictionaries()
        
        return self.gene_names

