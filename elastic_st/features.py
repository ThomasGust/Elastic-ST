from tqdm import tqdm
import numpy as np

from elastic_st.data import SpatialTranscriptomicsData

class ModelFeature:
    def __init__(self, name:str, bias:float, data:SpatialTranscriptomicsData):
        """
        Base class for any feature to be used in the Elastic-ST model. All features must inherit from this class and implement the get_feature and get_feature_names methods.

        Parameters:
            name (str): The name of the feature.
            bias (float): The bias of the feature. This is a hyperparameter that determines how much more important this feature is relative to the raw transcriptomics data.
            data (SpatialTranscriptomicsData): The spatial transcriptomics data object.
        """
        self.name = name
        self.bias = bias

        self.weight = 1/bias #This is the regularization weight for the actual model fitting. It is the inverse of how much more important we want it to be relative to the raw transcriptomics data.

        self.data = data

    def __repr__(self) -> str:
        return f"{self.name} (bias: {self.bias})"
    
    def get_feature(self, **kwargs):
        raise NotImplementedError
    
    def get_feature_names(self, **kwargs):
        raise NotImplementedError

class CellTypeAbundanceFeature(ModelFeature):

    def __init__(self, bias:float, data:SpatialTranscriptomicsData, cell_type:str, radius:float=0.1, verbose:bool=False):
        """
        A feature to compute the abundance of each cell type in the neighborhood of each cell of interest.

        Parameters:
            bias (float): The bias of the feature.
            data (SpatialTranscriptomicsData): The spatial transcriptomics data object.
            cell_type (str): The cell type to compute the abundance of.
            radius (float): The radius of the neighborhood to consider.
        """
        super().__init__(f"Neighborhood Cell Type Abundance Feature {radius}", bias, data)

        self.radius = radius
        self.cell_type = cell_type

        self.neighbors = self.data.get_neighbors(radius=radius)

        #Compute the amount of each cell type in the neighborhood of each cell.
        self.cell_type_abundance = self.compute_cell_type_abundance()
    
    def compute_cell_type_abundance(self, verbose=False) -> np.array:
        """
        Computes the cell type abundance vector for each cell of interest in the dataset.


        Parameters:
            verbose (bool): Whether to show a progress bar.
        Returns:
            np.array: The cell type abundance matrix (n_cells x n_cell_types).
        """
        #Using the neighbors, compute the amount of each cell type in the neighborhood of each cell.
        focal_cells = self.data.get_cell_type_indices(self.cell_type)
        num_cell_types = len(self.data.cell_types)
        counts = np.zeros((focal_cells.shape[0], num_cell_types))


        if verbose:
            enumerand = tqdm(focal_cells, desc="Computing Neighborhood Cell Type Abundance")
        else:
            enumerand = focal_cells
        

        for i, c in enumerate(enumerand):
            neighbors = self.neighbors[c]
            neighbor_types = self.data.T[neighbors]
            for cell_type in np.unique(neighbor_types):
                counts[i, cell_type] = np.sum(neighbor_types == cell_type) #Just count the number of neighbors of each type.
        
        return counts
    
    def get_feature(self, **kwargs) -> np.array:
        """
        Returns the feature matrix.

        Returns:
            np.array: The feature matrix (n_cells x n_cell_types).
        """
        return self.cell_type_abundance
    
    def get_feature_names(self, **kwargs) -> list[str]:
        """
        Returns:
            list: The names of the features. In this case, the name of each cell type in the data.
        """
        return self.data.cell_types
    
class MetageneAbundanceFeature(ModelFeature):
    def __init__(self, bias: float, data: SpatialTranscriptomicsData, cell_type: str, metagenes: dict[str, list[str]], radius:float=0.1, verbose:bool=False):
        """
        A feature to compute the abundance of metagenes in the neighborhood of each cell of interest. A metagene in this context is any set of genes that are believed to be co-expressed and or have some interesting functional relationship.

        Parameters:
            bias (float): The bias of the feature.
            data (SpatialTranscriptomicsData): The spatial transcriptomics data object.
            cell_type (str): The cell type to compute the abundance of.
            metagenes (dict): A dictionary of metagene names to lists of gene names.
            radius (float): The radius of the neighborhood to consider.
        """
        super().__init__(f"Metagene Abundance Feature {radius}", bias, data)
        
        self.radius = radius 
        self.cell_type = cell_type

        # Convert gene names to indices once
        self.metagenes = {metagene_name: [self.data.gene2idx[gene] for gene in genes] for metagene_name, genes in metagenes.items()}

        # Precompute neighbors (list of indices)
        self.neighbors = self.data.get_neighbors(radius=radius)
        
        # Compute metagene abundance
        self.metagene_abundance = self.compute_metagene_abundance()

    def compute_metagene_abundance(self, verbose=False) -> np.array:
        """
        Computes the metagene abundance for each cell of interest in the dataset. Returns the feature matrix for this feature.

        Parameters:
            verbose (bool): Whether to show a progress bar.
        Returns:
            np.array: The metagene abundance matrix (n_cells x n_metagenes).
        """
        focal_cells = self.data.get_cell_type_indices(self.cell_type)
        
        # Prepare metagene indices
        metagene_indices = [np.array(indices) for indices in self.metagenes.values()]

        num_metagenes = len(self.metagenes)
        abundance = np.zeros((len(focal_cells), len(metagene_indices)))

        # Convert metagene_indices to a mask for easier indexing
        metagene_mask = np.zeros((len(self.data.G[0]), len(metagene_indices)), dtype=bool)
        for j, gene_indices in enumerate(metagene_indices):
            metagene_mask[gene_indices, j] = True

        if verbose:
            enumerand = tqdm(focal_cells, desc="Computing Neighborhood Metagene Abundance")
        else:
            enumerand = focal_cells
            
        for i, cell_idx in enumerate(enumerand):
            neighbors = self.neighbors[cell_idx]

            if not neighbors:
                continue
            neighbor_expressions = self.data.G[neighbors]

            # Compute the sum of gene expressions for each metagene in one step
            metagene_sums = neighbor_expressions @ metagene_mask

            # Compute the mean across all neighbors for each metagene
            abundance[i, :] = metagene_sums.mean(axis=0)
        
        return abundance

    def get_feature(self, **kwargs) -> np.array:
        """
        Returns:
            np.array: The feature matrix (n_cells x n_metagenes).
        """
        return self.metagene_abundance
    
    def get_feature_names(self, **kwargs) -> list[str]:
        """
        Returns:
            list: The names of the features. In this case, the names of the metagenes.
        """
        return list(self.metagenes.keys())