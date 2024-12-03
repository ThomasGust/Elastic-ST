import numpy as np
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from weighted_elastic_net import WeightedElasticNet
from typing import Union
from joblib import Parallel, delayed

from elastic_st.data import SpatialTranscriptomicsData
from elastic_st.features import ModelFeature, CellTypeAbundanceFeature, MetageneAbundanceFeature

def fit_enet_parallel(gene_idx, expression_matrix, weights, alpha, l1_ratio, max_iter, tol) -> np.array:
    """
    Function to fit the Elastic Net model for a single gene, parallelizable.

    Parameters:
        gene_idx (int): The index of the target gene.
        expression_matrix (np.array): The feature matrix.
        weights (np.array): The feature weights.
        alpha (float): Regularization strength.
        l1_ratio (float): L1 to L2 regularization ratio.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
    
    Returns:
        np.array: The coefficients for the target gene.
    
    """
    target_expression = expression_matrix[:, gene_idx]

    #The below code makes sure we have proper index alignment even when we remove the target gene from the feature matrix.
    mask = np.ones(expression_matrix.shape[1], dtype=bool)
    mask[gene_idx] = False
    source_expression = expression_matrix[:, mask]
    source_weights = weights[mask]

    # Fit the model
    model = WeightedElasticNet(alpha=alpha, l1_ratio=l1_ratio, weights=source_weights, max_iter=max_iter, tol=tol)
    model.fit(source_expression, target_expression)

    # Reconstruct the full coefficient vector
    coef = np.zeros(expression_matrix.shape[1])
    coef[mask] = model.get_coefficients()

    return coef

def standard_scale(matrix:np.array) -> np.array:
    """
    Apply standard scaling to center data with unit variance. Critical to apply to all model features before training, called internally by ElasticST.
    """
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    return scaled_matrix

class ElasticST:

    def __init__(self, data:SpatialTranscriptomicsData, features:list[ModelFeature], cell_type:str, alpha:float=0.1, l1_ratio:float=0.25, max_iter:int=1000, tol:float=1e-4, subsample_to:Union[int, None]=None):
        """
        Elastic Spatial Transcriptomics model. Fits an elastic net wtih feature specific regularization to all genes in the dataset.
        Parameters:
            data (SpatialTranscriptomicsData): The spatial transcriptomics data object.
            features (list): List of feature objects to use in the model.
            cell_type (str): The cell type to fit the model for.
            alpha (float): Regularization strength.
            l1_ratio (float): L1 to L2 regularization ratio (0 <= l1_ratio <= 1).
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.
            subsample_to (int): Subsample the dataset to this number of cells. Useful for debugging.
        """
        self.data = data
        self.features = features
        self.cell_type = cell_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

        assert 0 <= l1_ratio <= 1, "l1_ratio must be between 0 and 1."

        gene_weights = np.full(data.G.shape[1], 1.0)

        
        if len(self.features) != 0:
            new_feature_weights = np.concatenate([np.full(feature.get_feature().shape[1], feature.weight) for feature in self.features])
        else:
            new_feature_weights = np.array([])
        self.feature_weights = np.concatenate([gene_weights, new_feature_weights])

        #Create the full feature matrix
        gene_features = data.get_expression_by_cell_type(cell_type)
        feature_matrices = [feature.get_feature() for feature in self.features]

        self.feature_matrix = np.hstack([gene_features] + feature_matrices)
        self.feature_matrix = standard_scale(self.feature_matrix)

        if subsample_to is not None:
            subsample_to = min(subsample_to, self.feature_matrix.shape[0])
            indices = np.random.choice(self.feature_matrix.shape[0], subsample_to, replace=False)
            self.feature_matrix = self.feature_matrix[indices]


    def fit(self, n_jobs:int=-1, verbose:bool=True, desc:str="Fitting STLasso") -> dict:
        """
        Fits a model for all target genes in the dataset.

        Parameters:
            n_jobs (int): Number of parallel jobs to run. -1 means using all processors.
            verbose (bool): Whether to show a progress bar.
            desc (str): Description for the progress bar.
        Returns:
            dict: A record containing the coefficients, feature names, target names, and other relevant information:
                - coefficients: The coefficients for each gene in the dataset.
                - feature_names: The names of the features used in the model.
                - target_names: The names of the target genes.
                - cell_type: The cell type the model was fit for.
                - alpha: The regularization strength.
                - l1_ratio: The L1 to L2 regularization ratio.
                - max_iter: The maximum number of iterations.
                - tol: The convergence tolerance.
                - feature_weights: The weights for each feature.
                - feature_type_names: The names of the feature types used in the model.
        """
        #Code done.
        assert n_jobs != 0, "n_jobs must be different from 0."

        #Setting it up to run in parallel makes everything go so much faster.
        if verbose:
            coefficients = Parallel(n_jobs=n_jobs)(delayed(fit_enet_parallel)(gene_idx, self.feature_matrix, self.feature_weights, self.alpha, self.l1_ratio, self.max_iter, self.tol) for gene_idx in tqdm(range(self.data.G.shape[1]), desc=desc))
        else:
            coefficients = Parallel(n_jobs=n_jobs)(delayed(fit_enet_parallel)(gene_idx, self.feature_matrix, self.feature_weights, self.alpha, self.l1_ratio, self.max_iter, self.tol) for gene_idx in range(self.data.G.shape[1]))
        
        #Harvest and then flatten the full feature names to go along with the coefficients.
        feature_names = [self.data.gene_names] + [feature.get_feature_names() for feature in self.features]
        feature_names = [item for sublist in feature_names for item in sublist]
        coeffs = {"coefficients": coefficients, "feature_names": feature_names, "target_names": self.data.gene_names, "cell_type": self.cell_type, "alpha": self.alpha, "l1_ratio": self.l1_ratio, "max_iter": self.max_iter, "tol": self.tol, "feature_weights": self.feature_weights, "feature_type_names": [feature.name for feature in self.features]}
        return coeffs
    

if __name__ == "__main__":
    G = np.load("data/G.npy")
    P = np.load("data/P.npy")
    T = np.load("data/T.npy")
    annotations = json.load(open("data/annotations.json"))
    cell_types = annotations['cell_types']
    gene_names = annotations['gene_names']

    data = SpatialTranscriptomicsData(G, P, T, gene_names, cell_types)

    abundances = CellTypeAbundanceFeature(bias=5, data=data, cell_type="B-cell", radius=0.1)
    metagene_abundance = MetageneAbundanceFeature(bias=5, data=data, cell_type="B-cell", metagenes={"Checkpoints":['CTLA4', "CD274", 'TIGIT']}, radius=0.1)

    data.variance_filter(threshold=0.2)
    model = ElasticST(data, [abundances, metagene_abundance], cell_type="B-cell", alpha=0.05, l1_ratio=0.5, subsample_to=5000)
    #model = STLasso(data, [], cell_type="B-cell", alpha=0.05, l1_ratio=0.5, subsample_to=5000)
    coefficients  = model.fit()
    np.savez_compressed("featured_coefficients.npy", **coefficients)