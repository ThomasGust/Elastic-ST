import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import json as json
import os
import pandas as pd
from tqdm import tqdm
from typing import Union
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LassoLarsCV, LassoCV, Lasso, LassoLars
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.neighbors import KDTree
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import json
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import gaussian_kde
import warnings

class FeatureSetData:
    def __init__(self, featureset2genes:Union[dict, None]=None, path:Union[str, None]=None, bin_key:Union[str, None]='+'):
        """
        Object for holding metagenes as sets of genes. This can either be instantiated with a dict or a path to a CSV file and a bin_key to specify the binarization value.

        Parameters:
            featureset2genes (dict): A dictionary with feature set names as keys and lists of gene names as values.
            path (str): Path to a CSV file containing the feature set data. Feature set names should be in the first row and gene names in the first column.
            bin_key (str): The value to use for binarization. (i.e. in the CSV file, all values equal to bin_key will be considered as part of the feature set).
        """
        self.path = path
        self.bin_key = bin_key
        self.featureset2genes = featureset2genes

        if path is not None:
            if bin_key is None:
                raise ValueError("bin_key must be specified if path is not None")
        
            self.annotations = pd.read_csv(path, index_col=0)
            self.feature_sets = list(self.annotations.columns)
            self.gene_names = list(self.annotations.index)

            self.featureset2genes = {feature_set: list(self.annotations.index[np.where(self.annotations[feature_set] == self.bin_key)]) for feature_set in self.feature_sets}
        else:
            self.feature_sets = list(self.featureset2genes.keys())
            self.gene_names = list(set([gene for genes in self.featureset2genes.values() for gene in genes]))


    def get_genes_in_feature_set(self, feature_set:str):
        """
        Get all the genes present in a given feature set.

        Parameters:
            feature_set (str): The name of the feature set.
        Returns:
            list: List of gene names in the feature set.
        """
        return list(self.annotations.index[np.where(self.annotations[feature_set] == self.bin_key)])
    
    def get_feature_sets_for_gene(self, gene:str):
        """
        Get all the feature sets that contain a given gene.

        Parameters:
            gene (str): The name of the gene.
        Returns:
            list: List of feature set names containing the gene.
        """
        return list(self.annotations.columns[np.where(self.annotations.loc[gene] == self.bin_key)])
    
class SpatialTranscriptomicsData:

    def __init__(self, G:np.array, P:np.array, T:np.array, cell_types:list[str]=None, gene_names:list[str]=None, annotations:dict=None):
        """
        This is the basic data object to hold spatial transriptomics data.

        For TranscriptSpace, all data has a few key attributes:
        - G: a numpy array of shape (n_cells, n_genes) containing the gene expression data
        - P: a numpy array of shape (n_cells, 2) containing the spatiual coordinates of each cell
        - T: a sparse numpy matrix of shape (n_cells, n_cell_types) containing the cell type information
        - annotations: a dictionary with two lists of strings, 'cell_types' and 'gene_names'
        """

        self.G = G
        self.P = P
        self.T = T

        #Load annotations for cell types and gene names, needed for interpretability
        self.cell_types = cell_types
        self.gene_names = gene_names

        self.map_dicts()
    
    def map_dicts(self):
        """
        Creates a mapping between cell types and indices, and gene names and indices based on the annotations.

        Parameters:
            None
        Returns:
            None
        """
        self.celltype2idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}
        self.idx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.cell_types)}

        self.gene2idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.idx2gene = {idx: gene for idx, gene in enumerate(self.gene_names)}

    
    def remap_metagenes(self, feature_set_data: FeatureSetData, opfunc: callable = np.sum):
        """
        Transforms the gene expression matrix to represent metagenes using a FeatureSetData object.

        Parameters:
            feature_set_data (FeatureSetData): FeatureSetData object containing the metagenes and associated genes.
            opfunc (callable): The operation to apply to the gene expression values for each metagene. Default is np.sum.
        Returns:
            None
        """
    
        new_G = np.zeros((self.G.shape[0], len(feature_set_data.feature_sets)))
        metagene_names = feature_set_data.feature_sets
        normalized_G = self.G / np.sum(self.G, axis=1)[:, np.newaxis]
        
        #This kind of setup is much faster than using nested for loops
        gene_indices_dict = {
            feature_set: [self.gene2idx.get(gene) for gene in feature_set_data.get_genes_in_feature_set(feature_set)
                          if gene in self.gene2idx]
            for feature_set in metagene_names
        }
        
        for idx, (metagene_name, gene_indices) in enumerate(gene_indices_dict.items()):
            new_G[:, idx] = opfunc(normalized_G[:, gene_indices], axis=1) if gene_indices else 0
        
        self.G = new_G
        self.gene_names = metagene_names
        self.map_dicts()

class ModelFeature:

    def __init__(self, name):
        """
        Base class for all different features that can be used in the TranscriptSpace model.
        All user made features should inherit from this class and implement the compute_feature and get_feature methods.
        Parameters:
            name (str): The name of the feature.
        """
        self.name = name

    def compute_feature(self, **kwargs):
        """
        This method should be implemented in all subclasses. It should compute the feature matrix and any other necessary data for the feature.
        """
        raise NotImplementedError

    def get_feature(self, **kwargs):
        """
        This method should be implemented in all subclasses. It should return the feature matrix
        """
        raise NotImplementedError

    def get_feature_names(self, **kwargs):
        """
        Get the names of the features in the feature matrix.
        """
        raise NotImplementedError

class GeneExpressionFeature(ModelFeature):

    def __init__(self, data:SpatialTranscriptomicsData, t:Union[str, int, None]):
        """
        This feature is a model compatible representation of the gene expression data in the SpatialTranscriptomicsData object.

        Parameters:
            data (SpatialTranscriptomicsData): Spatial transcriptomics data object.
            t (Union[str, int, None]): The cell type to extract gene expression data for. If None, all cell types are included.
        """
        super().__init__("gene_expression")

        self.data = data

        if t is not None:
            if isinstance(t, str):
                self.t = self.data.celltype2idx[t]
            else:
                self.t = t
        
        i = np.where(self.data.T == self.t)
        self.G = self.data.G[i]
        self.G = StandardScaler().fit_transform(self.G)
    
    def compute_feature(self, **kwargs):
        """
        Nothing needs to happen here because the feature is already computed in the __init__ method as it is stolen from the SpatialTranscriptomicsData object.
        """
        pass

    def get_feature(self, **kwargs):
        """
        Get the feature matrix for the gene expression data for each different configuration.

        Parameters:
            genes (list): List of gene names to include in the feature matrix. Passed as a keyword argument. Default is all genes.
            exclude_genes (list): List of gene names to exclude from the feature matrix. Passed as a keyword argument. Default is an empty list.
        """

        #Includes functionality for excluding genes when this is used as an input feature
        genes = kwargs.get('genes', self.data.gene_names)
        exclude_genes = kwargs.get('exclude_genes', [])

        genes = [gene for gene in genes if gene not in exclude_genes]
        gene_indices = [self.data.gene2idx[gene] for gene in genes]

        return self.G[:, gene_indices]

    def get_feature_names(self, **kwargs):
        """
        Get the names of the genes in the filtered feature matrix

        Parameters:
            genes (list): List of gene names to include in the feature matrix. Passed as a keyword argument. Default is all genes.
            exclude_genes (list): List of gene names to exclude from the feature matrix. Passed as a keyword argument. Default is an empty list.
        """
        genes = kwargs.get('genes', self.data.gene_names)
        exclude_genes = kwargs.get('exclude_genes', [])
        return [gene for gene in genes if gene not in exclude_genes]
    

class NeighborhoodAbundanceFeature(ModelFeature):
    
        def __init__(self, data:SpatialTranscriptomicsData, cell_type:Union[str, int, None]=None, radius:float=0.1, alpha:float=1.0):
            """
            Feature to compute the neighborhood cell type abundances for a given cell type. For each cell of the cell type, it will count the number of cells of each type surrounding that cell.

            Parameters:
                data (SpatialTranscriptomicsData): Spatial transcriptomics data object.
                cell_type (Union[str, int, None]): The cell type to compute neighborhood abundances for. If None, all cell types are included.
                radius (float): The radius of the neighborhood to consider.
                alpha (float): Regularization parameter.
            """
            super().__init__("neighborhood_abundance")

            self.data = data

            if cell_type is not None:
                if isinstance(cell_type, str):
                    self.cell_type = self.data.celltype2idx[cell_type]
                else:
                    self.cell_type = cell_type
            else:
                self.cell_type = None
            
            self.radius = radius
            self.alpha = alpha
        
        def compute_feature(self, **kwargs):
            """
            Builds the neighborhood abundance matrix to be used by the model. This is a matrix of shape (n_cells, n_cell_types) where each row is the neighborhood abundance of each cell type for a given cell.

            Parameters:
                None
            Returns:
                None
            """
            
            self.G = self.data.G
            self.P = self.data.P
            self.T = self.data.T
    
            self.celltype2idx = self.data.celltype2idx
            self.idx2celltype = self.data.idx2celltype

            if cell_type is not None:
                cell_type_idx = self.celltype2idx[cell_type]
                indices = np.where(self.T == cell_type_idx)
            
            self.relevant_indices = list(indices)[0]
            self.neighborhood_abundances = self._compute_neighborhood_abundances(self.G, self.P, self.T)

            self.featureidx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.idx2celltype)}
            
            self.neighborhood_abundances = StandardScaler().fit_transform(self.neighborhood_abundances) * self.alpha
        
        def get_feature_names(self, **kwargs):
            """
            Fetch the names of the cell types in the feature matrix.
            """
            return list(self.idx2celltype.values())
        
        def get_feature(self, **kwargs):
            """
            This is static, the neighborhood abundances will not actually change because this is always an output feature.
            In the future, it might be cool to eventually let users play with this as an input feature as well, but I'm not sure if that makes sense biologically.
            """
            return self.neighborhood_abundances

        def _compute_neighborhood_abundances(self, G, P, T):
            """
            Internal method to compute the neighborhood abundances for each cell type.
            Uses a KDTree to efficiently find the neighbors of each cell within a given radius.

            Parameters:
                G (np.array): Gene expression matrix.
                P (np.array): Spatial coordinates matrix.
                T (np.array): Cell type matrix.
            Returns:
                np.array: Neighborhood abundance matrix.
            """

            n_cells = G.shape[0]

            #Reshape T to be a 2d array from a 1d array of strings. Sparsify.
            T_ = np.zeros((n_cells, len(self.celltype2idx)))
            for i in range(n_cells):
                T_[i, T[i]] = 1
            
            T = T_
            n_cell_types = T.shape[1]
            neighborhood_abundances = np.zeros((len(self.relevant_indices), n_cell_types))

            #There may be a way to speed this up but the current technique usually runs in a few seconds for what I need it to do.
            tree = KDTree(P)
            for i, idx in enumerate(tqdm(self.relevant_indices, desc="Computing Neighborhood Abundances")):
                neighbors = tree.query_radius(P[idx].reshape(1, -1), r=self.radius)[0]
                for neighbor in neighbors:
                    neighborhood_abundances[i] += T[neighbor][0]
            return neighborhood_abundances

class NeighborhoodMetageneFeature(ModelFeature):
    """Compute the neighborhood abundance of all the genes in a given feature set"""
    def __init__(self, data:SpatialTranscriptomicsData, feature_set:FeatureSetData, alpha:float=1.0, radius:float=0.1, cell_type:Union[str, int, None]=None):
        """
        Parameters:
            data (SpatialTranscriptomicsData): Spatial transcriptomics data object.
            feature_set (FeatureSetData): Feature set data object.
        """
        
        super().__init__("neighborhood_metagene")

        self.data = data
        self.feature_set = feature_set

        if cell_type is not None:
            if isinstance(cell_type, str):
                self.cell_type = self.data.celltype2idx[cell_type]
            else:
                self.cell_type = cell_type
        else:
            self.cell_type = None
        
        self.alpha = alpha
        self.radius = radius


    def compute_feature(self, **kwargs):
        """
        This method currently works ok with numba optimization, but still takes about 12 minutes to run on the full dataset
        Parameters:
            alpha (float): Regularization parameter.
            radius (float): Neighborhood radius.
        """
        self.G = self.data.G
        self.P = self.data.P
        self.T = self.data.T

        self.celltype2idx = self.data.celltype2idx
        self.idx2celltype = self.data.idx2celltype

        self.feature_set = self.feature_set

        if self.cell_type is not None:
            #indices = np.where(self.T == self.celltype2idx[self.cell_type])
            indices = np.where(self.T == self.cell_type)
        else:
            indices = np.arange(self.G.shape[0])
        
        self.relevant_indices = list(indices)[0]

        self.neighborhood_metagenes = self._compute_neighborhood_metagenes(self.G, self.P, self.T)

        self.featureidx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.idx2celltype)}

    @staticmethod
    def accumulate_metagenes(neighborhood_metagenes, neighbors, G, gene_indices, cell_idx, feature_set_idx):
        """
        Internal method to accumulate the metagenes for each cell type. This is a static method to allow for potential parallelization in the future.
        """
        if gene_indices.size > 0:
            neighborhood_metagenes[cell_idx, feature_set_idx] += np.mean(G[neighbors][:, gene_indices])

    
    def _compute_neighborhood_metagenes(self, G, P, T):
        """
        Internal method to compute the neighborhood metagenes for each cell type. This feature always uses mean gene expression of each gene in the feature set for the metagene.

        Parameters:
            G (np.array): Gene expression matrix.
            P (np.array): Spatial coordinates matrix.
            T (np.array): Cell type matrix.
        Returns:
            np.array: Neighborhood metagene matrix. Standardized and multiplied by alpha.
        """

        neighborhood_metagenes = np.zeros((len(self.relevant_indices), len(self.feature_set.feature_sets)))
        tree = KDTree(P)

        # Precompute gene indices for each feature set to avoid repetitive lookup, makes this method a lot faster
        feature_indices = {
            f: np.array([self.data.gene2idx[gene] for gene in self.feature_set.featureset2genes[feature_set_name] if gene in self.data.gene2idx])
            for f, feature_set_name in enumerate(self.feature_set.feature_sets)
        }

        # Batch processing to precompute neighbors list for all relevant cells, this is much faster than doing it in a loop
        neighbors_list = tree.query_radius(P[self.relevant_indices], r=self.radius)

        for i, neighbors in enumerate(tqdm(neighbors_list, desc='Computing Neighborhood Metagenes')):
            cell_idx = self.relevant_indices[i]
            for f, gene_indices in feature_indices.items():
                self.accumulate_metagenes(neighborhood_metagenes, neighbors, G, gene_indices, i, f)

        # Apply a standard scaler to the neighborhood metagenes
        neighborhood_metagenes = StandardScaler().fit_transform(neighborhood_metagenes) * self.alpha
        return neighborhood_metagenes
    
    def get_feature(self, **kwargs):
        """
        Gets the neighborhood metagene matrix, also static because this is always an output feature.
        """
        return self.neighborhood_metagenes
    
    def get_feature_names(self, **kwargs):
        """
        Returns the names of the feature sets in the feature matrix.
        """
        return self.feature_set.feature_sets


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class STLasso:

    def __init__(self, st, in_features:list[ModelFeature], lambd:float=1e-3, cell_type='epithelial.cancer.subtype_1'):
        self.st = st
        self.in_features = in_features
        self.cell_type = cell_type

        for feature in self.in_features:
            feature.compute_feature(cell_type=cell_type)
        
        self.lambd = lambd
        
    def fit(self, **kwargs):
        n_resamples = kwargs.get('n_resamples', None)
        n_retries = kwargs.get('n_retries', 10)
        resample_dim = kwargs.get('resample_dim', None)
        k_best = kwargs.get('k_best', 100)
        filter = kwargs.get('filter', None)
        include_expression = kwargs.get('include_expression', True)

        self.gene_expression = GeneExpressionFeature(self.st, self.cell_type)
        if filter is not None:
            indices = filter(self.st, **kwargs)
            self.st.G = self.st.G[:, indices]
            self.st.gene_names = [self.st.gene_names[i] for i in indices]
            self.st.map_dicts()
        
        #Get the feature dimension of each feature matrix
        if include_expression:
            indim = (self.st.G.shape[1]-1)+sum([feature.get_feature().shape[1] for feature in self.in_features])
        else:
            indim = sum([feature.get_feature().shape[1] for feature in self.in_features])

        outdim = self.st.G.shape[1]

        if include_expression:
            self.coefficients = np.zeros((indim+1, outdim))
        else:
            self.coefficients = np.zeros((indim, outdim))

        #Unwrap the in_feature names
        if include_expression:
            in_feature_names = flatten_list([list(feature.get_feature_names()) for feature in self.in_features]) + self.st.gene_names
        else:
            in_feature_names = flatten_list([list(feature.get_feature_names) for feature in self.in_features])
        out_feature_names = self.st.gene_names

        if n_resamples is not None:
            convergence_warnings = np.zeros((outdim, n_resamples, n_retries))
        else:
            convergence_warnings = np.zeros(outdim)
        
        for gi in tqdm(range(outdim), f"Training Models For Cell Type {self.cell_type}"):
            if len(self.in_features) != 0:
                feature_matrices = [feature.get_feature() for feature in self.in_features]
            else:
                feature_matrices = []

            if include_expression:
                expression_feature = self.gene_expression.get_feature(exclude_genes=[self.st.idx2gene[gi]])
                X = np.concatenate([expression_feature] + list(feature_matrices), axis=1)
            else:
                X = np.concatenate(feature_matrices, axis=1)
            
            y = self.st.G[np.where(self.st.T == self.st.celltype2idx[self.cell_type])][:, gi]
            y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten() # Standardize the output feature

            selector = SelectKBest(score_func=f_regression, k=min(k_best, X.shape[1]))
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            
            _X = X
            X = _X[:, selected_indices]

            resample_coefficients = []
            if resample_dim is None:
                resample_dim = X.shape[0]

            Xs, ys = resample(X, y, n_samples=n_resamples*resample_dim)
            Xs = np.reshape(Xs, (n_resamples, resample_dim, X.shape[1]))
            ys = np.reshape(ys, (n_resamples, resample_dim))


            for r in range(n_resamples):
                X = Xs[r]
                y = ys[r]
                
                model = Lasso(alpha=self.lambd)

                with warnings.catch_warnings():
                        warnings.filterwarnings('error')

                        try:
                            model.fit(X, y)
                            coeffs = model.coef_
                            resample_coefficients.append(coeffs)
                            break
                        except ValueError:
                            coeffs = np.zeros(X.shape[1])
                        resample_coefficients.append(coeffs)

            coeffs = np.mean(resample_coefficients, axis=0)
            new_coeffs = np.zeros(indim)
            new_coeffs[selected_indices] = coeffs
            coeffs = new_coeffs
            coeffs[np.abs(coeffs) < 1e-4] = 0

            if include_expression:
                gi_idx = in_feature_names.index(self.st.idx2gene[gi])
                self.coefficients[:, gi] = np.insert(coeffs, gi_idx, 0)
            else:
                self.coefficients[:, gi] = coeffs
        
        coefficient_record = {'coefficients': self.coefficients, 'in_feature_names': in_feature_names, 'out_feature_names': out_feature_names, 'convergence_warnings': convergence_warnings}
        return coefficient_record

class CoefficientAnalysis:
    def __init__(self, coefficients:np.array, in_feature_names:Union[np.array, list[str]], out_feature_names:Union[np.array, list[str]], graph_threshold:float, norm:bool=True):
        """
        This module contains functions for visualizing and analyzing the coefficients of a collection of sparse linear models trained on spatial transcriptomics data.

        Parameters:
            coefficients (np.array): Coefficient matrix.
            in_feature_names (Union[np.array, list[str]]): List of feature names for the input features.
            out_feature_names (Union[np.array, list[str]]): List of feature names for the output features.
            graph_threshold (float): Threshold for building the coefficient graph. Edges with weights below this threshold will be removed.
            norm (bool): Whether or not to normalize the coefficients between 0 and 1 before building the graph.
        """

        self.coefficients = coefficients
        self.in_feature_names = in_feature_names
        self.out_feature_names = out_feature_names
        self.graph_threshold = graph_threshold
        self.norm = norm

        if self.norm:
            self.coefficients = (self.coefficients - np.min(self.coefficients)) / (np.max(self.coefficients)-np.min(self.coefficients))
        else:
            pass

        #Construct a list of feature not in the output feature names so we can identify spatial features visually in the graph plots
        self.spatial_features = []
        for in_feature in self.in_feature_names:
            if in_feature not in self.out_feature_names:
                self.spatial_features.append(in_feature)

        #And then make a list of the gene features, not all of these will actually be in the graph
        self.gene_features = [feature for feature in self.in_feature_names if feature not in self.spatial_features]
        
        #Finally, make the graph object to represent the coefficients.
        self.graph = self.build_coefficient_graph(graph_threshold)
    
    def get_thresh_coefficients(self):
        """
        Threshold the coefficients matrix using the graph threshold for desired sparsity. All edges will either be 0 or 1.

        Parameters:
            None
        Returns:
            np.array: Thresholded coefficient matrix.
        """
        a = np.copy(self.coefficients)
        a[a <= self.graph_threshold] = 0.0
        a[np.where(a != 0.0)] = 1.0

        return a
        
    def build_coefficient_graph(self, threshold:float):
        """
        Build a coefficient graph from the coefficient matrix.

        Paramters:
            threshold (float): Threshold for building the coefficient graph. Edges with weights below this threshold will be removed.
        Returns:
            nx.Graph: Coefficient graph.
        """
        G = nx.Graph()

        #An edge list is much more efficient than blind iteration
        mask = np.abs(self.coefficients) > threshold
        rows, cols = np.where(mask)
        edge_list = [
            (self.in_feature_names[i], self.out_feature_names[j], self.coefficients[i, j])
            for i, j in zip(rows, cols)
        ]

        G.add_weighted_edges_from(edge_list)
        return G

    def plot_coefficient_graph(self, **kwargs):
        """
        Draws the internal coefficient graph to the users screen.

        Parameters:
            layout (nx.layout): Layout for the graph.
            multicolor (bool): Whether to color spatial features differently from gene features.
            with_labels (bool): Whether to display node labels.
            node_size (int): Size of the nodes.
            font_size (int): Size of the node labels.
            font_color (str): Color of the node labels.
            edge_color (str): Color of the edges.
            edge_width (float): Width of the edges.
        Returns:
            None
        """

        pos = kwargs.get('layout', nx.spring_layout(self.graph))
        multicolor = kwargs.get('multicolor', False)
        with_labels = kwargs.get('with_labels', True)
        node_size = kwargs.get('node_size', 20)
        font_size = kwargs.get('font_size', 8)
        font_color = kwargs.get('font_color', 'black')
        edge_color = kwargs.get('edge_color', 'black')
        edge_width = kwargs.get('edge_width', 0.1)

        if multicolor:

            node_colors = ['red' if node in self.spatial_features else 'blue' for node in self.graph.nodes]
        else:
            node_colors = 'blue'
        
        nx.draw(self.graph, pos, with_labels=with_labels, node_size=node_size, font_size=font_size, font_color=font_color, edge_color=edge_color, node_color=node_colors, width=edge_width)

        plt.show()
    
    def __str__(self):
        return f"Graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"

    def get_graph_communities(self):
        """
        An alias for the nx.algorithms.community.greedy_modularity_communities function. Works on the internal coefficient graph to find communities.

        Returns:
            list: List of sets of nodes in each community.
        """
        return list(nx.algorithms.community.greedy_modularity_communities(self.graph))
    
    def get_graph_cliques(self):
        """
        An alias for the nx.algorithms.clique.find_cliques function. Works on the internal coefficient graph to find cliques.

        Returns:
            list: List of cliques in the graph.
        """
        return list(nx.algorithms.clique.find_cliques(self.graph))
    
    def get_graph_max_clique(self):
        """
        An alias for the nx.algorithms.clique.graph_clique_number function. Works on the internal coefficient graph to find the maximum clique.

        Returns:
            int: The size of the maximum clique in the graph.
        """
        return int(nx.algorithms.clique.graph_clique_number(self.graph))
    
    def get_graph_components(self):
        """
        An alias for the nx.algorithms.components.connected_components function. Works on the internal coefficient graph to find connected components.
        
        Returns:
            list: List of sets of nodes in each connected component.
        """
        return list(nx.algorithms.components.connected_components(self.graph))
    
    def get_graph_diameter(self):
        """
        An alias for the nx.algorithms.distance_measures.diameter function. Works on the internal coefficient graph to find the diameter.

        Returns:
            int: The diameter of the graph.
        """
        return int(nx.algorithms.distance_measures.diameter(self.graph))
    
    def get_graph_degree_centrality(self):
        """
        An alias for nx.algorithms.centrality.degree_centrality function. Works on the internal coefficient graph to find the degree centrality of each node.

        Returns:
            dict: A dictionary with nodes as keys and degree centrality as values.
        """
        return dict(nx.algorithms.centrality.degree_centrality(self.graph))
    
    def get_graph_closeness_centrality(self):
        """
        An alias for nx.algorithms.centrality.closeness_centrality function. Works on the internal coefficient graph to find the closeness centrality of each node.

        Returns:
            dict: A dictionary with nodes as keys and closeness centrality as values.
        """
        return dict(nx.algorithms.centrality.closeness_centrality(self.graph))
    
    def get_graph_betweenness_centrality(self):
        """
        An alias for nx.algorithms.centrality.betweenness_centrality function. Works on the internal coefficient graph to find the betweenness centrality of each node.

        Returns:
            dict: A dictionary with nodes as keys and betweenness centrality as values.
        """
        return dict(nx.algorithms.centrality.betweenness_centrality(self.graph))
    
    def get_graph_eigenvector_centrality(self):
        """
        An alias for nx.algorithms.centrality.eigenvector_centrality function. Works on the internal coefficient graph to find the eigenvector centrality of each node.

        Returns:
            dict: A dictionary with nodes as keys and eigenvector centrality as values.
        """
        return dict(nx.algorithms.centrality.eigenvector_centrality(self.graph))
    
    def get_graph_clustering_coefficient(self):
        """
        An alias for nx.algorithms.cluster.clustering function. Works on the internal coefficient graph to find the clustering coefficient of each node.

        Returns:
            dict: A dictionary with nodes as keys and clustering coefficient as values.
        """
        
        return dict(nx.algorithms.cluster.clustering(self.graph))
    
    def get_graph_transitivity(self):
        """
        An alias for nx.algorithms.cluster.transitivity function. Works on the internal coefficient graph to find the transitivity of the graph.

        Returns:
            float: The transitivity of the graph.
        """
        return float(nx.algorithms.cluster.transitivity(self.graph))
    
    def get_graph_average_shortest_path_length(self):
        """
        An alias for nx.algorithms.shortest_paths.generic.average_shortest_path_length function. Works on the internal coefficient graph to find the average shortest path length.

        Returns:
            float: The average shortest path length of the graph.

        """
        return float(nx.algorithms.shortest_paths.generic.average_shortest_path_length(self.graph))
    
    def get_graph_dominating_set(self):
        """
        An alias for nx.algorithms.approximation.dominating_set.min_weighted_dominating_set function. Works on the internal coefficient graph to find the minimum weighted dominating set.

        Returns:
            dict: A dictionary with nodes as keys and dominating set as values.
        """
        return dict(nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(self.graph))
    
    def get_graph_max_matching(self):
        """
        An alias for nx.algorithms.matching.max_weight_matching function. Works on the internal coefficient graph to find the maximum weight matching.
        
        Returns:
            dict: A dictionary with nodes as keys and maximum weight matching as values
        """
        return dict(nx.algorithms.matching.max_weight_matching(self.graph))
    
    def get_graph_max_flow(self):
        """
        An alias for nx.algorithms.flow.maximum_flow function. Works on the internal coefficient graph to find the maximum flow.
        
        Returns:
            dict: A dictionary with nodes as keys and maximum flow as values
        """
        return dict(nx.algorithms.flow.maximum_flow(self.graph))

    def get_graph_degree(self):
        """
        An alias for graph.degree function. Works on the internal coefficient graph to find the degree of each node.
        
        Returns:
            dict: A dictionary with nodes as keys and degree as values
        """
        return dict(self.graph.degree)

class SpatialStatistics:
    """
    Module for computing spatial statistics on spatial transcriptomics data. We can get stuff like the genexgene covariance matrix and spatial autocorrelation.
    """

    def __init__(self, data:SpatialTranscriptomicsData):
        self.data = data
    
    def get_expression_position_(self, kwargs):
        cell_type = kwargs.get('cell_type', None)
        if cell_type is not None:
            expression = self.data.G[np.where(self.data.T == self.data.celltype2idx[cell_type])]
            position = self.data.P[np.where(self.data.T == self.data.celltype2idx[cell_type])]
        else:
            expression = self.data.G
            position = self.data.P
        return expression, position

    def compute_gene_covariance_matrix(self, **kwargs):
        r"""
        Computes the covariance matrix of gene expression values without considering spatial position.

        **Formula**:
        Given expression matrix `E` with `N` cells (rows) and `G` genes (columns), the centered expression `E_c` is:
        
        .. math::
            E_c = E - \text{mean}(E, \text{axis}=0)
        
        Then, the covariance matrix `\Sigma` is:
        
        .. math::
            \Sigma = \frac{1}{N-1} E_c^T E_c
        """
        #Get cell type
        expression, _ = self.get_expression_position_(kwargs)
        expression_centered = expression - np.mean(expression, axis=0)
        cov_matrix = np.cov(expression_centered, rowvar=False)
        return cov_matrix
    

    def compute_moran_I(self, **kwargs):
        r"""
        Computes Moran's I statistic to measure spatial autocorrelation in gene expression.

        **Formula**:
        Let `W` be a spatial weights matrix and `E` the centered expression. For each gene `g`, Moran's I is:
        
        .. math::
            I_g = \frac{N}{\sum_{i \neq j} W_{ij}} \frac{\sum_{i \neq j} W_{ij} E_{i,g} E_{j,g}}{\sum_i E_{i,g}^2}
        
        where `N` is the number of cells, `W` represents spatial proximity between cells, and `E_{i,g}` is the expression of gene `g` in cell `i`.
        """
        #Get threshold distance
        threshold_dist = kwargs.get('threshold_dist', 1.0)

        expression, position = self.get_expression_position_(kwargs)
        print(expression.shape)
        print(position.shape)
        
        distances = squareform(pdist(position)) #Get pariwise euclidean distances between cells

        #Now we need to deal with the spatial weights matrix
        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0) #No self connections

        expression_centered = expression - np.mean(expression, axis=0)

        WG = W @ expression_centered

        numerator = np.sum(expression_centered * WG, axis=0)
        denominator = np.sum(expression_centered ** 2, axis=0)+1e-6

        #Scale by the number of cells and the sum of weights
        n_cells = expression.shape[0]
        sum_weights = np.sum(W)

        morans_I = (n_cells / sum_weights) * (numerator / denominator)

        return morans_I

    def compute_geary_C(self, **kwargs):
        r"""
        Calculates Geary’s C statistic for spatial autocorrelation, where lower values indicate stronger positive spatial autocorrelation.

        **Formula**:
        For each gene `g`, Geary’s C is given by:
        
        .. math::
            C_g = \frac{(N - 1)}{2 \sum_{i \neq j} W_{ij}} \frac{\sum_{i \neq j} W_{ij} (E_{i,g} - E_{j,g})^2}{\sum_i (E_{i,g} - \bar{E}_g)^2}
        
        where `\bar{E}_g` is the mean expression of gene `g`.
        """
        threshold_dist = kwargs.get('threshold_dist', 1.0)

        expression, position = self.get_expression_position_(kwargs)
        N = expression.shape[0]

        # Center the expression values
        expression_centered = expression - np.mean(expression, axis=0)
        denominator = (np.sum(expression_centered**2, axis=0) * 2)+1e-6

        tree = cKDTree(position)

        numerator = np.zeros(expression.shape[1], dtype=np.float64)
        W_sum = 0

        for i in tqdm(range(N)):
            neighbors = tree.query_ball_point(position[i], threshold_dist)

            for j in neighbors:
                if i != j:
                    W_sum += 1
                    diff_squared = (expression_centered[i] - expression_centered[j]) ** 2
                    numerator += diff_squared

        gearys_C = ((N - 1) / W_sum) * (numerator / denominator)

        return gearys_C

    def compute_getis_ord_Gi(self, **kwargs):
        r"""
        Computes Getis-Ord \( G_i^* \) statistic, which identifies clusters of high or low values in gene expression data.

        **Formula**:
        For gene `g` in cell `i`:
        
        .. math::
            G_{i,g}^* = \frac{\sum_j W_{ij} E_{j,g} - \bar{E}_g \sum_j W_{ij}}{\sigma_g \sqrt{\frac{N \sum_j W_{ij}^2 - (\sum_j W_{ij})^2}{N-1}}}
        
        where `\bar{E}_g` is the mean expression, and `\sigma_g` is the standard deviation of expression for gene `g`.
        """

        threshold_dist = kwargs.get('threshold_dist', 1.0)
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))
        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0)

        N = expression.shape[0]
        W_sum = np.sum(W, axis=1)

        expression_mean = np.mean(expression, axis=0)
        expression_std = np.std(expression, axis=0, ddof=1)

        weighted_sums = W @ expression
        numerator = weighted_sums - (expression_mean * W_sum[:, np.newaxis])

        #Now we need the denominator for all cells and genes
        W_squared_sum = np.sum(W ** 2, axis=1)
        denominator = (expression_std * np.sqrt((N * W_squared_sum - W_sum ** 2) / (N - 1))[:, np.newaxis])+1e-6

        Gi_values = numerator/denominator
        
        return Gi_values
    
    def compute_ripleys_K(self, **kwargs):
        r"""
        Calculates Ripley’s K function to examine the spatial distribution of points.

        **Formula**:
        For distance `d`, Ripley’s K is:
        
        .. math::
            K(d) = \frac{A}{N^2} \sum_{i=1}^N \sum_{j \neq i} I(d_{ij} \leq d)
        
        where `A` is the area, `N` is the number of cells, `d_{ij}` is the distance between cells `i` and `j`, and `I` is an indicator function.
        """
        distances = kwargs.get('distances', np.linspace(0, 1, 100))
        area = kwargs.get('area', 1.0)

        expression, position = self.get_expression_position_(kwargs)
        N = expression.shape[0]

        tree = cKDTree(position)

        Kv = np.zeros_like(distances, dtype=np.float64)

        for idx, d in enumerate(distances):
            count_within_d = 0
            for i in range(N):
                neighbors = tree.query_ball_point(position[i], d)
                count_within_d += len(neighbors) - 1  # Exclude self
            Kv[idx] = (area / (N**2)) * count_within_d

        return Kv

    def compute_lisa(self, **kwargs):
        r"""
        Local Indicator of Spatial Association (LISA) statistic for identifying local autocorrelation.

        **Formula**:
        For each gene `g` in cell `i`:
        
        .. math::
            \text{LISA}_{i,g} = \frac{E_{i,g} - \bar{E}_g}{\sigma_g^2} \sum_j W_{ij} (E_{j,g} - \bar{E}_g)
        
        where `\bar{E}_g` and `\sigma_g^2` are the mean and variance of `g` expression.
        """
        threshold_dist = kwargs.get('threshold_dist', 1.0)
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0)

        N = expression.shape[0]
        X_mean = np.mean(expression, axis=0)
        X_var = np.var(expression, axis=0, ddof=1)+1e-6

        X_centered = expression - X_mean

        spatial_lag = W @ X_centered

        lisa = (X_centered / X_var) * spatial_lag

        return lisa
    
    def compute_disperion_index(self, **kwargs):
        r"""
        Calculates the dispersion index to indicate the level of variation in gene expression.

        **Formula**:
        For each gene `g`:
        
        .. math::
            \text{Dispersion Index}_g = \frac{\text{Var}(E_g)}{\text{Mean}(E_g)}
        
        where Var and Mean represent the variance and mean of gene expression.
        """
        expression, _ = self.get_expression_position_(kwargs)

        expression_mean = np.mean(expression, axis=0)+1e-6
        expression_var = np.var(expression, axis=0, ddof=1)

        dispersion_index = expression_var / expression_mean

        return dispersion_index
    
    def compute_spatial_cross_correlation(self, **kwargs):
        r"""
        Measures spatial correlation between gene expression pairs across spatially close cells.

        **Formula**:
        Given weights `W` and centered expression `E_c`, the cross-correlation between genes `g_1` and `g_2` is:
        
        .. math::
            \rho_{g1,g2} = \frac{(W E_c)_{g1}^T (W E_c)_{g2}}{\|E_{c,g1}\| \|E_{c,g2}\|}
        
        where `\| \cdot \|` denotes the vector norm.
        """
        """Finds spatial correlation between pairs of genes"""
        threshold_dist = kwargs.get('threshold_dist', 1.0)
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0)

        expression_centered = expression - np.mean(expression, axis=0)

        spatial_lag = W @ expression_centered
        numerator = spatial_lag.T @ spatial_lag

        norm = np.linalg.norm(expression_centered, axis=0)
        denominator = np.outer(norm, norm)+1e-6

        cross_corr_matrix = numerator/denominator

        return cross_corr_matrix
    
    def compute_spatial_co_occurence(self, **kwargs):
        r"""
        Calculates co-occurrence of gene expressions above a specified threshold.

        **Formula**:
        For genes `g_1` and `g_2`, co-occurrence is:
        
        .. math::
            \text{Co-occurrence}_{g1,g2} = \frac{\text{Count}(W \times H_{g1} \times H_{g2})}{\text{Count}(H_{g1}) \times \text{Count}(H_{g2})}
        
        where `H_g` is an indicator for high expression.
        """
        threshold_distance = kwargs.get('threshold_distance', 1.0)
        expression_threshold = kwargs.get('expression_threshold', 0.5)

        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        W = (distances < threshold_distance).astype(float)
        np.fill_diagonal(W, 0)

        high_expression = expression > np.percentile(expression, expression_threshold*100, axis=0)

        co_occurence_counts_matrix = (high_expression.T @ W @ high_expression)

        num_high_expressions = np.sum(high_expression, axis=0)
        denominator = np.outer(num_high_expressions, num_high_expressions)

        denominator[denominator == 0] = 1

        co_occurence_matrix = co_occurence_counts_matrix / denominator

        return co_occurence_matrix
    
    def compute_mark_correlation_function(self, **kwargs):
        r"""
        Examines spatial correlation of gene expression marks over varying distances.

        **Formula**:
        For distance `d`, the mark correlation for gene `g` is:
        
        .. math::
            M_g(d) = \frac{\sum_{i \neq j} I(d_{ij} \leq d) E_{i,g} E_{j,g}}{\sum_{i \neq j} I(d_{ij} \leq d)}
        
        where `I(d_{ij} \leq d)` indicates cells within distance `d`.
        """
        distances_to_evaluate = kwargs.get('distances', np.linspace(0, 1, 100))
        expression, position = self.get_expression_position_(kwargs)
        N = position.shape[0]
        mark_corr_values = np.zeros((len(distances_to_evaluate), expression.shape[1]), dtype=np.float64)

        tree = cKDTree(position)

        for idx, d in enumerate(distances_to_evaluate):
            weighted_mark_sum = np.zeros(expression.shape[1], dtype=np.float64)
            valid_pairs = 0

            for i in tqdm(range(N)):
                neighbors = tree.query_ball_point(position[i], d)
                for j in neighbors:
                    if i != j:
                        valid_pairs += 1
                        weighted_mark_sum += expression[i] * expression[j]

            if valid_pairs > 0:
                mark_corr_values[idx] = weighted_mark_sum / valid_pairs
            else:
                mark_corr_values[idx] = 0

        return mark_corr_values
    
    def bivariate_morans_I(self, **kwargs):
        r"""
        Computes bivariate Moran's I, examining spatial correlation between pairs of genes.

        **Formula**:
        For genes `g_1` and `g_2`:
        
        .. math::
            I_{g1,g2} = \frac{N}{\sum_{i \neq j} W_{ij}} \frac{\sum_{i \neq j} W_{ij} E_{i,g1} E_{j,g2}}{\|E_{c,g1}\| \|E_{c,g2}\|}
        
        """
        threshold_distance = kwargs.get('threshold_distance', 1.0)
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        W = (distances < threshold_distance).astype(float)
        np.fill_diagonal(W, 0)

        W_sum = np.sum(W)+1e-6

        expression_centered = expression - np.mean(expression, axis=0)

        spatial_lag = W @ expression_centered

        numerator = expression_centered.T @ spatial_lag

        norm_X = np.sqrt(np.sum(expression_centered ** 2, axis=0))
        norm_Y = np.sqrt(np.sum(spatial_lag ** 2, axis=0))

        denominator = np.outer(norm_X, norm_Y)+1e-6
        bivariate_morans_I = (expression.shape[0] / W_sum) * (numerator / denominator)

        return bivariate_morans_I
    
    def spatial_eigenvector_mapping(self, **kwargs):
        r"""
        Calculates eigenvectors of the spatial Laplacian matrix for mapping spatial autocorrelation patterns.

        **Formula**:
        Construct the Laplacian `L = D - W`, where `D` is the degree matrix of `W`. Solve:
        
        .. math::
            L \mathbf{v} = \lambda \mathbf{v}
        
        where `\mathbf{v}` and `\lambda` are eigenvectors and eigenvalues, sorted by smallest `\lambda`.
        """
        threshold_distance = kwargs.get('threshold_distance', 1.0)
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        W = (distances < threshold_distance).astype(float)
        np.fill_diagonal(W, 0)

        D = np.diag(W.sum(axis=1))
        L = D - W #Laplacian matrix

        #Use eigh here because L is symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvectors, eigenvalues

    def get_mean_expression(self, **kwargs):
        expression, _ = self.get_expression_position_(kwargs)
        return np.mean(expression, axis=0)
    
    def get_variance_expression(self, **kwargs):
        expression, _ = self.get_expression_position_(kwargs)
        return np.var(expression, axis=0, ddof=1)
    
    def full_report(self, **kwargs):
        #Compute all statistics
        gene_covariance = self.compute_gene_covariance_matrix(**kwargs)
        print('SPATIAL STATISTICS COMPUTED GENE COVARIANCE')
        morans_I = self.compute_moran_I(**kwargs)
        print('SPATIAL STATISTICS COMPUTED MORANS I')
        gearys_C = self.compute_geary_C(**kwargs)
        print('SPATIAL STATISTICS COMPUTED GEARYS C')
        getis_ord_Gi = self.compute_getis_ord_Gi(**kwargs)
        print('SPATIAL STATISTICS COMPUTED GETIS ORD GI')
        ripley_K = self.compute_ripleys_K(**kwargs)
        print('SPATIAL STATISTICS COMPUTED RIPLEYS K')
        lisa = self.compute_lisa(**kwargs)
        print('SPATIAL STATISTICS COMPUTED LISA')
        dispersion_index = self.compute_disperion_index(**kwargs)
        print('SPATIAL STATISTICS COMPUTED DISPERSION INDEX')
        spatial_cross_correlation = self.compute_spatial_cross_correlation(**kwargs)
        print('SPATIAL STATISTICS COMPUTED SPATIAL CROSS CORRELATION')
        spatial_co_occurence = self.compute_spatial_co_occurence(**kwargs)
        print('SPATIAL STATISTICS COMPUTED SPATIAL CO OCCURENCE')
        mark_correlation_function = self.compute_mark_correlation_function(**kwargs)
        print('SPATIAL STATISTICS COMPUTED MARK CORRELATION FUNCTION')
        bivariate_morans_I = self.bivariate_morans_I(**kwargs)
        print('SPATIAL STATISTICS COMPUTED BIVARIATE MORANS I')
        spatial_eigenvectors, spatial_eigenvalues = self.spatial_eigenvector_mapping(**kwargs)
        print('SPATIAL STATISTICS COMPUTED SPATIAL EIGENVECTORS + EIGENVALUES')
        mean_expression= self.get_mean_expression(**kwargs)
        print('SPATIAL STATISTICS COMPUTED MEAN EXPRESSION')
        variance_expression = self.get_variance_expression(**kwargs)
        print('SPATIAL STATISTICS COMPUTED VARIANCE EXPRESSION')

        d = {'gene_covariance': gene_covariance,
             'morans_I': morans_I,
             'gearys_C': gearys_C,
             'getis_ord_Gi': getis_ord_Gi,
             'ripley_K': ripley_K,
             'lisa': lisa,
             'dispersion_index': dispersion_index,
             'spatial_cross_correlation': spatial_cross_correlation,
             'spatial_co_occurence': spatial_co_occurence,
             'mark_correlation_function': mark_correlation_function,
             'bivariate_morans_I': bivariate_morans_I,
             'spatial_eigenvectors': spatial_eigenvectors,
             'spatial_eigenvalues': spatial_eigenvalues,
             'mean_expression': mean_expression,
             'variance_expression': variance_expression,
             'feature_names': self.data.gene_names,
             'kwargs': kwargs}
        return d

    def report_by_type(self, out_directory, **kwargs):
        """
        Analagous to full report but will compute for every cell type, useful for basic exploratory analysis
        Requires an out directory as full reports won't likely fit in RAM
        """
        cell_types = list(self.data.celltype2idx.keys())

        for cell_type in cell_types:
            kwargs['cell_type'] = cell_type
            report = self.full_report(**kwargs)
            np.savez(os.path.join(out_directory, f'{cell_type}_report.npz'), **report)

#I may need to redo a lot of the changes I made earlier today, something in this file got messed up when I merged the changes from both computers

def filter_by_gene(data, **kwargs):
    """
    Filters the data by genes
    """
    genes = kwargs.get('genes', [])
    gene_indices = [data.gene2idx[gene] for gene in genes]
    

    return gene_indices

def get_gaussian_kde_values(x, y, expression, bandwidth=0.3):
    """
    Smooths the expression values using a Gaussian KDE. Better than a scatter plot for visually analyzing spatial patterns.
    Plot with a simple plt.imshow to analyze.

    Parameters:
        x (np.array): X coordinates.
        y (np.array): Y coordinates.
        expression (np.array): Expression values.
        bandwidth (float): Bandwidth of the KDE.
    
    Returns:
        np.array: KDE values.
    """
    data = pd.DataFrame({'x': x, 'y': y, 'expression': expression})
    x_grid, y_grid = np.mgrid[0:5:100j, 0:5:100j]
    grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Use KDE with weighted expression levels to create a smooth gradient
    values = np.vstack([data['x'], data['y']])
    expression = data['expression']
    kde = gaussian_kde(values, weights=expression, bw_method=bandwidth)  # Adjust bandwidth for smoothness
    kde_values = kde(grid_positions).reshape(x_grid.shape)

    return kde_values

if __name__ == "__main__":
    feature_sets = FeatureSetData(path='cancer_annotations.csv', bin_key='+')
    
    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())

    for cell_type in annotations['cell_types']:
        #cell_type = 'T CD8 memory'
        print(cell_type)
        st = SpatialTranscriptomicsData(G, P, T, gene_names=annotations['gene_names'], cell_types=annotations['cell_types'])
        
        stats = np.load(f'sample_statistics\\{cell_type}_report.npz')
        morans_i = stats['morans_I']
        indices = np.argsort(morans_i)[::-1][:2500]
        genes = [stats['feature_names'][i] for i in indices]


        cell_type_abundance_feature = NeighborhoodAbundanceFeature(st, cell_type=cell_type, radius=0.1, alpha=1)
        metagene_feature = NeighborhoodMetageneFeature(st, feature_sets, cell_type=cell_type, radius=0.1, alpha=1)

        #ts = TranscriptSpace(st, [cell_type_abundance_feature, metagene_feature], [1.0,1.0], cell_type='epithelial.cancer.subtype_2', lambd=1e-2*5)
        #ts = TranscriptSpace(st, [cell_type_abundance_feature], [1.0], cell_type='epithelial.cancer.subtype_1', lambd=1e-2*5)
        #ts = TranscriptSpace(st, [], [], cell_type=cell_type, lambd=1e-2)
        #coeffs = ts.fit(radius=0.1, include_expression=True, filter=filter_by_gene, genes=genes, n_resamples=6, resample_dim=3000)
        #np.savez(f'coefficients\\no_features\\{cell_type}.npz', **coeffs)

        #ts = TranscriptSpace(st, [cell_type_abundance_feature, metagene_feature], [1.0, 1.0], cell_type=cell_type, lambd=1e-2)
<<<<<<< HEAD:transcript_space/transcript_space.py
        #ts = TranscriptSpace(st, [cell_type_abundance_feature, metagene_feature], cell_type=cell_type, lambd=1e-2)
        ts = TranscriptSpace(st, [], cell_type=cell_type, lambd=1e-2)
=======
        ts = STLasso(st, [cell_type_abundance_feature], cell_type=cell_type, lambd=1e-1*2)
>>>>>>> 500d5a31144b673f43b84d33804cee46b68db264:st_lasso/st_lasso.py
        coeffs = ts.fit(radius=0.1, include_expression=True, filter=filter_by_gene, genes=genes, n_resamples=1, n_retries=1000, resample_dim=3000, k_best=50)
        #coeffs = ts.fit(radius=0.1, include_expression=True, n_resamples=5, n_retries=1000, resample_dim=3000)
        np.savez(f'coefficients\\without\\{cell_type}.npz', **coeffs)

        print()
        print()

#Vignettes to use later
"""

    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())
    print(annotations)
    st = SpatialTranscriptomicsData(G, P, T,annotations=annotations)

    cancer_gene_sets = json.loads(open('c4.json').read())
    gene_set_names = list(cancer_gene_sets.keys())

    gene_sets = []
    for gene_set in gene_set_names:
        gene_sets.append((gene_set, list(cancer_gene_sets[gene_set]['geneSymbols'])))
    
    statistics = SpatialStastics(st)
    statistics.report_by_type('sample_statistics', threshold_dist=0.1, distances=np.linspace(0, 0.5, 2))"""
"""
st = SpatialTranscriptomicsData("data\\colon_cancer", "colon_cancer")
cancer_gene_sets = json.loads(open('c4.json').read())
gene_set_names = list(cancer_gene_sets.keys())

gene_sets = []
for gene_set in gene_set_names:
    gene_sets.append((gene_set, list(cancer_gene_sets[gene_set]['geneSymbols'])))
print(gene_sets)
st.remap_metagenes(gene_sets)
print("DONE")
"""

"""
    st = SpatialTranscriptomicsData("data\\colon_cancer", "colon_cancer")
    cancer_gene_sets = json.loads(open('c4.json').read())
    gene_set_names = list(cancer_gene_sets.keys())

    gene_sets = []
    for gene_set in gene_set_names:
        gene_sets.append((gene_set, list(cancer_gene_sets[gene_set]['geneSymbols'])))
    print(gene_sets)
    st.remap_metagenes(gene_sets)
    print("DONE")
    statistics = SpatialStastics(st)

    report = statistics.full_report(cell_type='Treg', distance_threshold=0.1, distances=np.linspace(0, 0.5, 2))
    np.savez('statistics.npz', **report)
"""
"""
    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())

    for cell_type in annotations['cell_types']:
        print(f"Training Model For Cell Type {cell_type}")
        st = SpatialTranscriptomicsData(G, P, T, annotations=annotations)
        stats = np.load(f'sample_statistics\\{cell_type}_report.npz')
        morans_i = stats['morans_I']
        indices = np.argsort(morans_i)[::-1][:250]

        genes = [stats['feature_names'][i] for i in indices]
        st.filter_genes(filter_by_gene, genes=genes)
        neighborhood_abundances = NeighborhoodAbundanceFeature(st)

        ts = TranscriptSpace(st, [neighborhood_abundances], [1.0], cell_type=cell_type, lambd=1e-2*5)
        coeffs = ts.fit(radius=0.1)
        np.savez(f'coefficients\\{cell_type}.npz', **coeffs)
        print()
        print()
"""