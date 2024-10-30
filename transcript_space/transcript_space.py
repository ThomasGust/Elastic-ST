import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import json as json
import os
import pandas as pd
from tqdm import tqdm
from typing import Union
import networkx as nx
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import multivariate_normal
from sklearn.linear_model import Lasso
#Get KDTree from sklearn
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import json

class SpatialTranscriptomicsData:
    """
    This is the basic data object to hold spatial transriptomics data.

    For TranscriptSpace, all data has a few key attributes:
    - G: a numpy array of shape (n_cells, n_genes) containing the gene expression data
    - P: a numpy array of shape (n_cells, 2) containing the spatiual coordinates of each cell
    - T: a sparse numpy matrix of shape (n_cells, n_cell_types) containing the cell type information
    - cell_types: a list of cell type names
    - gene_names: a list of gene names
    """

    def __init__(self, G:np.array, P:np.array, T:np.array, annotations:dict):
        self.G = G
        self.P = P
        self.T = T


        #Load annotations for cell types and gene names, needed for interpretability
        self.annotations = annotations
        self.cell_types = self.annotations['cell_types']
        self.gene_names = self.annotations['gene_names']

        self.map_dicts()
    
    def threshold_G(self, by:callable):
        """
        Apply a thresholding function to the expression matrix G.
        """
        self.G = by(self.G)
    
    def map_dicts(self):
        self.celltype2idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}
        self.idx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.cell_types)}

        self.gene2idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.idx2gene = {idx: gene for idx, gene in enumerate(self.gene_names)}
    
    @staticmethod
    def covariance_threshold(G, **kwargs):
        threshold = kwargs.get('threshold', 0.5)

        #Get only the genes with the highest average covariance (If this gene changes, other genes are likely to change as well)
        cov_matrix = np.cov(G, rowvar=False)
        avg_cov = np.mean(cov_matrix, axis=0)
        percentile = np.percentile(avg_cov, threshold * 100)
        gene_indices = np.where(avg_cov > percentile)[0]

        return gene_indices

    def filter_genes(self, filter:callable, **kwargs):
        gene_indices = filter(self.G, **kwargs)
        self.G = self.G[:, gene_indices]
        self.gene_names = [self.gene_names[idx] for idx in gene_indices]

        self.map_dicts()
    
    def remap_metagenes(self, metagenes:Union[list[tuple[str, list[str]]], list[tuple[str, list[str], float]]]):
        """
        Instead of having cellxgenes, the expression matrix G will now have cellxmetagenes. This is useful for reducing the dimensionality of the data and revealing better biological insights.
        metagenes can either be a list of tuples of metagene names, and a list of gene names, or a list of tuples of metagene names, a list of gene names, and a list of weights.
        """

        new_G = np.zeros((self.G.shape[0], len(metagenes)))
        metagene_names = []

        normalized_G = self.G / np.sum(self.G, axis=1)[:, np.newaxis]
        if len(list(metagenes[0])) == 2:
            #No weights
            for metagene in tqdm(metagenes):
                metagene_name, gene_names = metagene
                #gene_indices = [self.gene2idx[gene] for gene in gene_names]
                gene_indices = []
                for gene in gene_names:
                    if gene in self.gene2idx:
                        gene_indices.append(self.gene2idx[gene])
                new_G[:, metagenes.index(metagene)] = np.sum(normalized_G[:, gene_indices], axis=1)
                metagene_names.append(metagene_name)

        elif len(list(metagenes[0])) == 3:
            #Weights
            for metagene in metagenes:
                metagene_name, gene_names, weights = metagene
                #gene_indices = [self.gene2idx[gene] for gene in gene_names]
                gene_indices = []
                for gene in gene_names:
                    if gene in self.gene2idx:
                        gene_indices.append(self.gene2idx[gene])
                new_G[:, metagenes.index(metagene)] = np.sum(normalized_G[:, gene_indices] * weights, axis=1)
                metagene_names.append(metagene_name)
        
        self.G = new_G
        self.gene_names = metagene_names
        self.map_dicts()

class FeatureSetData:
    """
    This represents a functional annotation object.
    It holds a sparse matrix showing which genes are in which feature set.
    Essentially used to capture metagenes.
    """

    def __init__(self, path:str, bin_key=1):
        self.path = path
        self.bin_key = bin_key

        self.annotations = pd.read_csv(path, index_col=0)
        self.feature_sets = list(self.annotations.columns)
        self.gene_names = list(self.annotations.index)
        print(self.gene_names)

    def get_feature_sets_for_gene(self, gene:str):
        #Get index of gene row
        gene_idx = self.gene_names.index('A1BG')
        print(gene_idx)
class ModelFeature:

    def __init__(self, name):
        self.name = name

    def compute_feature(self, **kwargs):
        raise NotImplementedError

    def get_feature(self, **kwargs):
        raise NotImplementedError

class GeneExpressionFeature(ModelFeature):

    def __init__(self, data:SpatialTranscriptomicsData, t:Union[str, int, None]):
        super().__init__("gene_expression")

        self.data = data

        if t is not None:
            if isinstance(t, str):
                self.t = self.data.celltype2idx[t]
            else:
                self.t = t
        
        i = np.where(self.data.T == self.t)
        self.data.G = self.data.G[i]
    
    def compute_feature(self, **kwargs):
        self.G = self.data.G
        self.gene2idx = self.data.gene2idx
        #Get alpha or else default to 1.0
        self.alpha = kwargs.get('alpha', 1.0)

    def get_feature(self, **kwargs):
        #We may want to only use a subset of genes
        genes = kwargs.get('genes', self.data.gene_names)
        #Or exclude a list
        exclude_genes = kwargs.get('exclude_genes', [])

        genes = [gene for gene in genes if gene not in exclude_genes]
        gene_indices = [self.gene2idx[gene] for gene in genes]

        #Return the gene expression data and the gene names alongside the alpha vector
        return self.G[:, gene_indices], {idx: gene for idx, gene in enumerate(genes)}, [self.alpha] * len(genes)

class NeighborhoodAbundanceFeature(ModelFeature):
    
        def __init__(self, data:SpatialTranscriptomicsData):
            super().__init__("neighborhood_abundance")
    
            self.data = data
        
        def compute_feature(self, **kwargs):
            self.G = self.data.G
            self.P = self.data.P
            self.T = self.data.T
    
            self.celltype2idx = self.data.celltype2idx
            self.idx2celltype = self.data.idx2celltype
    
            #Get alpha or else default to 1.0
            alpha = kwargs.get('alpha', 1.0)
            self.alpha = alpha
    
            #Get the neighborhood radius
            radius = kwargs.get('radius', 1.0)
            self.radius = radius
    
            #Get the neighborhood abundances
            if not os.path.exists('neighborhood_abundances.npy'):
                self.neighborhood_abundances = self._compute_neighborhood_abundances(self.G, self.P, self.T, radius)
            else:
                self.neighborhood_abundances = np.load('neighborhood_abundances.npy')

            #Save the neighborhood abundances to a npy file
            np.save('neighborhood_abundances.npy', self.neighborhood_abundances)

            self.featureidx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.idx2celltype)}
        
        def get_feature(self, **kwargs):
            return self.neighborhood_abundances, self.featureidx2celltype, [self.alpha] * self.neighborhood_abundances.shape[1]

        def _compute_neighborhood_abundances(self, G, P, T, radius):
            n_cells = G.shape[0]
            #Reshape T to be a 2d array from a 1d array of strings
            T_ = np.zeros((n_cells, len(self.celltype2idx)))
            for i in range(n_cells):
                #print(T[i])
                T_[i, T[i]] = 1
            
            T = T_
            n_cell_types = T.shape[1]

            neighborhood_abundances = np.zeros((n_cells, n_cell_types))

            #Use a KDTree to find the nearest neighbors
            tree = KDTree(P)
            for i in tqdm(range(n_cells)):
                
                #Get the neighbors within the radius
                neighbors = tree.query_radius(P[i].reshape(1, -1), r=radius)[0]
                for neighbor in neighbors:
                    neighborhood_abundances[i] += T[neighbor][0]
            return neighborhood_abundances

class NeighborhoodMetageneFeature(ModelFeature):
    """Compute the neighborhood abundance of all the genes in a given feature set"""
    def __init__(self, data:SpatialTranscriptomicsData, feature_set:FeatureSetData):
        super().__init__("neighborhood_metagene")
        """
        Parameters:
            data (SpatialTranscriptomicsData): Spatial transcriptomics data object.
            feature_set (FeatureSetData): Feature set data object.
        """

        self.data = data
        self.feature_set = feature_set

    def compute_feature(self, **kwargs):
        """
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

        #Get alpha or else default to 1.0
        alpha = kwargs.get('alpha', 1.0)
        self.alpha = alpha

        #Get the neighborhood radius
        radius = kwargs.get('radius', 1.0)
        self.radius = radius

        #Get the neighborhood abundances
        self.neighborhood_metagenes = self._compute_neighborhood_metagenes(self.G, self.P, self.T, self.feature_set, radius)

        self.featureidx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.idx2celltype)}
    
    def _compute_neighborhood_metagenes(self, G, P, T, feature_set, radius):
        """
        Parameters:
            G (numpy.ndarray): Cells-by-genes expression matrix.
            P (numpy.ndarray): Cells-by-2 position matrix.
            T (numpy.ndarray): Cells-by-cell types matrix.
            feature_set (FeatureSetData): Feature set data object.
            radius (float): Neighborhood radius.
        """
        n_cells = G.shape[0]

        neighborhood_metagenes = np.zeros((n_cells, len(feature_set.feature_sets)))

        #For each metagene in each cell, count the number of times any gene in the metagene is expressed in the neighborhood
        for i in range(n_cells):
            neighborhood_metagenes[i] = self._compute_neighborhood_metagene(G, P, i, feature_set, radius)
    
        return neighborhood_metagenes
    
    def _compute_neighborhood_metagene(self, G, P, i, feature_set, radius):
        """
        Parameters:
            G (numpy.ndarray): Cells-by-genes expression matrix.
            P (numpy.ndarray): Cells-by-2 position matrix.
            i (int): Cell index.
            feature_set (FeatureSetData): Feature set data object.
            radius (float): Neighborhood radius.
        """
        n_cells = G.shape[0]
        neighborhood_metagene = np.zeros(len(feature_set.feature_sets))

        for j in range(n_cells):
            if i == j:
                continue

            distance = np.linalg.norm(P[i] - P[j])

            if distance <= radius:
                for gene in feature_set.gene_names:
                    genes_in_feature_set = feature_set.get_genes_in_feature_set(gene)
                    if gene in genes_in_feature_set:
                        gene_idx = feature_set.gene2idx[gene]
                        neighborhood_metagene += G[j, gene_idx]
        
        return neighborhood_metagene

    def get_feature(self, **kwargs):
        """
        Parameters:
            alpha (float): Regularization parameter.
            radius (float): Neighborhood radius.
        """
        return self.neighborhood_metagenes, self.featureidx2celltype, [self.alpha] * self.neighborhood_metagenes.shape[1]


def flatten_list(l):
    return [item for sublist in l for item in sublist]
class TranscriptSpace:

    def __init__(self, in_features:list[ModelFeature], out_feature:GeneExpressionFeature, alphas=list[float], cell_type='epithelial.cancer.subtype_1'):
        """
        Parameters:
            in_features (list): List of input features.
            out_feature (ModelFeature): Output feature.
            alphas (list): List of regularization parameters for each input feature.
        """
        self.in_features = in_features
        
        self.out_feature = out_feature
        self.out_feature.compute_feature()

        self.cell_type = cell_type

        #Compute feature for every feature
        for feature in self.in_features:
            feature.compute_feature()
    
        self.alphas = alphas
        
    def fit(self, **kwargs):
        in_feature_dim = sum([len(list(feature.get_feature()[1].values())) for feature in self.in_features])
        out_feature_dim = self.out_feature.data.G.shape[1]

        in_feature_names = flatten_list([list(feature.get_feature()[1].values()) for feature in self.in_features])
        out_feature_names = self.out_feature.data.gene_names

        self.coefficients = np.zeros((in_feature_dim, out_feature_dim))
        print(self.coefficients.shape)

        #Get l1 ratio, n_resamples, and stability threshold from kwargs
        l1_ratio = kwargs.get('l1_ratio', 0.5)
        n_resamples = kwargs.get('n_resamples', 4)
        stability_threshold = kwargs.get('stability_threshold', 0.5)

        out_path = kwargs.get('out_path', 'coefficients')

        for i in tqdm(range(out_feature_dim)):
            feature_attributes = []
            for feature in self.in_features:
                epoch_feature_matrix, epoch_idx2feature, epoch_feature_alpha = feature.get_feature(exclude_genes=[self.out_feature.data.idx2gene[i]], alpha=self.alphas[self.in_features.index(feature)])
                feature_dict = {'matrix': epoch_feature_matrix, 'idx2feature': epoch_idx2feature, 'alpha': epoch_feature_alpha}
                feature_attributes.append(feature_dict)
            
            y = self.out_feature.G[:, i]
            X = np.concatenate([feature['matrix'] * np.sqrt(1/np.array((feature['alpha']))) for feature in feature_attributes], axis=1)


            c = []
            for r in range(n_resamples):
                X, y = resample(X, y)
                model = ElasticNet(alpha=1e-3, l1_ratio=l1_ratio)
                model.fit(X, y)
                
                sample_coeffs = model.coef_
            
            #For any coefficents that existed a percent of times greater than the stability threshold, add the mean of their non-zero values, else add 0
            times_existed = np.sum(np.array(c) != 0, axis=0)
            mean_coeffs = np.mean(np.array(c), axis=0)

            print(mean_coeffs.shape)
            coeffs = np.where(times_existed > stability_threshold, mean_coeffs, 0)
            print(coeffs.shape)
            print(self.coefficients.shape)
                
            self.coefficients[:, i] = coeffs


        #self.coefficient_matrix = CoefficientFeatureMatrix(self.coefficients, in_feature_names, out_feature_names)
        #self.coefficient_matrix.save(out_path)
    

class CoefficientAnalysis:

    def __init__(self, coefficient_matrix, graph_threshold):
        self.coefficient_matrix = coefficient_matrix

        self.zero_diagonal()
        self.graph = self.build_coefficient_graph(graph_threshold)


    def zero_diagonal(self):
        m = self.coefficient_matrix.coefficients.shape[1]
        # Create a zero matrix of shape (m, m)
        new_matrix = np.zeros((m, m))

        # Fill the new matrix with the original data
        new_matrix[~np.eye(m, dtype=bool)] = self.coefficient_matrix.coefficients.ravel()
        self.coefficient_matrix.coefficients = new_matrix
    def build_coefficient_graph(self, threshold:float):
        G = nx.Graph()

        for i, in_feature in enumerate(self.coefficient_matrix.in_features):
            for j, out_feature in enumerate(self.coefficient_matrix.out_features):
                weight = self.coefficient_matrix.coefficients[i, j]
                if i != j and weight != 0 and abs(weight) > threshold:
                    G.add_edge(in_feature, out_feature, weight=weight)
        
        return G

    def plot_coefficient_graph(self, **kwargs):
        #Get the layout
        layout = kwargs.get('layout', 'spring')
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)

        plt.show()
    
    def __str__(self):
        return f"Graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"

    def get_graph_communities(self):
        return nx.algorithms.community.greedy_modularity_communities(self.graph)
    
    def get_graph_cliques(self):
        return nx.algorithms.clique.find_cliques(self.graph)
    
    def get_graph_max_clique(self):
        return nx.algorithms.clique.graph_clique_number(self.graph)
    
    def get_graph_components(self):
        return nx.algorithms.components.connected_components(self.graph)
    
    def get_graph_diameter(self):
        return nx.algorithms.distance_measures.diameter(self.graph)
    
    def get_graph_degree_centrality(self):
        return nx.algorithms.centrality.degree_centrality(self.graph)
    
    def get_graph_closeness_centrality(self):
        return nx.algorithms.centrality.closeness_centrality(self.graph)
    
    def get_graph_betweenness_centrality(self):
        return nx.algorithms.centrality.betweenness_centrality(self.graph)
    
    def get_graph_eigenvector_centrality(self):
        return nx.algorithms.centrality.eigenvector_centrality(self.graph)
    
    def get_graph_clustering_coefficient(self):
        return nx.algorithms.cluster.clustering(self.graph)
    
    def get_graph_transitivity(self):
        return nx.algorithms.cluster.transitivity(self.graph)
    
    def get_graph_average_shortest_path_length(self):
        return nx.algorithms.shortest_paths.generic.average_shortest_path_length(self.graph)
    
    def get_graph_dominating_set(self):
        return nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(self.graph)
    
    def get_graph_max_matching(self):
        return nx.algorithms.matching.max_weight_matching(self.graph)
    
    def get_graph_max_flow(self):
        return nx.algorithms.flow.maximum_flow(self.graph)

class SpatialStastics:
    #TODO:
    #Spatial eigenvector mapping
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
        """Does not take into account any spatial information, relies purely on expression data"""
        #Get cell type
        expression, _ = self.get_expression_position_(kwargs)
        expression_centered = expression - np.mean(expression, axis=0)
        cov_matrix = np.cov(expression_centered, rowvar=False)
        return cov_matrix
    

    def compute_moran_I(self, **kwargs):

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
        """Bad implementation, currently is not vectorized for memory purposes"""
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
        """Also does not take into account spatial information"""

        expression, _ = self.get_expression_position_(kwargs)

        expression_mean = np.mean(expression, axis=0)+1e-6
        expression_var = np.var(expression, axis=0, ddof=1)

        dispersion_index = expression_var / expression_mean

        return dispersion_index
    
    def compute_spatial_cross_correlation(self, **kwargs):
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

if __name__ == "__main__":
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
    statistics.report_by_type('sample_statistics', threshold_dist=0.1, distances=np.linspace(0, 0.5, 2))


#Vignettes to use later
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