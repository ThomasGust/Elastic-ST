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
from scipy.spatial.distance import pdist, squareform

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

    def __init__(self, root_path:str, name:str):
        self.root_path = root_path
        self.name = name

        self.G_path = os.path.join(root_path, f"{name}_G.npy")
        self.P_path = os.path.join(root_path, f"{name}_P.npy")
        self.T_path = os.path.join(root_path, f"{name}_T.npy")
        self.annotation_path = os.path.join(root_path, f"{name}_annotation.json")

        #Load all of the core data primtives representing the data
        self.G = np.load(self.G_path)
        self.P = np.load(self.P_path)
        self.T = np.load(self.T_path)


        #Load annotations for cell types and gene names, needed for interpretability
        self.annotations = json.load(open(self.annotation_path))
        self.cell_types = self.annotations['cell_types']
        self.gene_names = self.annotations['gene_names']

        self.celltype2idx = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}
        self.idx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.cell_types)}

        self.gene2idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.idx2gene = {idx: gene for idx, gene in enumerate(self.gene_names)}

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
        self.feature_sets = self.annotations.columns
        self.gene_names = self.annotations.index
    
    def get_genes_in_feature_set(self, feature_set:str):
        return self.annotations.index[self.annotations[feature_set] == self.bin_key].tolist()
    
    def get_feature_sets_for_gene(self, gene:str):
        return self.annotations.columns[self.annotations.loc[gene] == self.bin_key].tolist()

class ModelFeature:

    def __init__(self, name):
        self.name = name

    def compute_feature(self, **kwargs):
        raise NotImplementedError

    def get_feature(self, **kwargs):
        raise NotImplementedError

class GeneExpressionFeature(ModelFeature):

    def __init__(self, data:SpatialTranscriptomicsData):
        super().__init__("gene_expression")

        self.data = data
    
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
            self.neighborhood_abundances = self._compute_neighborhood_abundances(self.G, self.P, self.T, radius)

            self.featureidx2celltype = {idx: cell_type for idx, cell_type in enumerate(self.idx2celltype)}
        
        def get_feature(self, **kwargs):
            return self.neighborhood_abundances, self.featureidx2celltype, [self.alpha] * self.neighborhood_abundances.shape[1]

        def _compute_neighborhood_abundances(self, G, P, T, radius):
            n_cells = G.shape[0]
            n_cell_types = T.shape[1]
    
            neighborhood_abundances = np.zeros((n_cells, n_cell_types))
    
            for i in range(n_cells):
                cell_type = T[i].nonzero()[1][0]
                neighborhood_abundances[i, cell_type] += 1
    
                for j in range(n_cells):
                    if i == j:
                        continue
    
                    distance = np.linalg.norm(P[i] - P[j])
    
                    if distance <= radius:
                        cell_type = T[j].nonzero()[1][0]
                        neighborhood_abundances[i, cell_type] += 1
    
            return neighborhood_abundances

class NeighborhoodMetageneFeature(ModelFeature):
    """Compute the neighborhood abundance of all the genes in a given feature set"""
    def __init__(self, data:SpatialTranscriptomicsData, feature_set:FeatureSetData):
        super().__init__("neighborhood_metagene")

        self.data = data
        self.feature_set = feature_set

    def compute_feature(self, **kwargs):
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
        n_cells = G.shape[0]

        neighborhood_metagenes = np.zeros((n_cells, len(feature_set.feature_sets)))

        #For each metagene in each cell, count the number of times any gene in the metagene is expressed in the neighborhood
        for i in range(n_cells):
            neighborhood_metagenes[i] = self._compute_neighborhood_metagene(G, P, i, feature_set, radius)
    
        return neighborhood_metagenes
    
    def _compute_neighborhood_metagene(self, G, P, i, feature_set, radius):
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
        return self.neighborhood_metagenes, self.featureidx2celltype, [self.alpha] * self.neighborhood_metagenes.shape[1]
            

class MASSENLasso:
    """
    This module implements a MASSEN (multi alpha stability selection elastic net) lasso model.
    """

    def __init__(self, l:int, alphas:np.array, l1_ratio:float, n_resamples:50, stability_threshold:float):
        self.l = l
        self.alphas = alphas * l
        self.l1_ratio = l1_ratio
        self.n_resamples = n_resamples
        self.stability_threshold = stability_threshold

        self.scaler = StandardScaler()
        self.selected_features_ = None
        self.coef_ = None

    
    def _objective(self, coef, X, y):
        residual = y - X @ coef
        rss = np.sum(residual ** 2)

        l1_penalty = np.sum(self.alphas * np.abs(coef))
        l2_penalty = np.sum((1-self.alphas) * coef ** 2)

        reg = self.l1_ratio * l1_penalty (1 - self.l1_ratio) * l2_penalty

        return 0.5 * rss + reg
    
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        
        selection_counts = np.zeros(X.shape[1]) #Helper vector to perform stability selection
 
        for _ in range(self.n_resamples):
            X_resampled, y_resampled = resample(X, y)

            initial_coef = np.zeros(X_resampled.shape[1])

            result = minimize(self._objective, initial_coef, args=(X_resampled, y_resampled), method='L-BFGS-B')
            coef_resampled = result.x
            selection_counts += (np.abs(coef_resampled) > 1e-5).astype(int)
        
        self.selected_features_ = np.where(selection_counts / self.n_resamples >= self.stability_threshold)[0]

        if len(self.selected_features_) >= 0:
            X_selected = X[:, self.selected_features_]
            initial_coef = np.zeros(X_selected.shape[1])
            result = minimize(self._objective, initial_coef, args=(X_selected, y), method='L-BFGS-B')
            self.coef_ = np.zeros(X.shape[1])
            self.coef_[self.selected_features_] = result.x
        
        else:
            self.coef_ = np.zeros(X.shape[1])
    
    def predict(self, X):
        X = self.scalar.transform(X)
        return X @ self.coef_

    def score(self, X, y):
        y_pred=  self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class CoefficientFeatureMatrix:
    """Object to hold a coefficient matrix alonside the input and output feature names"""
    def __init__(self, coefficients:Union[np.array, None], in_features:Union[np.array, None], out_features:Union[np.array, None]):
        self.coefficients = coefficients
        self.in_features = in_features
        self.out_features = out_features

    def save(self, root_path:str):
        np.save(os.path.join(root_path, 'coefficients.npy'), self.coefficients)
        np.save(os.path.join(root_path, 'in_features.npy'), self.in_features)
        np.save(os.path.join(root_path, 'out_features.npy'), self.out_features)
    
    def load(self, root_path:str):
        self.coefficients = np.load(os.path.join(root_path, 'coefficients.npy'))
        self.in_features = np.load(os.path.join(root_path, 'in_features.npy'))
        self.out_features = np.load(os.path.join(root_path, 'out_features.npy'))


class ModularTranscriptSpace:

    def __init__(self, in_features:list[ModelFeature], out_feature:GeneExpressionFeature, alphas=list[float]):
        self.in_features = in_features
        
        self.out_feature = out_feature
        self.out_feature.compute_feature()

        #Compute feature for every feature
        for feature in self.in_features:
            feature.compute_feature()
    
        self.alphas = alphas
        
    def fit(self, **kwargs):
        #For each gene train a new MASSENLasso and add the coefficient slice to the coefficient matrix
        #Coefficient matrix of shape (in_features, out_features)

        in_feature_dim = sum([feature.G.shape[1] for feature in self.in_features])
        out_feature_dim = self.out_features[0].data.G.shape[1]

        in_feature_names = flatten_list([list(feature.get_feature()[1].values()) for feature in self.in_features])
        out_feature_names = self.out_features[0].data.gene_names

        self.coefficients = np.zeros((in_feature_dim, out_feature_dim))

        #Get l1 ratio, n_resamples, and stability threshold from kwargs
        l1_ratio = kwargs.get('l1_ratio', 0.5)
        n_resamples = kwargs.get('n_resamples', 50)
        stability_threshold = kwargs.get('stability_threshold', 0.5)

        out_path = kwargs.get('out_path', 'coefficients')

        for i in tqdm(range(out_feature_dim)):
            feature_attributes = []
            for feature in self.in_features:
                epoch_feature_matrix, epoch_idx2feature, epoch_feature_alpha = feature.get_feature(exclude_genes=[self.out_features[0].data.idx2gene[i]], alpha=self.alphas[self.in_features.index(feature)])
                feature_dict = {'matrix': epoch_feature_matrix, 'idx2feature': epoch_idx2feature, 'alpha': epoch_feature_alpha}
                feature_attributes.append(feature_dict)
            
            X = np.concatenate([feature['matrix'] for feature in feature_attributes], axis=1)
            y = self.out_feature.G[:, i]

            model = MASSENLasso(l=1e-3, alphas=flatten_list([fd['alpha'] for fd in feature_attributes]), l1_ratio=l1_ratio, n_resamples=n_resamples, stability_threshold=stability_threshold)
            model.fit(X, y)
            
            coeffs = model.coef_
            #Insert 0s for the genes that were excluded
            for feature in feature_attributes:
                if feature['idx2feature'] == self.out_features[0].data.idx2gene[i]:
                    continue
                else:
                    coeffs = np.insert(coeffs, feature['idx2feature'], 0)
            
            self.coefficients[:, i] = coeffs

        self.coefficient_matrix = CoefficientFeatureMatrix(self.coefficients, in_feature_names, out_feature_names)
        self.coefficient_matrix.save(out_path)
    

class CoefficientAnalysis:

    def __init__(self, coefficient_matrix:CoefficientFeatureMatrix, graph_threshold):
        self.coefficient_matrix = coefficient_matrix

        self.graph = self.build_coefficient_graph(graph_threshold)

    def build_coefficient_graph(self, threshold:float):
        G = nx.Graph()

        for i, in_feature in enumerate(self.coefficient_matrix.in_features):
            for j, out_feature in enumerate(self.coefficient_matrix.out_features):
                weight = self.coefficient_matrix.coefficients[i, j]
                if i != j and weight != 0 and abs(weight) > threshold:
                    G.add_edge(in_feature, out_feature, weight=weight)
        
        return G
    
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
    #Mark correlation function
    #Bivariate spatial dependence
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
        
        distances = squareform(pdist(position)) #Get pariwise euclidean distances between cells

        #Now we need to deal with the spatial weights matrix
        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0) #No self connections

        expression_centered = expression - np.mean(expression, axis=0)

        WG = W @ expression_centered

        numerator = np.sum(expression_centered * WG, axis=0)
        denominator = np.sum(expression_centered ** 2, axis=0)

        #Scale by the number of cells and the sum of weights
        n_cells = expression.shape[0]
        sum_weights = np.sum(W, axis=0)

        morans_I = (n_cells / sum_weights) * (numerator / denominator)

        return morans_I

    def compute_geary_C(self, **kwargs):
        #Get threshold distance
        threshold_dist = kwargs.get('threshold_dist', 1.0)

        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        #Spatial weights matrix

        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0)

        W_sum = np.sum(W, axis=0)

        expression_centered = expression - np.mean(expression, axis=0)

        differences_squared = (expression_centered[:, np.newaxis, :] - expression_centered[np.newaxis, :, :]) ** 2
        weighted_diffs = W[:, :, np.newaxis] * differences_squared

        numerator = np.sum(weighted_diffs, axis=(0, 1))
        denominator = np.sum(expression_centered**2, axis=0)*2

        N = expression.shape[0]
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
        denominator = expression_std * np.sqrt((N * W_squared_sum - W_sum ** 2) / (N - 1))[:, np.newaxis]

        Gi_values = numerator/denominator
        
        return Gi_values
    
    def compute_ripleys_K(self, **kwargs):
        #Get distances to evaluate
        distances = kwargs.get('distances', np.linspace(0, 1, 100))
        area = kwargs.get('area', 1.0)

        expression, position = self.get_expression_position_(kwargs)
        distances = squareform(pdist(position))

        N = expression.shape[0]

        indicators = (distances[:, :, np.newaxis] <= distances).astype(float)
        weights = np.ones_like(indicators)
        weighted_sums = np.sum(indicators * weights, axis=(0, 1))

        Kv = (area / (N**2)) * weighted_sums

        return Kv

    def compute_lisa(self, **kwargs):
        threshold_dist = kwargs.get('threshold_dist', 1.0)
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))

        W = (distances < threshold_dist).astype(float)
        np.fill_diagonal(W, 0)

        N = expression.shape[0]
        X_mean = np.mean(expression, axis=0)
        X_var = np.var(expression, axis=0, ddof=1)

        X_centered = expression - X_mean

        spatial_lag = W @ X_centered

        lisa = (X_centered / X_var) * spatial_lag

        return lisa
    
    def compute_disperion_index(self, **kwargs):
        """Also does not take into account spatial information"""

        expression, _ = self.get_expression_position_(kwargs)

        expression_mean = np.mean(expression, axis=0)
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
        denominator = np.outer(norm, norm)

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
        threshold_distance = kwargs.get('threshold_distance', 1.0)
        distances_to_evaluate = kwargs.get('distances', np.linspace(0, 1, 100))
        expression, position = self.get_expression_position_(kwargs)

        distances = squareform(pdist(position))
        N = position.shape[0]

        mark_corr_values = np.zeros((len(distances_to_evaluate), expression.shape[1]))
        mark_products = np.einsum('ij,ik->ijk', expression, expression)

        indicators = distances[: , :, np.newaxis] <= distances_to_evaluate

        weighted_mark_sum = np.einsum('ijk,ij->ik', indicators, mark_products)

        valid_pairs = np.sum(indicators, axis=(0, 1))

        valid_pairs[valid_pairs == 0] = 1
        mark_corr_values = weighted_mark_sum / valid_pairs[:, np.newaxis]

        return mark_corr_values

