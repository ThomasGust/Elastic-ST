import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

class SpatialStatistics:
    """
    Module for computing spatial statistics on spatial transcriptomics data. We can get stuff like the genexgene covariance matrix and spatial autocorrelation.
    """

    def __init__(self, data):
        self.data = data
    
    def get_expression_position_(self, kwargs):
        cell_type = kwargs.get('cell_type', None)
        if cell_type is None:
            expression = self.data.G
            position = self.data.P
        else:
            position = self.data.get_position_by_cell_type(cell_type)
            expression = self.data.get_expression_by_cell_type(cell_type)
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
        verbose = kwargs.get('verbose', False)

        expression, position = self.get_expression_position_(kwargs)
        N = expression.shape[0]

        # Center the expression values
        expression_centered = expression - np.mean(expression, axis=0)
        denominator = (np.sum(expression_centered**2, axis=0) * 2)+1e-6

        tree = cKDTree(position)

        numerator = np.zeros(expression.shape[1], dtype=np.float64)
        W_sum = 0

        if verbose:
            iterator = tqdm(range(N))
        else:
            iterator = range(N)

        for i in iterator:
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

        if kwargs.get('verbose', False):
            iterator = tqdm(range(N))
        else:
            iterator = range(N)
        for idx, d in enumerate(distances_to_evaluate):
            weighted_mark_sum = np.zeros(expression.shape[1], dtype=np.float64)
            valid_pairs = 0

            for i in iterator:
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
        verbose = kwargs.get('verbose', False)
        #Compute all statistics
        gene_covariance = self.compute_gene_covariance_matrix(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED GENE COVARIANCE')
        morans_I = self.compute_moran_I(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED MORANS I')
        gearys_C = self.compute_geary_C(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED GEARYS C')
        getis_ord_Gi = self.compute_getis_ord_Gi(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED GETIS ORD GI')
        ripley_K = self.compute_ripleys_K(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED RIPLEYS K')
        lisa = self.compute_lisa(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED LISA')
        dispersion_index = self.compute_disperion_index(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED DISPERSION INDEX')
        spatial_cross_correlation = self.compute_spatial_cross_correlation(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED SPATIAL CROSS CORRELATION')
        spatial_co_occurence = self.compute_spatial_co_occurence(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED SPATIAL CO OCCURENCE')
        mark_correlation_function = self.compute_mark_correlation_function(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED MARK CORRELATION FUNCTION')
        bivariate_morans_I = self.bivariate_morans_I(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED BIVARIATE MORANS I')
        spatial_eigenvectors, spatial_eigenvalues = self.spatial_eigenvector_mapping(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED SPATIAL EIGENVECTORS + EIGENVALUES')
        mean_expression= self.get_mean_expression(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED MEAN EXPRESSION')
        variance_expression = self.get_variance_expression(**kwargs)
        if verbose: print('SPATIAL STATISTICS COMPUTED VARIANCE EXPRESSION')

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