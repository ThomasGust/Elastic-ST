# STLasso
A python data science library to create sparse coefficient matrices from spatial transcriptomics data.

# Features
### Examples of how to use STLasso can be further found below but here is a synopsis of the features currently provided:

STLasso offers a several features to build off of existing analysis tools for spatial transcriptomics data:

1. Coefficient Matrix Computation. At the core of the package, users can compute a coefficient matrix relating every gene and spatial feature to every other gene. This essentially infers a genetic network for a subpopulation of cells and can provide insight into how a given cell type is regulated and how it is affected by its environment.

 We allow users to engineer whatever input features they want, but also include two by default in the package:
   * Neighborhood Cell Type Abundance. Counts the abundance of each cell type within a certain radius of each cell of interest.
   
   * Neighborhood Metagene Score. Finds the expression of a given metagene or gene set within a certain radius of each cell of interest.

We also allow users to 'remap' their raw transcript expression data to metagenes/metatranscripts  to better capture genetic functional modules.

2. Coefficient Analysis. In order to work with computed model coefficients ST-Lasso provides simple network generation, plotting, and analysis utilities. StLasso also passes through some basic networkx functionality like the ability to find:
   * Graph Communities
   * Graph Cliques
   * Graph Components
   * Graph Diameter
   * Node Centrality and Degree
   * Graph Transitivity
   * Graph Average Shortest Path Length
   * Graph Dominating Set

3. Compute basic spatial statistics to better characterize spatially variable genes in your data such as:
   * Moran's I
   * Gearys C
   * Getis-Ord Gi*
   * Ripley's K
   * Local Indicators of Spatial Autocorrelation
   * Dispersion Index
   * Between Gene Spatial Cross Correlation
   * Spatial Co-Occurence
   * Mark Correlation Function
   * Bivariate Moran's I
   * Spatial Eigenvector Mapping
   * Non-spatial Gene-Gene Correlation

4. Gene Heatmap Plotting. To visualize spatial patterns, we also allow users to create smoothed, spatial heatmap of gene expression.

# Installation
Leaving this blank for now, none of the libraries are super exotic so it should hopefully distribute really easily once it is up on Pypi.

# Examples
Firstly, how can we create data? Well, for STLasso, all data is initialized with 3 matrices, the (Cells, Genes) shape expression matrix G, the (Cells, Cell Types) shape cell type matrix T, and the (Cells, 2) shape position matrix P. Additionally, so we can understand the features, users must also provide a list 'cell_types', and a list 'gene_names' to initialize the SpatialTranscriptomicsData object. In code:

```
import st-lasso as stl
import numpy as np
import json

if __name__ == "__main__":
   G = np.load('data\\colon_cancer\\G.npy')
   P = np.load('data\\colon_cancer\\P.npy')
   T = np.load('data\\colon_cancer\\T.npy')
   annotations = json.loads(open('data\\colon_cancer\\annotations.json'))

   cell_types = annotations['cell_types']
   gene_names = annotations['gene_names']

   data = SpatialTranscriptomicsData(G, P, T, gene_names=gene_names, cell_tpes=cell_types)
   print("Successfully Created SpatialTranscriptomicsData object")
```
