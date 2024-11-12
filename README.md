# STLasso
A python data science library to create sparse coefficient matrices from spatial transcriptomics data.

# Installation
Leaving this blank for now, none of the libraries are super exotic so it should hopefully distribute really easily once it is up on Pypi.

# Features
STLasso offers a several features to build off of existing analysis tools for spatial transcriptomics data:

1. Coefficient Analysis. In order to work with computed model coefficients ST-Lasso provides simple network generation, plotting, and analysis utilities. StLasso also passes through some basic networkx functionality like the ability to find:
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
  
  
3.
