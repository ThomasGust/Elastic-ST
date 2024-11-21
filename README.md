# STLasso
STLasso gives scientists the ability to discover more about their spatial transcriptomics data by learning how environmental features impact cell expression.
STLasso trains sparse linear models (lasso's by default) to predict cellular gene expression from the rest of a cell's transcriptome, and features expressed in the environment.

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

```python
import stlasso as stl
import numpy as np
import json

if __name__ == "__main__":
   G = np.load('data\\colon_cancer\\G.npy')
   P = np.load('data\\colon_cancer\\P.npy')
   T = np.load('data\\colon_cancer\\T.npy')
   annotations = json.loads(open('data\\colon_cancer\\annotations.json'))

   cell_types = annotations['cell_types']
   gene_names = annotations['gene_names']

   data = stl.SpatialTranscriptomicsData(G, P, T, gene_names=gene_names, cell_tpes=cell_types)
   print("Successfully Created SpatialTranscriptomicsData object")
```

It can also be useful to load a feature set to use for a neighborhood metagene feature or to remap your SpatialTranscriptomicsData to metagenes (#TODO, make remapping compatible with the FeatureSetData module)
The recommended form to hold a feature set is a pandas dataframe with Genes rows and Metagenes columns. #Todo, clean up FeatureSetDataModule to accept more data inputs and not load on the interior.

```python
import stlasso as stl
import numpy as np
import json

if __name__ == "__main__":
   G = np.load('data\\colon_cancer\\G.npy')
   P = np.load('data\\colon_cancer\\P.npy')
   T = np.load('data\\colon_cancer\\T.npy')
   annotations = json.loads(open('data\\colon_cancer\\annotations.json'))

   cell_types = annotations['cell_types']
   gene_names = annotations['gene_names']

   data = stl.SpatialTranscriptomicsData(G, P, T, gene_names=gene_names, cell_types=cell_types)
   print("Successfully Created SpatialTranscriptomicsData object")

   feature_set = stl.FeatureSetData(path='cancer_annotations.csv', bin_key='+')
   print("Successfully Created FeatureSetData object")

   data.remap_metagenes(feature_set) # This currently doesn't work, as the remap_metagenes is messed up, yet another fix for code review.
   print(data.gene_names)
```

# core model, I still need to write docs for this obviously but I don't want to do that until it is absolutely finalized

Once you have your data loaded, it can be useful to make spatial gradients plots if you have any particular genes of interest such as those below:
<div align='center'>
<img src="https://github.com/ThomasGust/STLasso/blob/main/figs/MMP11.png" alt="drawing" width="200"/>
<img src="https://github.com/ThomasGust/STLasso/blob/main/figs/OLFM4.png" alt="drawing" width="200"/>
<img src="https://github.com/ThomasGust/STLasso/blob/main/figs/CD74.png" alt="drawing" width="200"/>
</div>
<div align='center'>
<img src="https://github.com/ThomasGust/STLasso/blob/main/figs/S100A6.png" alt="drawing" width="200"/>
<img src="https://github.com/ThomasGust/STLasso/blob/main/figs/CEACAM6.png" alt="drawing" width="200"/>
</div>

