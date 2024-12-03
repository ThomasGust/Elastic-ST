# Elastic-ST
![Spatial Transcriptomics Header](https://www.science.org/do/10.1126/webinar.adc3330/abs/resolve_16x9large_v1.png)

Elastic-ST is a python package designed to let scientists learn more from their spatial transcriptomics data. Using pre-typed spatial transcriptomics data, users train a series of ElasticNet models to build a biological graph for a cell type or cell population of interest.
These networks include features generated from the position information provided spatial transcriptomics techniques (such as neighborhood cell type or metagene abundance), providing a level of understanding beyond networks derived from scRNA-seq techniques. 

Elastic-ST also features a new, blazingly fast Cython implementation of a weighted elastic net. Users can specify different regularization penalties for each feature in their dataset, allowing fine control over how central each feature group is in the final networks, and opening up exploration of a whole new range of questions.
In this manner, spatial information doesn't need to be drowned out by the single cell transcriptome.

Below, an example of how spatial information can change a biological network:
![B Cell Graph Topology Comparison](https://github.com/ThomasGust/Elastic-ST/blob/main/figures/adjacency/b_cell_topology_comparison.png)

# Installation
Installation is simple through pip (not yet available)

```bash
pip install elastic_st
```

or alternatively install directly through GitHub

```bash
pip install git+https://github.com/ThomasGust/Elastic-ST.git
```
# Getting Started
To get started, users need 3 basic data primitives. Firstly, a cells by genes shaped expression matrix to hold the raw transcripts per cell information. Secondly, a cells by 2 shaped position matrix to carry spatial information is required. And thirdly, Elastic-ST also requires a (cells,) shaped array representing the cell type of each cell in the dataset.
For interpretability, users must also provide a list of gene names, and a list of cell types to create a dataset.

All datasets are represented through the SpatialTranscriptomicsData object:

```python
import elastic_st as est
import numpy as np
import json

if __name__ == "__main__":
  G = np.load('data\\G.npy') # (cells, genes)
  P = np.load('data\\P.npy') # (cells, 2)
  T = np.load('data\\T.npy') # (cells,)

  annotations = json.load(open('data\\annotations.json'))
  gene_names = annotations['gene_names'] # (genes,)
  cell_types = annnotations['cell_types']# (cell_types,)

  data = SpatialTranscriptomicsData(G, P, T, gene_names=gene_names, cell_types=cell_types)
```

# Examples
For a full list of examples, please refer to the examples folder of this repository. Below are a few common techniques to start moving with the library.

## Model Training
The core function of Elastic-ST is to create a semi-sparse coefficient matrix by training a bunch of ElasticNet models. This functionality is provided through the ElasticST module and it's associated feature objects that allow the inclusion of spatial information.

```python
import elastic_st as est
import numpy as np
import json

if __name__ == "__main__":
    #Load all raw data for the SpatialTranscriptomicsData dataset
    G = np.load('data/G.npy')
    P = np.load('data/P.npy')
    T = np.load('data/T.npy')

    annotations = json.load(open('data/annotations.json'))
    cell_types = annotations['cell_types']
    gene_names = annotations['gene_names']

    #Initialize dataset and features
    data = est.SpatialTranscriptomicsData(G, P, T, gene_names, cell_types)

    #Have a 5x weaker penalty for the spatial features
    metagene_abundance_feature = est.MetageneAbundanceFeature(bias=5, data=data, cell_type='B-cell', metagenes={'Checkpoints': ['CTLA4', 'CD274', 'TIGIT']}, radius=0.1)
    cell_type_abundance_feature = est.CellTypeAbundanceFeature(bias=5, data=data, cell_type='B-cell', radius=0.1)

    #A variance threshold is applied to make sure we don't pick up any 'useless' or 'unimportant' genes. This also speeds up model training
    data.variance_filter(threshold=0.2)

    #Train and save the model
    model = est.ElasticST(data, [metagene_abundance_feature, cell_type_abundance_feature], cell_type='B-cell', alpha=0.05, l1_ratio=0.5, subsample_to=5000)
    coeffs = model.fit(n_jobs=-1)
    np.savez_compressed('coefficients.npy', **coeffs)
```

## Network Visualization
A potential network for b-cells (with high spatial bias):

![B Cell Network](https://github.com/ThomasGust/Elastic-ST/blob/main/figures/networks/b_cell_network.png)

Once we have trained our model, we will want to build and plot the biological network from our model's coefficient matrix. This, and several other network analysis utilities are included through the CoefficientGraphAnalysis object.

```python
import elastic_st as est
import numpy as np

if __name__ == "__main__":
    #Load coefficients record and parse it
    coeffs = np.load('coefficients.npy.npz')

    coefficients = coeffs['coefficients']
    target_names = coeffs['target_names']
    feature_names = coeffs['feature_names']

    #Create the analysis object and plot the graph
    analysis = est.CoefficientGraphAnalysis(coefficients, feature_names, target_names, graph_threshold=0.07)
    analysis.plot_graph(show=True, node_size=10, width=0.1, font_size=10, with_labels=True)

    #Also find the most central nodes as an example of graph analysis
    degree_centrality = analysis.get_graph_degree_centrality()

    #Sort the nodes by centrality
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    #Print the most central nodes
    print("Most central nodes:")
    for node, centrality in sorted_nodes[:10]:
        print(f"{node}: {centrality}")

```

## Spatial Heatmapping
It is always necessary to make sure the results of any computational technique have some basis in reality. Elastic-ST provides the ability to create spatial plots of gene or metagene expression such as those seen below:

<div justify='center'>
<img src="https://github.com/ThomasGust/Elastic-ST/blob/main/figures/heatmaps/Checkpoints_epc1.png" width="200" height="200" /><img src="https://github.com/ThomasGust/Elastic-ST/blob/main/figures/heatmaps/Checkpoints_epc2.png" width="200" height="200" /><img src="https://github.com/ThomasGust/Elastic-ST/blob/main/figures/heatmaps/cd74_b_cell.png" width="200" height="200" />
</div>

<div justify='center'>
<img src="https://github.com/ThomasGust/Elastic-ST/blob/main/figures/heatmaps/cd74_macrophage.png" width="200" height="200" /><img src="https://github.com/ThomasGust/Elastic-ST/blob/main/figures/heatmaps/gzmm_tcd8.png" width="200" height="200" /><img src="https://github.com/ThomasGust/Elastic-ST/blob/main/figures/heatmaps/pigr_treg.png" width="200" height="200" />
</div>

```python
import elastic_st as est
import numpy as np
import json

if __name__ == "__main__":
    G = np.load('data/G.npy')
    P = np.load('data/P.npy')
    T = np.load('data/T.npy')

    annotations = json.load(open('data/annotations.json'))
    cell_types = annotations['cell_types']
    gene_names = annotations['gene_names']

    data = est.SpatialTranscriptomicsData(G, P, T, gene_names, cell_types)
    data.group_genes({'Checkpoints': ['CTLA4', 'CD274', 'TIGIT']}, by=np.mean) #This example aims to see the distribution of a few checkpoints in cancer cells.
    # Grouping scheme above is np.mean, by default it is np.sum for determining gene set expression for each cell.

    #Plot the spatial heatmap for the first cancer subtype in our data, plot as a fraction of total cellular expression instead of raw expression.
    est.plot_heatmap(data, cell_type='epithelial.cancer.subtype_1', gene_name='Checkpoints', show=True, as_fraction=True)
```

## Spatial Statistics
In order to gain better context and insights into the models output, Elastic-ST also provides a module, SpatialStatistics, to compute a few common spatial statistics indicators such as moran's I, geary's C etc...

```python
import elastic_st as est
import numpy as np
import json

if __name__ == "__main__":
    G = np.load('data/G.npy')
    P = np.load('data/P.npy')
    T = np.load('data/T.npy')

    annotations = json.load(open('data/annotations.json'))
    cell_types = annotations['cell_types']
    gene_names = annotations['gene_names']

    data = est.SpatialTranscriptomicsData(G, P, T, gene_names, cell_types)
    statistics = est.SpatialStatistics(data)

    out = statistics.full_report(cell_type='macrophage', verbose=True)
    print(list(out.keys))
    np.savez('statistics_report_macrophage.npz', **out)
```
