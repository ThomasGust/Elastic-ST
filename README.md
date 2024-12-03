# Elastic-ST
![Spatial Transcriptomics Header](https://www.science.org/do/10.1126/webinar.adc3330/abs/resolve_16x9large_v1.png)

Elastic-ST is a python package designed to let scientists learn more from their spatial transcriptomics data. Using pre-typed spatial transcriptomics data: an expression matrix, a position matrix, and a cell type matrix, users train a series of ElasticNet models to build a biological graph for a cell type or cell population of interest.
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
pip install pip install git@github.com:ThomasGust/Elastic-ST.git
```
