#This example demonstrates how to create a spatial gradient plot for a gene of interest.
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