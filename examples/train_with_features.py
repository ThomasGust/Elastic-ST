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

    metagene_abundance_feature = est.MetageneAbundanceFeature(bias=5, data=data, cell_type='B-cell', metagenes={'Checkpoints': ['CTLA4', 'CD274', 'TIGIT']}, radius=0.1)
    cell_type_abundance_feature = est.CellTypeAbundanceFeature(bias=5, data=data, cell_type='B-cell', radius=0.1)

    #Filter out 'unimportant' genes
    data.variance_filter(threshold=0.2)

    #Train and save the model
    model = est.ElasticST(data, [metagene_abundance_feature, cell_type_abundance_feature], cell_type='B-cell', alpha=0.05, l1_ratio=0.5, subsample_to=5000)
    coeffs = model.fit(n_jobs=-1)
    np.savez_compressed('coefficients.npy', **coeffs)