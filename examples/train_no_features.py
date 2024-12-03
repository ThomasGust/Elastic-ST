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

    #Initialize dataset
    data = est.SpatialTranscriptomicsData(G, P, T, gene_names, cell_types)

    #Filter out 'unimportant' genes
    data.variance_filter(threshold=0.2)

    #Train and save the model
    model = est.ElasticST(data, [], cell_type='B-cell', alpha=0.05, l1_ratio=0.5, subsample_to=5000)
    coeffs = model.fit(n_jobs=-1)
    np.savez_compressed('coefficients.npy', **coeffs)