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