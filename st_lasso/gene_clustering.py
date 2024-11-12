import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from st_lasso import SpatialTranscriptomicsData, SpatialStatistics
from sklearn.cluster import KMeans

if __name__ == "__main__":
    coefficients = np.load('go_coefficients.npz')
    coeffs = coefficients['coefficients']
    in_feature_names = coefficients['in_feature_names']
    out_feature_names = coefficients['out_feature_names']
    #print(coeffs.shape)

    full_coefficients = np.load('coefficients.npz')
    full_coeffs = full_coefficients['coefficients']
    full_in_feature_names = full_coefficients['in_feature_names']
    full_out_feature_names = full_coefficients['out_feature_names']
    #print(full_coeffs.shape)

    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())
    st = SpatialTranscriptomicsData(G, P, T, annotations)
    spatial_statistics = SpatialStatistics(st)

    #covariance_matrix = spatial_statistics.compute_gene_covariance_matrix(cell_type='Treg')

    #Get only the gene indices in out feature names
    gene_indices = [st.gene2idx[gene] for gene in out_feature_names]

    #Pad the coefficients with zeros to match the full dataset
    _coeffs = np.zeros((full_coeffs.shape[0], coeffs.shape[1]))
    _coeffs[29:] = coeffs
    #print(_coeffs.shape)

    #Get gene expression values for the genes in the out feature names for Tregs
    treg_idx = st.celltype2idx['Treg']
    treg_exp = st.G[np.where(st.T == treg_idx)]
    treg_exp = treg_exp[:, gene_indices]
    print(treg_exp.shape)

    #Save those gene indices to a json
    gene_names = [list(st.idx2gene.values())[i] for i in gene_indices]
    with open('treg_genes.json', 'w') as f:
        json.dump(gene_names, f)

    
