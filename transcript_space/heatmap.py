import seaborn as sns
from transcript_space import SpatialTranscriptomicsData
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import json



if __name__ == "__main__":
    #st = SpatialTranscriptomicsData(root_path='C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer', name='colon_cancer')
    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())
    st = SpatialTranscriptomicsData(G, P, T, annotations)
    cancer_gene_sets = json.loads(open('c4.json').read())
    gene_set_names = list(cancer_gene_sets.keys())

    gene_sets = []
    for gene_set in gene_set_names:
        gene_sets.append((gene_set, list(cancer_gene_sets[gene_set]['geneSymbols'])))
    
    #st.remap_metagenes(gene_sets)
    
    #meta = 'GAVISH_3CA_MALIGNANT_METAPROGRAM_12_EMT_1'

    meta = 'LDHA'
    t = 'Treg'
    _i = np.where(st.T == st.celltype2idx[t])


    #expression = np.log1p(st.G[_i][:, st.gene2idx[meta]])
    #Instead of log, do a normalization between 0 and 1
    expression = st.G[_i][:, st.gene2idx[meta]]
    expression = (expression - np.min(expression)) / (np.max(expression) - np.min(expression))

    x = st.P[_i][:, 0]
    y = st.P[_i][:, 1]
    data = pd.DataFrame({
        'x': x,
        'y': y, 
        'expression': expression
    })

    x_grid, y_grid = np.mgrid[0:5:100j, 0:5:100j]
    grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Use KDE with weighted expression levels to create a smooth gradient
    values = np.vstack([data['x'], data['y']])
    expression = data['expression']
    kde = gaussian_kde(values, weights=expression, bw_method=0.3)  # Adjust bandwidth for smoothness
    kde_values = kde(grid_positions).reshape(x_grid.shape)

    plt.figure(figsize=(8, 6))
    plt.imshow(kde_values, extent=(0, 5, 0, 5), origin='lower', cmap='viridis', alpha=0.8)
    plt.colorbar(label=f'Gene Expression Level')
    #plt.scatter(data['x'], data['y'], c=data['expression'], cmap='viridis', edgecolor='white', s=50)
    plt.title(f"Spatial Gradient of Gene Expression for {meta} in {t}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()

