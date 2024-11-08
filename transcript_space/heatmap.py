import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transcript_space import CoefficientAnalysis, SpatialTranscriptomicsData
import json
import pandas as pd
from scipy.stats import gaussian_kde
import os

if __name__ == "__main__":
    coefficients = np.load('treg.npz')
    analysis = CoefficientAnalysis(coefficients, 0.08)
    #analysis.plot_coefficient_graph()

    #print(analysis.gene_features)
    #analysis.plot_coefficient_graph()

    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())
    st = SpatialTranscriptomicsData(G, P, T,annotations=annotations)


    for meta in tqdm(analysis.gene_features):
        m  = meta.replace("/", "")
        #print(m)
        path = f'tregs\\{m}.png'

        if os.path.exists(path):
            continue

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
        plt.savefig(path)
        plt.close()