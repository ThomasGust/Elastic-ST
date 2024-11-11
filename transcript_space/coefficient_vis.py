import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transcript_space import CoefficientAnalysis, SpatialTranscriptomicsData
import json
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm


def binarize_coefficients(a, thresh):
    a[a <= thresh] = 0.0
    a[np.where(a != 0.0)] = 1.0
    return a
if __name__ == "__main__":
    
    coefficients = np.load('coefficients\\withc_features\\B-cell.npz')
    analysis = CoefficientAnalysis(coefficients['coefficients'], coefficients['in_feature_names'], coefficients['out_feature_names'], graph_threshold=0.0, norm=False)
    deg = analysis.get_graph_degree()

    deg = {k: v for k, v in sorted(deg.items(), key=lambda item: item[1])}
    print(deg)
    print(analysis.graph.edges('epithelial.cancer.subtype_2'))
    analysis.plot_coefficient_graph(multicolor=True)
    """efgd
    full_coefficients = np.load('coefficients\\with_features\\B-cell.npz')
    full_coefficient_matrix = full_coefficients['coefficients']
    full_coefficient_matrix = binarize_coefficients(full_coefficient_matrix, 0.00)

    coefficients = np.load('coefficients\\no_features\\B-cell.npz')
    coefficient_matrix = coefficients['coefficients']
    coefficient_matrix = binarize_coefficients(coefficient_matrix, 0.00)
    cm = np.zeros(full_coefficient_matrix.shape)

    print(full_coefficient_matrix.shape[0]-coefficient_matrix.shape[0])
    cm[full_coefficient_matrix.shape[0]-coefficient_matrix.shape[0]:] = coefficient_matrix
    coefficient_matrix = cm
    print(coefficient_matrix.shape, full_coefficient_matrix.shape)
    

    adjacency_comparison = np.zeros(full_coefficient_matrix.shape)
    adjacency_comparison[np.where((full_coefficient_matrix == 0) & (coefficient_matrix == 0))] = 0
    adjacency_comparison[np.where((full_coefficient_matrix == 0) & (coefficient_matrix != 0))] = -1
    adjacency_comparison[np.where((full_coefficient_matrix != 0) & (coefficient_matrix == 0))] = 1
    adjacency_comparison[np.where((full_coefficient_matrix != 0) & (coefficient_matrix != 0))] = 2

    print(np.unique(adjacency_comparison))
    
    plt.figure(figsize=(16, 12))
    plt.imshow(adjacency_comparison)
    plt.show()
"""
"""
print(analysis.gene_features)
#analysis.plot_coefficient_graph()

G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())
st = SpatialTranscriptomicsData(G, P, T,annotations=annotations)


for meta in tqdm(analysis.gene_features):
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
    plt.savefig(f'tregs\\{meta}.png')
    plt.close()
"""



"""
    coefficients = np.load('go_coefficients.npz')
    coeffs = coefficients['coefficients']
    in_feature_names = coefficients['in_feature_names']
    out_feature_names = coefficients['out_feature_names']
    print(coeffs.shape)

    full_coefficients = np.load('coefficients.npz')
    full_coeffs = full_coefficients['coefficients']
    full_in_feature_names = full_coefficients['in_feature_names']
    full_out_feature_names = full_coefficients['out_feature_names']
    print(full_coeffs.shape)

    #Pad the coefficients with zeros to match the full dataset
    _coeffs = np.zeros((full_coeffs.shape[0], coeffs.shape[1]))
    _coeffs[29:] = coeffs
    print(_coeffs.shape)

    #Get a matrix of edge added, edge removed, and no change
    adjacency_comparison = np.zeros((full_coeffs.shape[0], full_coeffs.shape[1]))

    #Iterate through indices of the full coefficients
    
    
    for i in range(full_coeffs.shape[0]):
        #Iterate through indices of the coefficients
        for j in range(coeffs.shape[0]):
            with_feature = full_coeffs[i, j]
            without_feature = _coeffs[i, j]

            if with_feature == 0 and without_feature == 0:
                adjacency_comparison[i, j] = 0
            
            elif with_feature == 0 and without_feature != 0:
                adjacency_comparison[i, j] = -1
            
            elif with_feature != 0 and without_feature == 0:
                adjacency_comparison[i, j] = 1
            
            elif with_feature != 0 and without_feature != 0:
                adjacency_comparison[i, j] = 2
    


    #Redo with np.where
    #adjacency_comparison = np.where(full_coeffs == _coeffs, 0, full_coeffs - _coeffs)

    #Plot the adjacency comparison
    plt.figure(figsize=(16, 12))
    sns.heatmap(adjacency_comparison, cmap='coolwarm', center=0, cbar=False)
    
    #Create a dictionary for the legend
    d = {-1: "Edge Removed", 0:"Never Existed", 1:"Edge Added", 2:"Existed in both"}
    #Legend
    for key in d:
        plt.plot([], [], color='k', label=d[key])


    #plt.figure(figsize=(16, 12))
    plt.yticks(ticks=np.arange(len(full_in_feature_names)), labels=full_in_feature_names)
    plt.xticks(ticks=np.arange(len(full_out_feature_names)), labels=full_out_feature_names, rotation=90)
    plt.title("Adjacency Comparison With and Without Spatial Features", fontsize=16)
    plt.show()

"""