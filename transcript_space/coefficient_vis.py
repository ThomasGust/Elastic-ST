import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#stats = np.load('sample_statistics\\T CD8 memory_report.npz')
#morans_i = stats['morans_I']
#indices = np.argsort(morans_i)[::-1][:200]

if __name__ == "__main__":
    coefficients = np.load('treg.npz')
    coeffs = coefficients['coefficients']
    in_feature_names = coefficients['in_feature_names']
    out_feature_names = coefficients['out_feature_names']
    print(coeffs.shape)

    sns.heatmap(coeffs, cmap='viridis', center=0)
    plt.yticks(ticks=np.arange(len(in_feature_names)), labels=in_feature_names)
    plt.xticks(ticks=np.arange(len(out_feature_names)), labels=out_feature_names, rotation=90)
    plt.title("Treg Coefficients", fontsize=16)
    plt.show()

    
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
"""
"""
    for i in range(full_coeffs.shape[0]):
        #Iterate through indices of the coefficients
        for j in range(coeffs.shape[0]):
            with_feature = full_coeffs[i, j]
            without_feature = _coeffs[i, j]

            if with_feature == without_feature:
                adjacency_comparison[i, j] = 0
            
            elif with_feature == 0 and without_feature != 0:
                adjacency_comparison[i, j] = -1
            
            elif with_feature != 0 and without_feature == 0:
                adjacency_comparison[i, j] = 1
"""
"""
    #Redo with np.where
    adjacency_comparison = np.where(full_coeffs == _coeffs, 0, full_coeffs - _coeffs)


    sns.heatmap(adjacency_comparison, cmap='viridis', center=0)
    plt.yticks(ticks=np.arange(len(full_in_feature_names)), labels=full_in_feature_names)
    plt.xticks(ticks=np.arange(len(full_out_feature_names)), labels=full_out_feature_names, rotation=90)
    plt.title("Adjacency Comparison With and Without Spatial Features", fontsize=16)
    plt.show()
"""