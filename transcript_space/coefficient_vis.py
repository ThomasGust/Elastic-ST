import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transcript_space import CoefficientAnalysis

#stats = np.load('sample_statistics\\T CD8 memory_report.npz')
#morans_i = stats['morans_I']
#indices = np.argsort(morans_i)[::-1][:200]

if __name__ == "__main__":
    coefficients = np.load('treg.npz')
    analysis = CoefficientAnalysis(coefficients, 0.08)
    analysis.plot_coefficient_graph()

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