#This example plots the topological difference between two graphs with different nodes (one trained with spatial features, the other without).

import elastic_st as est
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":
    #Load and parse both coefficient records
    with_coeffs = np.load('coefficients.npy.npz')
    without_coeffs = np.load('coefficients_no_features.npy.npz')

    with_coefficients = with_coeffs['coefficients']
    with_target_names = with_coeffs['target_names']
    with_feature_names = with_coeffs['feature_names']

    without_coefficients = without_coeffs['coefficients']
    without_target_names = without_coeffs['target_names']
    without_feature_names = without_coeffs['feature_names']

    #Create the analysis objects to harvest adjacency matrices
    with_analysis = est.CoefficientGraphAnalysis(with_coefficients, with_target_names, with_feature_names, graph_threshold=0.07)
    without_analysis = est.CoefficientGraphAnalysis(without_coefficients, without_target_names, without_feature_names, graph_threshold=0.07)

    #Make sure to convert adjacency matrices to binary (0, 1) vs weighted connections
    withadj, withorder = with_analysis.get_adjacency_matrix(as_binary=True)
    withoutadj, withoutorder = without_analysis.get_adjacency_matrix(as_binary=True)

    #Compute the comparison, 0 is no edges in either, 1 is only in with, 2 is only in without, 3 is in both
    change_matrix, all_nodes = est.compare_adjacency_matrices(withadj, withorder, withoutadj, withoutorder)

    #Create a colormap so that each value in the matrix corresponds to a color
    cmap = mcolors.ListedColormap(['white', 'red', 'blue', 'purple'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(12, 12))
    im = plt.imshow(change_matrix, cmap=cmap, norm=norm)

    # Set the ticks and labels
    plt.xticks(ticks=np.arange(len(all_nodes)), labels=all_nodes, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(all_nodes)), labels=all_nodes, fontsize=8)

    # Add a colorbar with labels
    cbar = plt.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['0: No Edge', '1: Only in Feature Inclusive Graph', '2: Only in Featureless Graph', '3: In Both Graphs'])
    plt.title("Topological Changes in Coefficient Graphs: Features vs. No Features")
    plt.tight_layout()

    # Display the plot
    plt.show()