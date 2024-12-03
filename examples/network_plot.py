import elastic_st as est
import numpy as np

if __name__ == "__main__":
    #Load coefficients record and parse it
    coeffs = np.load('coefficients.npy.npz')

    coefficients = coeffs['coefficients']
    target_names = coeffs['target_names']
    feature_names = coeffs['feature_names']

    #Create the analysis object and plot the graph
    analysis = est.CoefficientGraphAnalysis(coefficients, feature_names, target_names, graph_threshold=0.07)
    analysis.plot_graph(show=True, node_size=10, width=0.1, font_size=10, with_labels=True)

    #Also find the most central nodes
    degree_centrality = analysis.get_graph_degree_centrality()

    #Sort the nodes by centrality
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    #Print the most central nodes
    print("Most central nodes:")
    for node, centrality in sorted_nodes[:10]:
        print(f"{node}: {centrality}")

    