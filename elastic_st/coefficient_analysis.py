import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from typing import Union, Tuple

def compare_adjacency_matrices(adj1:np.array, adj2:np.array, order1:list, order2:list) -> Tuple[np.array, list]:
    """
    This function compares two adjacency matrices and returns a change matrix that shows the topological differences between the two graphs. The change matrix is a matrix where each element represents the change in the edge between two nodes. The change matrix is defined as follows:
    - 0: No edge in either graph
    - 1: Edge in the first graph, not in the second
    - 2: Edge in the second graph, not in the first
    - 3: Edge in both graphs

    Parameters:
        adj1 (np.array): The first adjacency matrix.
        adj2 (np.array): The second adjacency matrix.
        order1 (list): The order of nodes in the first graph.
        order2 (list): The order of nodes in the second graph.
    Returns:
        change_matrix (np.array): The change matrix.
        all_nodes (list): The order of nodes in the change matrix.
    """

    all_nodes = sorted(set(order1).union(set(order2)))

    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

    # Initialize new adjacency matrices for both graphs
    size = len(all_nodes)
    new_adj1 = np.zeros((size, size), dtype=int)
    new_adj2 = np.zeros((size, size), dtype=int)
    
    #These loops below are a convoluted way to align the nodes correctly, all they really do is create new adjacency matrices for each graph including the nodes in the other graph.
    #This allows us to compare the two adjacency matrices directly, because the order of the nodes is the same in both matrices.
    for i, node_i in enumerate(order1):
        idx_i = node_to_index[node_i]
        for j, node_j in enumerate(order1):
            idx_j = node_to_index[node_j]
            new_adj1[idx_i, idx_j] = adj1[i, j]

    for i, node_i in enumerate(order2):
        idx_i = node_to_index[node_i]
        for j, node_j in enumerate(order2):
            idx_j = node_to_index[node_j]
            new_adj2[idx_i, idx_j] = adj2[i, j]

    # Compute the =final change matrix
    change_matrix = new_adj1 + 2 * new_adj2

    return change_matrix, list(all_nodes)

class CoefficientGraphAnalysis:

    def __init__(self, coefficients:np.array, feature_names:Union[list[str], np.array], target_names:Union[list[str], np.array], graph_threshold:float=0.1):
        """
        A utility module for analyzing and plotting the coefficients of an Elastic-ST model in graph form. This is a much more intuitive way to visualize the relationships the model may have learned.
        Parameters:
            coefficients (np.array): The coefficients of the model.
            feature_names (list): The names of the features in the model.
            target_names (list): The names of the targets in the model.
            graph_threshold (float): The threshold for including an edge in the graph.
        """

        self.coefficients = coefficients
        self.feature_names = feature_names
        self.target_names = target_names
        self.graph_threshold = graph_threshold

        self.graph = self.create_graph()

    def create_graph(self) -> nx.Graph:
        """
        A vectorized approach to building a networkx graph from the coefficient matrix. This network is a much cleaner way to understand the model's output versus a raw coefficient matrix.

        Parameters:
            None (Acts on the class attributes)
        Returns:
            graph (nx.Graph): A networkx graph object representing the coefficients.
        """

        graph = nx.Graph()

        mask = np.abs(self.coefficients) > self.graph_threshold
        rows, cols = np.where(mask) #Gets the indices of all connections that meet the threshold.

        #This is faster than a nested for loop.
        edge_list = [
            (self.feature_names[j], self.target_names[i], self.coefficients[i, j]) for i, j in zip(rows, cols)
        ]


        graph.add_weighted_edges_from(edge_list)
        return graph
    
    def get_ego_graph(self, node:str, radius:int=1) -> nx.Graph:
        """
        Returns the ego graph of a node in the graph. An ego graph is just the subgraph of the graph that includes the node and all of its neighbors within a certain radius.
        For example, if I am interested in the CTLA4 gene, I can get the ego graph of CTLA4 with a radius of 1 to see all the genes that are directly connected to CTLA4.

        Parameters:
            node (str): The node of interest.
            radius (int): The radius of the ego graph.
        Returns:
            ego_graph (nx.Graph): The ego graph of the node
        """
        return nx.ego_graph(self.graph, node, radius=radius)
    
    def plot_graph(self, show=True, **kwargs) -> None:
        """
        Plots the graph using nx.draw, all kwargs are passed through to nx.draw.

        Parameters:
            show (bool): Whether to show the plot or leave as an undrawn figure.
            **kwargs: Additional keyword arguments for nx.draw.
        """
        nx.draw(self.graph, **kwargs)

        if show:
            plt.show()
    
    def get_adjacency_matrix(self, all_features=False, as_binary=True) -> Tuple[np.array, list]:
        """
        A function to find the adjacency matrix of the internal graph. Only features that were kept in the graph are included by default, but all features can be included if desired. (Many nodes may be cut during graph creation if they do not meet the graph threshold set by the user).
        This function also returns an ordered list of the nodes represented in the adjacency matrix.

        Parameters:
            all_features (bool): Whether to include all features in the adjacency matrix.
            as_binary (bool): Whether to return the adjacency matrix as a binary matrix.
        Returns:
            matrix (np.array): The adjacency matrix of the graph.
            node_order (list): The order of nodes in the adjacency matrix.
        """

        if all_features:
            G = self.graph
            # Ensure all nodes are added to the graph
            G.add_nodes_from(self.feature_names)
            G.add_nodes_from(self.target_names)
            
            # Create a list of all nodes in the desired order
            node_order = list(self.feature_names) + list(self.target_names)
            # Remove duplicates while preserving order
            node_order = list(dict.fromkeys(node_order))
            matrix = nx.to_numpy_array(G, nodelist=node_order)
        else:
            node_order = list(self.graph.nodes)
            matrix = nx.to_numpy_array(self.graph)
        
        if as_binary:
            matrix = np.where(matrix > 0, 1, 0)
        
        return matrix, node_order
        
    #Now we have some utility functions to pass through analysis functionality from networkx
    def get_graph_commmunities(self) -> list:
        """
        Gets the communities in the graph using the greedy modularity algorithm.

        Returns:
            communities (list): A list of communities in the graph. 
        """
        return list(nx.algorithms.community.greedy_modularity_communities(self.graph))

    def get_graph_cliques(self) -> list:
        """
        Gets the cliques in a graph. A clique is a subset of vertices of an undirected graph such that every two distinct vertices in the clique are adjacent.

        Returns:
            cliques (list): A list of cliques in the graph
        """
        return list(nx.algorithms.clique.find_cliques(self.graph))
    
    def get_graph_clique_number(self) -> int:
        """
        Finds the clique number of the graph. The clique number is the size of the largest clique in the graph in nodes.

        Returns:
            clique_number (int): The size of the largest clique in the graph.
        """
        return int(nx.graph_clique_number(self.graph))
    
    def get_graph_diameter(self) -> int:
        """
        Gets the total diameter of the graph. The diameter of a graph is the maximum eccentricity of any vertex in the graph.

        Returns:
            diameter (int): The diameter of the graph.
        """
        return int(nx.diameter(self.graph))
    
    def get_graph_degree_centrality(self) -> dict:
        """
        Finds a dictionary of the degree centrality of each node in the graph. The degree centrality of a node is the fraction of nodes it is connected to.

        Returns:
            degree_centrality (dict): A dictionary of the degree centrality of each node.
        """
        return nx.degree_centrality(self.graph)
    
    def get_graph_betweenness_centrality(self) -> dict:
        """
        Finds a dictionary of the betweenness centrality of each node in the graph. The betweenness centrality of a node is the fraction of shortest paths that pass through that node.

        Returns:
            betweenness_centrality (dict): A dictionary of the betweenness centrality of each node.
        """
        return nx.betweenness_centrality(self.graph)
    
    def get_graph_closeness_centrality(self) -> dict:
        """
        Gets a dictionary of the closeness centrality of each node in the graph. The closeness centrality of a node is the reciprocal of the sum of the shortest path distances from that node to all other nodes.

        Returns:
            closeness_centrality (dict): A dictionary of the closeness centrality of each node.
        """
        return nx.closeness_centrality(self.graph)
    
    def get_graph_eigenvector_centrality(self) -> dict:
        """
        Gets the eigenvector centrality of each node in the graph. The eigenvector centrality of a node is the eigenvector of the largest eigenvalue of the adjacency matrix of the graph.

        Returns:
            eigenvector_centrality (dict): A dictionary of the eigenvector centrality of each node.
        """
        return nx.eigenvector_centrality(self.graph)
    
    def get_graph_clustering_coefficient(self) -> dict:
        """
        Finds the clustering coefficient of each node in the graph. The clustering coefficient of a node is the fraction of possible triangles through that node that exist.

        Returns:
            clustering_coefficient (dict): A dictionary of the clustering coefficient of each node.
        """
        return nx.clustering(self.graph)
    
    def get_graph_transitivity(self) -> float:
        """
        Computes the transitivity of the graph. The transitivity of a graph is the ratio of triangles to triplets in the graph.

        Returns:
            transitivity (float): The transitivity of the graph.
        """
        return nx.transitivity(self.graph)
    
    def get_graph_average_shortest_path_length(self) -> float:
        """
        Finds the average shortest path length in the graph. The average shortest path length is the average shortest path length between all pairs of nodes in the graph.

        Returns:
            average_shortest_path_length (float): The average shortest path length in the graph.
        """
        return nx.average_shortest_path_length(self.graph)
    
    def get_graph_dominating_set(self) -> list:
        """
        Computes the dominating set of the graph. A dominating set is a subset of vertices such that every vertex is either in the dominating set or adjacent to a vertex in the dominating set.

        Returns:
            dominating_set (list): A list of nodes in the dominating set.
        """
        return list(nx.dominating_set(self.graph))
    
    def get_graph_degree(self) -> dict:
        """
        Gets the degree of each node in the graph. The degree of a node is the number of edges connected to that node.

        Returns:
            degree (dict): A dictionary of the degree of each node.
        """
        return dict(self.graph.degree())

if __name__ == "__main__":

    coefficients = np.load("featured_coefficients.npy.npz", allow_pickle=True)
    
    feature_names = coefficients['feature_names']
    target_names = coefficients['target_names']
    coefficients = coefficients['coefficients']

    graph_analysis = CoefficientGraphAnalysis(coefficients, feature_names, target_names, graph_threshold=0.1)
    adj, order = graph_analysis.get_adjacency_matrix(all_features=False)
    binary_adj = np.where(adj > 0, 1, 0)

    featureless_coefficients = np.load("coefficients.npy.npz", allow_pickle=True)

    feature_names = featureless_coefficients['feature_names']
    target_names = featureless_coefficients['target_names']
    coefficients = featureless_coefficients['coefficients']

    graph_analysis = CoefficientGraphAnalysis(coefficients, feature_names, target_names, graph_threshold=0.1)
    adj2, order2 = graph_analysis.get_adjacency_matrix(all_features=False)
    binary_adj2 = np.where(adj2 > 0, 1, 0)

    change_matrix, all_nodes = compare_adjacency_matrices(binary_adj, binary_adj2, order, order2)

    cmap = mcolors.ListedColormap(['white', 'red', 'blue', 'purple'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create the plot
    plt.figure(figsize=(12, 12))
    im = plt.imshow(change_matrix, cmap=cmap, norm=norm)

    # Set the ticks and labels
    plt.xticks(ticks=np.arange(len(all_nodes)), labels=all_nodes, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(all_nodes)), labels=all_nodes, fontsize=8)

    # Add a colorbar with labels
    cbar = plt.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['0: No Edge', '1: Only in Feature Inclusive Graph', '2: Only in Featureless Graph', '3: In Both Graphs'])
    plt.title("Topological Changes in Coefficient Graphs: Features vs. No Features, B-Cells")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Display the plot
    plt.show()
        