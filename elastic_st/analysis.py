import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Union
import matplotlib.pyplot as plt

from elastic_st.data import SpatialTranscriptomicsData

def plot_heatmap(data:SpatialTranscriptomicsData, cell_type:str, gene_name:str, show:bool=False, return_kde:bool=False, as_fraction:bool=False, save_path:str=None, bw_method:float=0.3) -> Union[None, np.array]:
    """
    A function to plot a heatmap of the spatial expression of a gene in a cell type. Useful for visually determining if the expression of a gene follows some spatial organization.
    Helpful to validate the results of spatial statistics indicators or the Elastic-ST model.

    Parameters:
        data (SpatialTranscriptomicsData): The spatial transcriptomics data object.
        cell_type (str): The cell type to plot the heatmap for.
        gene_name (str): The gene to plot the heatmap for.
        show (bool): Whether to show the plot.
        return_kde (bool): Whether to return the KDE values. Useful for further analysis.
        as_fraction (bool): Whether to plot the gene expression as a fraction of the total expression in the cell.
        save_path (str): The path to save the plot to.
        bw_method (float): The bandwidth to use for the KDE. Smaller values will make the heatmap more sensitive to local changes in expression.
    """
    if cell_type == "all":
        #Fetch the full expression and position data in this case.
        expression = data.G[:, data.gene2idx[gene_name]]
        x = data.P[:, 0]
        y = data.P[:, 1]
    else:
        #Otherwise, we thankfully have a function in the data object that can get type specific information for us.
        expression = data.get_expression_by_cell_type(cell_type)
        
        if as_fraction:
            #If we do this as a fraction, we get the expression of the gene as a fraction of the total expression in the cell. Not really recommended, but its an option.
            expression = expression[:, data.gene2idx[gene_name]]/np.sum(expression, axis=1)
        else:
            expression = expression[:, data.gene2idx[gene_name]]
        
        pos = data.get_position_by_cell_type(cell_type)
        x = pos[:, 0]
        y = pos[:, 1]

    expression = (expression - np.min(expression)) / (np.max(expression) - np.min(expression)) #Normalize the expression for better visualization
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
    kde = gaussian_kde(values, weights=expression, bw_method=bw_method)  # Adjust bandwidth for smoothness
    kde_values = kde(grid_positions).reshape(x_grid.shape)
    if return_kde:
        #For if the user wants to do some other analysis with the spatial gradient.
        return np.array(kde_values)

    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(kde_values, origin='lower', extent=(0, 5, 0, 5), cmap='viridis', alpha=0.8)
    plt.colorbar(label='Expression')
    plt.title("Spatial Expression of " + gene_name + " in " + cell_type)
    plt.xlabel("x")
    plt.ylabel("y")

    #And show or save if the user wants.
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)