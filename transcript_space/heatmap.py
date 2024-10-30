import seaborn as sns
from transcript_space import SpatialTranscriptomicsData
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np



if __name__ == "__main__":
    st = SpatialTranscriptomicsData(root_path='C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer', name='colon_cancer')
    _i = np.where(st.T == st.celltype2idx['Treg'])
    expression = np.log1p(st.G[_i][:, st.gene2idx['NDUFA3']])
    x = st.P[_i][:, 0]
    y = st.P[_i][:, 1]
    data = pd.DataFrame({
        'x': x,  # X coordinates
        'y': y,  # Y coordinates
        'expression': expression # Gene expression levels
    })

    # Step 2: Apply KDE for spatial smoothing
    # Create a 2D grid over the area of interest
    x_grid, y_grid = np.mgrid[0:5:100j, 0:5:100j]
    grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Use KDE with weighted expression levels to create a smooth gradient
    values = np.vstack([data['x'], data['y']])
    expression = data['expression']
    kde = gaussian_kde(values, weights=expression, bw_method=0.3)  # Adjust bandwidth for smoothness
    kde_values = kde(grid_positions).reshape(x_grid.shape)

    # Step 3: Plot the smoothed gradient map
    plt.figure(figsize=(8, 6))
    plt.imshow(kde_values, extent=(0, 5, 0, 5), origin='lower', cmap='viridis', alpha=0.8)
    plt.colorbar(label='Gene Expression Level')
    #plt.scatter(data['x'], data['y'], c=data['expression'], cmap='viridis', edgecolor='white', s=50)
    plt.title("Spatial Gradient of Gene Expression")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

