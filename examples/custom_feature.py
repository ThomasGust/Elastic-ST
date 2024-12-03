from elastic_st import ModelFeature, SpatialTranscriptomicsData, ElasticST
import numpy as np
import json

class NearestXCellFeature(ModelFeature):
    
    def __init__(self, bias, data, cell_type):
        """
        A bogus feature to find the distance to the nearest cell of each cell type from every focal cell of 'cell_type'.

        Parameters:
        bias (float): The bias term for the feature.
        data (SpatialTranscriptomicsData): The spatial transcriptomics data object.
        cell_type (str): The cell type to calculate the nearest cell distance for.
        """
        super().__init__(bias, data, cell_type)
        self.cell_type = cell_type
        self.data = data
        self.bias = bias
        
        self.matrix = self.compute_nearest_cell_distance()
    
    def compute_nearest_cell_distance(self) -> np.array:
        #For every cell, find the nearest neighboring cell of every cell type

        #Get the positions of the cells of interest
        focal_cells = self.data.get_cell_type_indices(self.cell_type)
        num_cells = len(focal_cells)
        num_cell_types = len(self.data.cell_types)
        distances = np.zeros((num_cells, num_cell_types))

        #Get the positions of all cells
        positions = self.data.P

        #Very inefficient, but it works for the purpose of the example
        for i, focal_cell in enumerate(focal_cells):
            focal_position = positions[focal_cell]
            for j, cell in enumerate(positions):
                if j == focal_cell:
                    continue
                distance = np.linalg.norm(cell - focal_position)
                cell_type = self.data.T[j]
                if distance < distances[i, cell_type]:
                    distances[i, cell_type] = distance
            
        return distances

    #All ModelFeatures must implement two methods 'get_feature' and 'get_feature_names' to be accessible to the ElasticST model.

    def get_feature(self, **kwargs) -> np.array:
        return self.matrix

    def get_feature_names(self, **kwargs) -> list:
        return [f"NearestXCell_{self.data.cell_types[i]}" for i in range(len(self.data.cell_types))]
        

if __name__ == "__main__":
    G = np.load('data/G.npy')
    P = np.load('data/P.npy')
    T = np.load('data/T.npy')

    annotations = json.load(open('data/annotations.json'))
    cell_types = annotations['cell_types']
    gene_names = annotations['gene_names']

    data = SpatialTranscriptomicsData(G, P, T, gene_names, cell_types)

    #Initialize the feature
    nearest_cell_feature = NearestXCellFeature(bias=5, data=data, cell_type='B-cell')

    #Filter out 'unimportant' genes
    data.variance_filter(threshold=0.2)

    #Train and save the model
    model = ElasticST(data, [nearest_cell_feature], cell_type='B-cell', alpha=0.05, l1_ratio=0.5, subsample_to=5000)
    coeffs = model.fit(n_jobs=-1)
    np.savez_compressed('coefficients.npy', **coeffs)