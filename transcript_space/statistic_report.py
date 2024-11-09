from transcript_space import SpatialStatistics, SpatialTranscriptomicsData
import numpy as np
import json

if __name__ == "__main__":
    G = np.load('data\\colon_cancer\\colon_cancer_G.npy')
    P = np.load('data\\colon_cancer\\colon_cancer_P.npy')
    T = np.load('data\\colon_cancer\\colon_cancer_T.npy')
    annotations = json.loads(open('data\\colon_cancer\\colon_cancer_annotation.json').read())

    st = SpatialTranscriptomicsData(G, P, T, annotations)
    statistics = SpatialStatistics(st)

    report = statistics.full_report(cell_type='epithelial.cancer.subtype_1', distance_threshold=0.1, distances=np.linspace(0, 0.5, 2))
    np.savez('statistics.npz', **report)