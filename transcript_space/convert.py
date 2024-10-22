import json
import numpy as np


if __name__ == "__main__":
    clust = np.load("C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer\\clust.npy")
    
    str2int = {s:i for i, s in enumerate(np.unique(clust))}
    #Convert clust from numpy array of strings to numpy array of integers
    clust = np.array([str2int[c] for c in clust])
    #Save clust as colon_cancer_T.npy in the same directory
    np.save("C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer\\clust_T.npy", clust)