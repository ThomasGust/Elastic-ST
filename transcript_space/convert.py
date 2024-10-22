import json
import numpy as np


if __name__ == "__main__":
    geneidx2gene = json.load(open("C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer\\geneidx2gene.json"))
    clust = np.load("C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer\\clust.npy")

    cell_types = list(np.unique(clust))
    print(cell_types)

    gene_names = list(geneidx2gene.values())

    metadata = {'gene_names': gene_names, 'cell_types': cell_types}
    #Save metadata to a json file
    with open("C:\\Users\\Thoma\\Documents\\GitHub\\TranscriptSpace\\data\\colon_cancer\\metadata.json", 'w') as f:
        json.dump(metadata, f)
