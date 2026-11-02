import numpy as np

import elastic_st as est
from elastic_st.weighted_elastic_net import WeightedElasticNet


def test_weighted_elastic_net_recovers_sparse_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 10))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 4] + 0.1 * rng.standard_normal(500)

    model = WeightedElasticNet(alpha=0.01, l1_ratio=0.5, weights=np.ones(10))
    model.fit(X, y)
    beta = model.get_coefficients()

    assert abs(beta[0] - 3.0) < 0.1
    assert abs(beta[4] + 2.0) < 0.1
    assert np.all(np.abs(np.delete(beta, [0, 4])) < 0.05)


def test_feature_weights_change_the_penalty():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 10))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 4] + 0.1 * rng.standard_normal(500)

    #A large weight on feature 0 penalizes it heavily enough to remove it from the model.
    weights = np.ones(10)
    weights[0] = 1000.0
    model = WeightedElasticNet(alpha=0.01, l1_ratio=0.5, weights=weights)
    model.fit(X, y)
    beta = model.get_coefficients()

    assert beta[0] == 0.0
    assert abs(beta[4] + 2.0) < 0.1


def make_data(n_cells=200, n_genes=8, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.poisson(3.0, size=(n_cells, n_genes)).astype(float)
    P = rng.random((n_cells, 2))
    T = rng.integers(0, 2, size=n_cells)
    gene_names = [f"gene{i}" for i in range(n_genes)]
    return est.SpatialTranscriptomicsData(G, P, T, gene_names, ["A", "B"])


def test_string_cell_types_match_integer_cell_types():
    rng = np.random.default_rng(0)
    T = rng.integers(0, 2, size=200)
    G = rng.poisson(3.0, size=(200, 8)).astype(float)
    P = rng.random((200, 2))
    gene_names = [f"gene{i}" for i in range(8)]

    from_ints = est.SpatialTranscriptomicsData(G, P, T, gene_names, ["A", "B"])
    from_strings = est.SpatialTranscriptomicsData(G, P, np.array(["A", "B"])[T], gene_names, ["A", "B"])

    assert np.array_equal(from_strings.get_cell_type_indices("A"), from_ints.get_cell_type_indices("A"))


def test_elastic_st_fit_with_spatial_feature():
    data = make_data()
    feature = est.CellTypeAbundanceFeature(bias=5, data=data, cell_type="A", radius=0.2)
    model = est.ElasticST(data, [feature], cell_type="A", alpha=0.05, l1_ratio=0.5)
    record = model.fit(n_jobs=1, verbose=False)

    #One coefficient vector per gene, over all genes plus one abundance feature per cell type.
    n_features = 8 + 2
    coefficients = np.array(record["coefficients"])
    assert coefficients.shape == (8, n_features)
    assert np.all(np.isfinite(coefficients))
    assert len(record["feature_names"]) == n_features
