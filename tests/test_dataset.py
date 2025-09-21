from clusterviz import datasets


def test_load_iris():
    X, y, features = datasets.load_iris()
    assert X.shape[0] == 150
    assert X.shape[1] == 4
    assert y.shape[0] == 150
    assert len(features) == 4

def test_make_blobs_dataset():
    X, y, features = datasets.make_blobs_dataset(n_samples=100, centers=3, 
n_features=2)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(features) == 2
