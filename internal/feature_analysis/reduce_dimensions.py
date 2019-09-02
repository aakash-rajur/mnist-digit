import pandas as pd
from sklearn.decomposition import PCA


def reduce_dimensions(dimensions: int, coordinates: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=dimensions)
    pca.fit(coordinates)
    transformed = pca.transform(coordinates)
    reduced = pd.DataFrame(transformed)
    reduced.index = coordinates.index

    return reduced
