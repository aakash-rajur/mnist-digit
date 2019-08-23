import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats

from internal.feature_analysis.index_character_epochs import flatten_character_index


def reduce_dimensions(
        dimensions: int,
        coordinates: pd.DataFrame,
        outlier_threshold: int = 0
) -> pd.DataFrame:
    pca = PCA(n_components=dimensions)
    pca.fit(coordinates)
    transformed = pca.transform(coordinates)
    reduced = pd.DataFrame(transformed)
    reduced.index = coordinates.index

    return reduced
