import pandas as pd
from sklearn.cluster import KMeans


def calculate_k_means_clusters(df: pd.DataFrame, cluster_size: int, x_key: str, y_key: str) -> pd.DataFrame:
    k_means = KMeans(n_clusters=cluster_size)

    k_means.fit(df)

    clusters = list(
        map(
            lambda each: ({x_key: each[0], y_key: each[1]}),
            k_means.cluster_centers_
        )
    )

    return pd.DataFrame(clusters)
