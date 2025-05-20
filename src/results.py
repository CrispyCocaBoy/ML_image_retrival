import numpy as np
from scipy.spatial.distance import cdist

def compute_results(query_df, gallery_df, top_k=3, metric="euclidean"):
    """
    Restituisce lista di dizionari con formato:
    [
      {
        "filename": <query image path>,
        "gallery_images": [<gallery path 1>, <gallery path 2>, ...]
      },
      ...
    ]
    """

    query_embeddings = np.stack(query_df['embedding'].to_numpy())
    gallery_embeddings = np.stack(gallery_df['embedding'].to_numpy())

    distances = cdist(query_embeddings, gallery_embeddings, metric=metric)

    results = []

    for i, query_row in query_df.iterrows():
        query_path = query_row["filename"]  # full path o path relativo
        dist_row = distances[i]
        top_indices = np.argsort(dist_row)[:top_k]
        gallery_paths = gallery_df.iloc[top_indices]["filename"].tolist()

        results.append({
            "filename": query_path,
            "gallery_images": gallery_paths
        })

    return results
