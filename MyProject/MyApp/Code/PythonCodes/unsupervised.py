import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import warnings

def unsupervised_learning(data):
    df = data

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    # Models to train
    models = {
        'KMeans': KMeans(n_clusters=len(set(y)), random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Hierarchical': AgglomerativeClustering(n_clusters=len(set(y))),
        'GMM': GaussianMixture(n_components=len(set(y)), random_state=42),
        'MeanShift': MeanShift()
    }

    results = {}

    warnings.filterwarnings("ignore", category=UserWarning)

    for name, model in models.items():
        try:
            if name == 'GMM':
                model.fit(X_processed)
                labels = model.predict(X_processed)
            else:
                labels = model.fit_predict(X_processed)

            # External clustering metrics
            ari = round(adjusted_rand_score(y, labels), 4)
            nmi = round(normalized_mutual_info_score(y, labels), 4)
            homo = round(homogeneity_score(y, labels), 4)

            # Internal clustering metrics
            if len(set(labels)) > 1 and len(set(labels)) < len(X_processed):
                sil = round(silhouette_score(X_processed, labels), 4)
                db = round(davies_bouldin_score(X_processed, labels), 4)
                ch = round(calinski_harabasz_score(X_processed, labels), 4)
            else:
                sil, db, ch = None, None, None

            results[name] = {
                "Adjusted Rand Index": ari,
                "Normalized Mutual Info": nmi,
                "Homogeneity Score": homo,
                "Silhouette Score": sil if sil is not None else "N/A",
                "Davies-Bouldin Index": db if db is not None else "N/A",
                "Calinski-Harabasz Index": ch if ch is not None else "N/A"
            }

        except Exception as e:
            results[name] = {"Error": str(e)}

    print(results)
    return results
