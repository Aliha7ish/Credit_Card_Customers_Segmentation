from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN

def train_fit_dbscan_model(X, eps=0.3, min_samples=10):
    db_model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db_model.labels_

    return db_model, labels

def train_fit_optics_model(X, min_samples=10):
    opt_model = OPTICS(min_samples=min_samples).fit(X)
    labels = opt_model.labels_

    return opt_model, labels

def train_fit_hdbscan_model(X, min_cluster_size=10, min_samples=10):
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(X)
    labels = hdbscan_model.labels_

    return hdbscan_model, labels