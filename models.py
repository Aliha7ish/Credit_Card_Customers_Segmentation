from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hdbscan.validity import validity_index


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

def dbcv_heatmap_best_params(model_func, param_grid, X, x_param_name, y_param_name=None, cmap="viridis"):
    best_score = -np.inf
    best_params = {}

    # Single parameter case
    if y_param_name is None:
        scores = []
        x_vals = param_grid[x_param_name]
        
        for x_val in x_vals:
            _, labels = model_func(X, **{x_param_name: x_val})
            
            # Need at least 2 clusters (excluding noise)
            if len(set(labels)) > 1 and len(set(labels)) - (1 if -1 in labels else 0) > 1:
                score = validity_index(X, labels)
            else:
                score = np.nan
            
            scores.append(score)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_params = {x_param_name: x_val}

        # Plot
        plt.figure(figsize=(8,4))
        plt.plot(x_vals, scores, marker='o')
        plt.title(f"DBCV Score vs {x_param_name}")
        plt.xlabel(x_param_name)
        plt.ylabel("DBCV Score")
        plt.grid(True)
        plt.show()

    # Two-parameter case
    else:
        y_vals = param_grid[y_param_name]
        x_vals = param_grid[x_param_name]
        dbcv_matrix = np.zeros((len(y_vals), len(x_vals)))

        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                _, labels = model_func(X, **{x_param_name: x_val, y_param_name: y_val})

                if len(set(labels)) > 1 and len(set(labels)) - (1 if -1 in labels else 0) > 1:
                    score = validity_index(X, labels)
                else:
                    score = np.nan

                dbcv_matrix[i, j] = score

                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_params = {x_param_name: x_val, y_param_name: y_val}

        # Heatmap
        plt.figure(figsize=(10,6))
        sns.heatmap(
            dbcv_matrix,
            xticklabels=x_vals,
            yticklabels=y_vals,
            annot=True,
            fmt=".3f",
            cmap=cmap
        )
        plt.title(f"DBCV Heatmap: {x_param_name} vs {y_param_name}")
        plt.xlabel(x_param_name)
        plt.ylabel(y_param_name)
        plt.show()

    print(f"Best DBCV Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    return best_params, best_score
