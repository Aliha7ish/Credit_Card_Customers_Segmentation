from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from .visualizations import *

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

def silhouette_heatmap_best_params(model_func, param_grid, X, x_param_name, y_param_name=None, cmap="viridis"):
    best_score = -np.inf
    best_params = {}

    # Single parameter case
    if y_param_name is None:
        scores = []
        x_vals = param_grid[x_param_name]
        for x_val in x_vals:
            _, labels = model_func(X, **{x_param_name: x_val})
            mask = labels != -1
            if len(np.unique(labels[mask])) > 1:
                score = silhouette_score(X[mask], labels[mask])
            else:
                score = np.nan
            scores.append(score)
            
            # Track best
            if score > best_score:
                best_score = score
                best_params = {x_param_name: x_val}
        
        # Plot
        plt.figure(figsize=(8,4))
        plt.plot(x_vals, scores, marker='o')
        plt.title(f"Silhouette Score vs {x_param_name}")
        plt.xlabel(x_param_name)
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.show()
    
    # Two-parameter case
    else:
        y_vals = param_grid[y_param_name]
        x_vals = param_grid[x_param_name]
        silhouette_matrix = np.zeros((len(y_vals), len(x_vals)))
        
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                _, labels = model_func(X, **{x_param_name: x_val, y_param_name: y_val})
                mask = labels != -1
                if len(np.unique(labels[mask])) > 1:
                    score = silhouette_score(X[mask], labels[mask])
                else:
                    score = np.nan
                silhouette_matrix[i, j] = score
                
                # Track best
                if score > best_score:
                    best_score = score
                    best_params = {x_param_name: x_val, y_param_name: y_val}
        
        # Plot heatmap
        plt.figure(figsize=(10,6))
        sns.heatmap(
            silhouette_matrix,
            xticklabels=x_vals,
            yticklabels=y_vals,
            annot=True,
            fmt=".2f",
            cmap=cmap
        )
        plt.title(f"Silhouette Score Heatmap: {x_param_name} vs {y_param_name}")
        plt.xlabel(x_param_name)
        plt.ylabel(y_param_name)
        plt.show()
    
    print(f"Best Silhouette Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    return best_params, best_score

def tsne_2d_plot(data, labels=None, perplexity=50, init_state="pca", random_state=42, title="2D Scatter Plot"):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init=init_state,
        random_state=random_state
    )
    
    tsne_result = tsne.fit_transform(data)
    
    scatter_plot2D(tsne_result, color=labels, title=title)
    
    return tsne_result

def tsne_grid_plot(scaled_data, perplexities, n_rows, n_cols, init_state="pca", random_state=42, main_title=None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle case when axes is 1D
    if n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for row_idx, (name, data) in enumerate(scaled_data.items()):
        for col_idx, perplex in enumerate(perplexities):
            tsne = TSNE(
                n_components=2,
                perplexity=perplex,
                learning_rate='auto',
                init=init_state,
                random_state=random_state
            )
            data_emb = tsne.fit_transform(data)
            
            ax = axes[row_idx][col_idx]
            ax.scatter(data_emb[:, 0], data_emb[:, 1], alpha=0.6)
            ax.set_title(f"{name} | perplexity={perplex}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True)
    
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.show()
