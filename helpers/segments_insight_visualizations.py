import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_group_feature_grid(
    df,
    group_col,
    features=None,
    agg_func="mean",
    n_cols=3,
    figsize_per_row=5
):
    """
    Generic grid plot to compare selected features across groups.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing data
        group_col (str): Column used for grouping (e.g., 'Cluster')
        features (list): List of feature names to plot (default: all numeric except group_col)
        agg_func (str): Aggregation function ('mean' or 'median')
        n_cols (int): Number of columns in grid
        figsize_per_row (int): Height per row
    """
    
    if features is None:
        features = df.select_dtypes(include="number").columns.drop(group_col)
    
    n_features = len(features)
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, figsize_per_row*n_rows))
    
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.barplot(
            data=df,
            x=group_col,
            y=feature,
            estimator=agg_func,
            ax=axes[i]
        )
        axes[i].set_title(f"{agg_func.capitalize()} {feature} by {group_col}")
        axes[i].set_xlabel(group_col)
        axes[i].set_ylabel(agg_func.capitalize())
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
