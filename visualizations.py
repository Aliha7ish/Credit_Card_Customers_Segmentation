import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.io as pio

# Use browser renderer if nbformat issues occur
# pio.renderers.default = "browser"

def plot_feature_relation(df, feature_x, feature_y):
    """
    Plot the relationship between two numeric features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    feature_x : str
        First feature name
    feature_y : str
        Second feature name
    """

    x = df[feature_x]
    y = df[feature_y]

    # Correlation
    corr = np.corrcoef(x, y)[0, 1]

    plt.figure(figsize=(6, 5))
    sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6})
    plt.title(f"{feature_x} vs {feature_y}\nCorrelation = {corr:.3f}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(df, feature, bins=20, kde=True):
    """
    Plot the distribution of a numeric feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    feature : str
        Feature name
    """

    plt.figure(figsize=(6, 4))
    sns.histplot(df[feature], kde=True, bins=bins)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, features):
    """
    Plot a correlation heatmap for the specified features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    features : list of str
        List of feature names to include in the heatmap
    """

    corr_matrix = df[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, feature):
    """
    Plot a boxplot for a numeric feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    feature : str
        Feature name
    """

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()

def plot_scatter(df, feature_x, feature_y):
    """
    Plot a scatter plot between two numeric features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    feature_x : str
        First feature name
    feature_y : str
        Second feature name
    """

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=df[feature_x], y=df[feature_y], alpha=0.6)
    plt.title(f"{feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.tight_layout()
    plt.show()

def plot_scatter_with_hue(df, feature_x, feature_y, hue):
    """
    Plot a scatter plot between two numeric features with a hue.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset
    feature_x : str
        First feature name
    feature_y : str
        Second feature name
    hue : str
        Categorical feature name for coloring the points
    """

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df[hue], alpha=0.6)
    plt.title(f"{feature_x} vs {feature_y} colored by {hue}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.show()

def screen_plot(pca_data, scaled_data=""):
    explained_variance = pca_data.explained_variance_
    total_variance = np.sum(explained_variance)

    explained_variance_ratio = explained_variance / total_variance * 100  # percentage

    plt.ylabel("Eigenvalues | Variance Explained %")
    plt.xlabel("# of Features")
    plt.title(f"PCA Eigenvalues of {scaled_data} Data")
    plt.ylim(0, max(explained_variance_ratio) + 5)
    plt.plot(explained_variance_ratio, marker='o')
    plt.grid(True)
    plt.show()

def scatter_plot2D(data, color=None, title="2D Scatter Plot"):
    plt.figure(figsize=(8,6))
    if color is not None:
        scatter = plt.scatter(data[:,0], data[:,1], c=color, cmap='tab10', alpha=0.6)
        plt.legend(*scatter.legend_elements(), title="Clusters")
    else:
        plt.scatter(data[:,0], data[:,1], alpha=0.6)
        
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()


def scatter_plot3D(data, color=None, title="3D Scatter Plot"):
    x, y, z = data[:, :3].T
    fig = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        color=color,
        title=title,
        labels={"x": "PC1", "y": "PC2", "z": "PC3"},
        color_discrete_sequence=px.colors.qualitative.Bold if color is not None else None
    )
    
    fig.show()

