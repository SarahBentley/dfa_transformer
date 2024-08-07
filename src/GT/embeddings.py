import plotly.graph_objects as go
import torch
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Dimensionality reduction
def reduce_dimensions(weights, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=3)
    elif method == 't-SNE':
        reducer = TSNE(n_components=3, perplexity=30, n_iter=300)
    else:
        raise ValueError("Method should be 'PCA' or 't-SNE'")
    reduced_weights = reducer.fit_transform(weights)
    return reduced_weights


def plot_embedding_weights(reduced_weights, labels, title='Embedding Weights Visualization'):
    """
    Create a 3D scatter plot of embedding weights using Plotly.

    Parameters:
    - reduced_weights (np.ndarray): The reduced embedding weights with shape (n_samples, 3).
    - labels (list of str): Labels for the points.
    - title (str): The title of the plot.

    Returns:
    - fig (plotly.graph_objects.Figure): The 3D scatter plot figure.
    """
    fig = go.Figure(data=go.Scatter3d(
        x=reduced_weights[:, 0],
        y=reduced_weights[:, 1],
        z=reduced_weights[:, 2],
        mode='markers+text',
        text=labels,
        marker=dict(size=5, opacity=0.8)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        template='plotly_dark'
    )
    
    return fig

def visualize_wte(model, dataloader, reduction_method='PCA'):
    model.eval()
    weights = model.transformer.wte.weight.detach().cpu().numpy()
    reduced_weights = reduce_dimensions(weights, method=reduction_method)
    vocab = dataloader.vocab
    fig = plot_embedding_weights(reduced_weights, vocab)
    return {'Embedding weights': fig}

def visualize_wpe(model, reduction_method='PCA'):
    model.eval()
    lengths = list(range(model.config.block_size))
    weights = model.transformer.wpe.weight.detach().cpu().numpy()
    reduced_weights = reduce_dimensions(weights, method=reduction_method)
    fig = plot_embedding_weights(reduced_weights, lengths)
    return {'Positional encoding weights': fig}