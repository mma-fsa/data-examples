import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_group_size_surface(
    df,
    x_col,
    y_col="log_AE_centered",
    x_label=None,
    title=None,
    n_grid=250,
    frac=0.35,
):
    """Create a 3D Plotly surface from loess-style smooths by group-size bin."""
    if x_label is None:
        x_label = x_col
    if title is None:
        title = f"Centered log(A/E) surface by {x_label} and group-size bin"

    # Use the categorical bin labels in order.
    if hasattr(df["GROUP_SIZE_BIN"], "cat"):
        bin_labels = list(df["GROUP_SIZE_BIN"].cat.categories)
    else:
        bin_labels = sorted(df["GROUP_SIZE_BIN"].dropna().unique().tolist())

    x_vals = np.linspace(df[x_col].min(), df[x_col].max(), n_grid)
    surface_matrix = np.full((len(bin_labels), len(x_vals)), np.nan)

    for i, bin_label in enumerate(bin_labels):
        bin_df = df[df["GROUP_SIZE_BIN"] == bin_label].copy()
        if bin_df.empty:
            continue

        x = bin_df[x_col].to_numpy(dtype=float)
        y = bin_df[y_col].to_numpy(dtype=float)
        fit = lowess(y, x, frac=frac, it=3, return_sorted=False)

        order = np.argsort(x)
        x_sorted = x[order]
        y_smooth = fit[order]
        surface_matrix[i, :] = np.interp(x_vals, x_sorted, y_smooth)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_vals,
                y=np.arange(len(bin_labels)),
                z=surface_matrix,
                colorscale="Viridis",
                showscale=True,
                opacity=0.95,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis=dict(
                title="GROUP_SIZE_BIN",
                tickvals=np.arange(len(bin_labels)),
                ticktext=bin_labels,
            ),
            zaxis_title=y_col,
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))
    return fig
