from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from matbench.constants import CLF_KEY, REG_KEY
from matbench.metadata import mbv01_metadata
from matbench.metadata import mbv01_metadata as matbench_metadata
from sklearn.metrics import accuracy_score, auc, roc_curve

if TYPE_CHECKING:
    import pandas as pd
    from plotly.graph_objs._figure import Figure

__author__ = "Janosh Riebesell"
__date__ = "2022-04-25"

pio.templates.default = "plotly_white"


def scale_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the errors in a Matbench dataframe for comparability in heatmaps.

    Args:
        df (pd.DataFrame): Dataframe with unscaled errors for matbench tasks across columns and
            models along rows. Missing entries are fine.

    Returns:
        pd.DataFrame: Dataframe with scaled errors
    """

    def scale_regr_task(series: pd.Series, mad: float) -> pd.Series:
        # scale regression problems by mad/mae
        mask = series > 0
        mask_iix = np.where(mask)
        series.iloc[mask_iix] = series.iloc[mask_iix] / mad
        series.loc[~mask] = np.nan
        return series

    def scale_clf_task(series: pd.Series) -> pd.Series:
        mask = series > 0
        mask_idx = np.where(mask)
        series.iloc[mask_idx] = 1 - (series.iloc[mask_idx] - 0.5) / 0.5
        series.loc[~mask] = np.nan
        return series

    scaled_df = df.copy(deep=True).T
    for dataset in scaled_df:
        task_type = mbv01_metadata[dataset].task_type
        if task_type not in [REG_KEY, CLF_KEY]:
            raise ValueError(f"Unknown {task_type = } for {dataset = }")
        if task_type == REG_KEY:
            task_mad = mbv01_metadata[dataset].mad
            scaled_df[dataset] = scale_regr_task(scaled_df[dataset], task_mad)
        elif task_type == CLF_KEY:
            scaled_df[dataset] = scale_clf_task(scaled_df[dataset])

    return scaled_df.T


dataset_labels = {
    "matbench_steels": "oᵧ Steel alloys",
    "matbench_jdft2d": "Eˣ 2D Materials",
    "matbench_phonons": "ωᵐᵃˣ Phonons",
    "matbench_dielectric": "n",
    "matbench_expt_gap": "Eᵍ Experimental",
    "matbench_expt_is_metal": "Expt. Metallicity Clf",
    "matbench_glass": "Metallic Glass Clf",
    "matbench_log_kvrh": "log₁₀Kᵛʳʰ",
    "matbench_log_gvrh": "log₁₀Gᵛʳʰ",
    "matbench_perovskites": "Eᶠ Perovskites, DFT",
    "matbench_mp_gap": "Eᵍ DFT",
    "matbench_mp_is_metal": "Metallicity DFT",
    "matbench_mp_e_form": "Eᶠ DFT",
}
dataset_sizes = {k: v["n_samples"] for k, v in matbench_metadata.items()}
dataset_labels_html = {
    k: f"<b>{v}</b>  {dataset_sizes[k]:,}" for k, v in dataset_labels.items()
}


def plot_leaderboard(
    df: pd.DataFrame, html_path: str | None = None, **kwargs: Any
) -> Figure:
    """Generate the Matbench scaled errors graph seen on
    https://matbench.materialsproject.org. Adapted from https://bit.ly/38fDdgt.

    Args:
        df (pd.DataFrame): Dataframe with columns for matbench tasks and rows for different
            models. Missing entries are fine.
        html_path (str): HTML file path where to save the plotly figure.
        **kwargs: Additional keyword arguments to pass to plotly.express.scatter.

    Returns:
        Figure: Plotly graph objects Figure instance
    """
    df = df.rename(dataset_labels_html)
    # deep copy df so we don't modify the original
    best_values = df.min(axis=1)
    best_algos = df.idxmin(axis=1)

    fig = px.scatter(df, log_y=True, labels=dataset_labels, **kwargs)

    fig.update_layout(
        title=dict(text="Scaled Errors", font_size=25, x=0.4),
        legend=dict(font_size=15, title=dict(font_size=15, text="Algorithm")),
        yaxis_title="Scaled MAE (regression) or <br> (1-ROCAUC)/0.5 (classification)",
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    # add scatter for the best algorithms on scaled error
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=best_values.index,
            y=best_values,
            marker_color="yellow",
            text=best_algos,
            visible="legendonly",
            name="Best algorithms",
        )
    )
    fig.update_traces(marker_size=10)
    fig.update_xaxes(linecolor="grey", gridcolor="grey")
    fig.update_yaxes(linecolor="grey", gridcolor="grey")

    if html_path:
        fig.write_html(html_path, include_plotlyjs="cdn")
        # change plot background to black since Matbench site uses dark mode
        with open(html_path, "r+") as file:
            html = file.read()
            file.seek(0)  # rewind file pointer to start of file
            html = html.replace(
                "</head>", "<style>body { background-color: black; }</style></head>"
            )
            file.write(html)
            file.truncate()

    return fig


def error_heatmap(
    df_err: pd.DataFrame, log: bool = False, prec: int = 3, **kwargs: Any
) -> Figure:
    """Create a heatmap of the errors with a column for mean error across all tasks
    added on the right. Title assumes the errors are scaled relative to dummy
    performance but works with unscaled errors too.

    Args:
        df_err (pd.DataFrame): Dataframe with columns for matbench tasks and rows for different
                models. Missing entries are fine.
        log (bool, optional): Whether to log10 the errors before plotting. Defaults to False.
        prec (int, optional): Number of decimal places to round the errors to. Defaults to 3.
        **kwargs: Additional keyword arguments to pass to plotly.express.imshow().

    Returns:
        Figure: Plotly graph objects Figure instance
    """
    # deep copy df so we don't modify the original

    df_err_scaled = scale_errors(df_err).T
    # rename column names for prettier axis ticks (must be after scale_errors() to have
    # correct dict keys)
    df_err_scaled = df_err_scaled.rename(columns=dataset_labels_html)

    if log:
        df_err_scaled = np.log10(df_err_scaled)

    # mean scaled error across all recorded tasks for each model
    df_err_scaled["mean scaled error"] = df_err_scaled.mean(axis=1)
    # mean scaled error across only those tasks recorded for all models
    # usually jdft2d, log_gvrh, dielectric, perovskites, mp_gap, mp_e_form
    df_err_scaled["dense scaled error"] = df_err_scaled.dropna(axis=1).mean(axis=1)

    df_err_scaled = df_err_scaled.sort_values(by="dense scaled error").round(prec)

    fig = px.imshow(df_err_scaled, text_auto=f".{prec}f", aspect="auto", **kwargs)

    fig.update_layout(
        title=dict(text="<b>Matbench Scaled Errors</b>", x=0.5, font_size=20),
        font_size=14,
        coloraxis_colorbar_x=1.05,
    )

    if log:
        ticks = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        vals = np.log10(ticks)
        fig.update_layout(coloraxis_colorbar=dict(tickvals=vals, ticktext=ticks))

    return fig


def annotate_fig(fig: Figure, **kwargs) -> None:
    """Add text to a plotly figure.

    Wrapper around fig.add_annotation() sensible defaults.
    """
    defaults = dict(
        xref="paper",
        yref="paper",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="gray",
        showarrow=False,
        borderpad=3,
    )
    fig.add_annotation({**defaults, **kwargs})


def plotly_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Figure:
    """Plot a ROC curve with accuracy and ROCAUC annotation.

    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted class probabilities.

    Returns:
        Figure: Plotly Figure instance.
    """
    false_pos_rate, true_pos_rate, _ = roc_curve(y_true, y_pred_proba)

    labels = dict(x="False Positive Rate", y="True Positive Rate")
    fig = px.area(x=false_pos_rate, y=true_pos_rate, labels=labels)

    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    ROCAUC = auc(false_pos_rate, true_pos_rate)
    accuracy = accuracy_score(y_true, y_pred_proba > 0.5)
    text = f"{accuracy=:.2f}<br>{ROCAUC=:.2f}"
    annotate_fig(
        fig,
        text=text,
        x=0.95,
        y=0.05,
        xanchor="right",
        yanchor="bottom",
        xref=None,
        yref=None,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    return fig
