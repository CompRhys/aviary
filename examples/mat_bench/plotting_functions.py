import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from matbench.constants import CLF_KEY, REG_KEY
from matbench.metadata import mbv01_metadata
from plotly.graph_objs._figure import Figure

__author__ = "Janosh Riebesell"
__date__ = "2022-04-25"


def scale_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the errors in a Matbench dataframe with columns for tasks and rows for
    different models. Missing entries are fine.

    Returns:
        pd.DataFrame: Dataframe with scaled errors
    """

    # make scaled data for heatmap coloring
    # scale regression problems by mad/mae
    def scale_regr_task(series, mad):
        mask = series > 0.0
        mask_iix = np.where(mask)
        series.iloc[mask_iix] = series.iloc[mask_iix] / mad
        series.loc[~mask] = np.nan
        return series

    def scale_clf_task(series, mad):
        mask = series > 0.0
        mask_iix = np.where(mask)
        series.iloc[mask_iix] = 1 - (series.iloc[mask_iix] - 0.5) / 0.5
        series.loc[~mask] = np.nan
        return series

    scaled_df = df.copy(deep=True)
    for task in scaled_df:
        task_type = mbv01_metadata[task].task_type
        assert task_type in [REG_KEY, CLF_KEY], f"Unknown {task_type = }"
        scaler = scale_clf_task if task_type == CLF_KEY else scale_regr_task

        scaled_df[task] = scaler(scaled_df[task], mbv01_metadata[task].mad)

    return scaled_df


x_labels = {
    "matbench_steels": "σᵧ Steel alloys",
    "matbench_jdft2d": "Eˣ 2D Materials",
    "matbench_phonons": "ωᵐᵃˣ Phonons",
    "matbench_dielectric": "𝑛",
    "matbench_expt_gap": "Eᵍ Experimental",
    "matbench_expt_is_metal": "Expt. Metallicity Classification",
    "matbench_glass": "Metallic Glass Classification",
    "matbench_log_kvrh": "log₁₀Kᵛʳʰ",
    "matbench_log_gvrh": "log₁₀Gᵛʳʰ",
    "matbench_perovskites": "Eᶠ Perovskites, DFT",
    "matbench_mp_gap": "Eᵍ DFT",
    "matbench_mp_is_metal": "Metallicity DFT",
    "matbench_mp_e_form": "Eᶠ DFT",
}


def plot_leaderboard(df: pd.DataFrame) -> Figure:
    """Generate the Matbench scaled errors graph seen on
    https://matbench.materialsproject.org. Adapted from https://bit.ly/38fDdgt.

        Args:
            df (pd.DataFrame): Dataframe with columns for matbench tasks and rows for different
                models. Missing entries are fine.

        Returns:
            Figure: Plotly graph objects Figure instance
    """
    # deep copy df so we don't modify the original
    df = df.copy(deep=True)

    best_values = df.min(axis=1)
    best_algos = df.idxmin(axis=1)

    fig = px.scatter(df, log_y=True, labels=x_labels)

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

    return fig


def scaled_error_heatmap(df: pd.DataFrame) -> Figure:
    """Create a heatmap of the scaled errors with a column for mean scaled error added
    on the right.

    Args:
        df (pd.DataFrame): Dataframe with columns for matbench tasks and rows for different
                models. Missing entries are fine.

    Returns:
        Figure: Plotly graph objects Figure instance
    """
    # deep copy df so we don't modify the original
    df = df.copy(deep=True)

    df["mean scaled error"] = df.mean(1)
    df = df.sort_values(by="mean scaled error")

    fig = px.imshow(df, width=1000, height=600, text_auto=".2f", aspect="auto")

    fig.update_layout(
        title=dict(text="<b>Matbench Scaled Errors</b>", x=0.5, font_size=20),
        font_size=14,
        coloraxis_colorbar_x=1.05,
    )

    return fig
