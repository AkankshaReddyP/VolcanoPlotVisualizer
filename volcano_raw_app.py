import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash

# --- Load Excel ---
file_path = "Raw_Normalized_FC_Cohensd_MW_p.xlsx"

# Load raw
df_raw = pd.read_excel(file_path, sheet_name="Raw")
df_raw["neg_log10_pval"] = -np.log10(df_raw["MW_pvalue_ALN_vs_iSLE"])
df_raw["log2FC"] = np.log2(df_raw["FC_LN_Vs_iSLE"].replace(0, np.nan))
df_raw["log2CohenD"] = np.sign(df_raw["Cohen's_d_LN_Vs_iSLE"]) * np.log2(1 + abs(df_raw["Cohen's_d_LN_Vs_iSLE"]))

# Load Cr_Norm
df_norm = pd.read_excel(file_path, sheet_name="Cr_Norm")
df_norm["neg_log10_pval"] = -np.log10(df_norm["MW_pvalue_ALN_vs_iSLE"])
df_norm["log2FC"] = np.log2(df_norm["FC_LN_Vs_iSLE"].replace(0, np.nan))
df_norm["log2CohenD"] = np.sign(df_norm["Cohen's_d_LN_Vs_iSLE"]) * np.log2(1 + abs(df_norm["Cohen's_d_LN_Vs_iSLE"]))

# --- Dash app ---
app = Dash(__name__)


def slider_block(prefix, label_fc, label_cohen=None, default_fc=1.0, default_p=0.05,
                 max_fc=10, combined=False):
    """Reusable block for sliders under each plot"""
    children = [
        html.Label(label_fc),
        dcc.Slider(
            id=f"{prefix}-fc-slider",
            min=0, max=max_fc, step=1, value=default_fc,
            marks={i: {"label": str(i), "style": {"font-size": "10px"}}
                   for i in range(0, max_fc + 1)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ]

    if combined:  # add Cohen’s d slider too
        children.extend([
            html.Br(),
            html.Label(label_cohen),
            dcc.Slider(
                id=f"{prefix}-cohen-slider",
                min=0, max=max_fc, step=1, value=default_fc,
                marks={i: {"label": str(i), "style": {"font-size": "10px"}}
                       for i in range(0, max_fc + 1)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ])

    children.extend([
        html.Br(),
        html.Label("log(p-value) cutoff:"),
        html.Div([
            html.Div([
                dcc.Slider(
                    id=f"{prefix}-p-slider",
                    min=-np.log10(1.0), max=-np.log10(1e-6), step=0.01,
                    value=-np.log10(default_p),
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={"width": "70%", "display": "inline-block"}),

            html.Div([
                html.Label("p-value:"),
                dcc.Input(
                    id=f"{prefix}-p-input", type="number",
                    value=default_p,
                    min=1e-10, max=1.0,
                    style={"width": "100%"}
                )
            ], style={"width": "25%", "display": "inline-block",
                      "margin-left": "10px"})
        ], style={"display": "flex", "align-items": "center"}),

        html.Div(id=f"{prefix}-p-output",
                 style={"margin-top": "5px", "font-weight": "bold"})
    ])

    if combined:
        children.extend([
            html.Br(),
            html.Label("X-axis choice:"),
            dcc.Dropdown(
                id=f"{prefix}-xaxis",
                options=[
                    {"label": "log2(FC)", "value": "log2FC"},
                    {"label": "log2(Cohen’s d)", "value": "log2CohenD"}
                ],
                value="log2FC",
                clearable=False,
                style={"width": "60%"}
            )
        ])

    return html.Div(children)


def make_volcano(df, x_col, fc_cutoff, p_cutoff_log, raw_p_cutoff,
                 title, xlabel, ylabel):
    df = df.copy()
    df["Category"] = "Not Significant"
    df.loc[(df[x_col] >= fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log),
           "Category"] = "Up"
    df.loc[(df[x_col] <= -fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log),
           "Category"] = "Down"

    up_count = (df["Category"] == "Up").sum()
    down_count = (df["Category"] == "Down").sum()
    notsig_count = (df["Category"] == "Not Significant").sum()
    color_map = {
        "Up": f"Up (n={up_count})",
        "Down": f"Down (n={down_count})",
        "Not Significant": f"Not Significant (n={notsig_count})"
    }

    fig = px.scatter(
        df, x=x_col, y="neg_log10_pval",
        color="Category",
        color_discrete_map={"Up": "red", "Down": "blue",
                            "Not Significant": "grey"},
        hover_data={"Compounds": True, "Index": True,
                    "FC_LN_Vs_iSLE": True, "Cohen's_d_LN_Vs_iSLE": True,
                    "MW_pvalue_ALN_vs_iSLE": True},
        title=title,
        labels={x_col: xlabel, "neg_log10_pval": ylabel}
    )
    for trace in fig.data:
        trace.name = color_map.get(trace.name, trace.name)

    fig.add_hline(y=p_cutoff_log, line_dash="dash", line_color="black")
    fig.add_vline(x=fc_cutoff, line_dash="dash", line_color="black")
    fig.add_vline(x=-fc_cutoff, line_dash="dash", line_color="black")
    return fig


def make_combined(df, fc_cutoff, cohen_cutoff, p_cutoff_log, raw_p_cutoff,
                  x_col, label):
    df = df.copy()
    df["Category"] = "Not Significant"
    # Require BOTH FC and Cohen’s d cutoffs
    sig_mask = (
        (df["neg_log10_pval"] >= p_cutoff_log) &
        (df["log2FC"].abs() >= fc_cutoff) &
        (df["log2CohenD"].abs() >= cohen_cutoff)
    )
    df.loc[sig_mask & (df["log2FC"] > 0), "Category"] = "Up"
    df.loc[sig_mask & (df["log2FC"] < 0), "Category"] = "Down"

    up_count = (df["Category"] == "Up").sum()
    down_count = (df["Category"] == "Down").sum()
    notsig_count = (df["Category"] == "Not Significant").sum()
    color_map = {
        "Up": f"Up (n={up_count})",
        "Down": f"Down (n={down_count})",
        "Not Significant": f"Not Significant (n={notsig_count})"
    }

    fig = px.scatter(
        df, x=x_col, y="neg_log10_pval",
        color="Category",
        color_discrete_map={"Up": "red", "Down": "blue",
                            "Not Significant": "grey"},
        hover_data={"Compounds": True, "Index": True,
                    "FC_LN_Vs_iSLE": True, "Cohen's_d_LN_Vs_iSLE": True,
                    "MW_pvalue_ALN_vs_iSLE": True},
        title=f"{label} Combined Volcano Plot ({x_col} on x-axis, requires FC + Cohen’s d cutoffs)",
        labels={x_col: x_col, "neg_log10_pval": "-log10(p-value)"}
    )
    for trace in fig.data:
        trace.name = color_map.get(trace.name, trace.name)

    fig.add_hline(y=p_cutoff_log, line_dash="dash", line_color="black")
    fig.add_vline(x=fc_cutoff, line_dash="solid", line_color="red")
    fig.add_vline(x=-fc_cutoff, line_dash="solid", line_color="red")
    fig.add_vline(x=cohen_cutoff, line_dash="dot", line_color="blue")
    fig.add_vline(x=-cohen_cutoff, line_dash="dot", line_color="blue")
    return fig


# --- Layout ---
app.layout = html.Div([
    html.H2("Volcano Plots of Metabolites (ALN vs iSLE)", style={"text-align": "center"}),

    # RAW
    html.H3("Raw Data"),
    html.Div([
        html.Div([
            dcc.Graph(id="fc-plot-raw"),
            slider_block("fc-raw", "log(FC) cutoff", max_fc=10)
        ], style={"width": "48%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(id="cohen-plot-raw"),
            slider_block("cohen-raw", "log(Cohen’s d) cutoff", max_fc=10)
        ], style={"width": "48%", "display": "inline-block", "margin-left": "2%"})
    ]),
    html.Div([
        dcc.Graph(id="combined-plot-raw"),
        slider_block("combined-raw", "log(FC) cutoff", label_cohen="log(Cohen’s d) cutoff",
                     max_fc=10, combined=True)
    ], style={"margin-top": "30px"}),

    # Cr_Norm
    html.H3("Cr_Norm Data"),
    html.Div([
        html.Div([
            dcc.Graph(id="fc-plot-norm"),
            slider_block("fc-norm", "log(FC) cutoff", max_fc=10)
        ], style={"width": "48%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(id="cohen-plot-norm"),
            slider_block("cohen-norm", "log(Cohen’s d) cutoff", max_fc=10)
        ], style={"width": "48%", "display": "inline-block", "margin-left": "2%"})
    ]),
    html.Div([
        dcc.Graph(id="combined-plot-norm"),
        slider_block("combined-norm", "log(FC) cutoff", label_cohen="log(Cohen’s d) cutoff",
                     max_fc=10, combined=True)
    ], style={"margin-top": "30px"})
])


# --- Callbacks for Raw ---
@app.callback(
    [Output("fc-plot-raw", "figure"),
     Output("fc-raw-p-output", "children"),
     Output("fc-raw-p-input", "value"),
     Output("fc-raw-p-slider", "value")],
    Input("fc-raw-fc-slider", "value"),
    Input("fc-raw-p-slider", "value"),
    Input("fc-raw-p-input", "value")
)
def update_fc_raw(fc_cutoff, p_cutoff_log, p_input_val):
    raw_p_cutoff = float(p_input_val) if dash.callback_context.triggered_id == "fc-raw-p-input" else 10**(-p_cutoff_log)
    raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
    p_cutoff_log = -np.log10(raw_p_cutoff)
    fig = make_volcano(df_raw, "log2FC", fc_cutoff, p_cutoff_log, raw_p_cutoff,
                       "Raw log2(FC) vs -log10(p-value)", "log2(FC)", "-log10(p-value)")
    return fig, f"p ≤ {raw_p_cutoff:.5g} (-log10={p_cutoff_log:.2f})", raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("cohen-plot-raw", "figure"),
     Output("cohen-raw-p-output", "children"),
     Output("cohen-raw-p-input", "value"),
     Output("cohen-raw-p-slider", "value")],
    Input("cohen-raw-fc-slider", "value"),
    Input("cohen-raw-p-slider", "value"),
    Input("cohen-raw-p-input", "value")
)
def update_cohen_raw(cohen_cutoff, p_cutoff_log, p_input_val):
    raw_p_cutoff = float(p_input_val) if dash.callback_context.triggered_id == "cohen-raw-p-input" else 10**(-p_cutoff_log)
    raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
    p_cutoff_log = -np.log10(raw_p_cutoff)
    fig = make_volcano(df_raw, "log2CohenD", cohen_cutoff, p_cutoff_log, raw_p_cutoff,
                       "Raw log2(Cohen’s d) vs -log10(p-value)", "log2(Cohen’s d)", "-log10(p-value)")
    return fig, f"p ≤ {raw_p_cutoff:.5g} (-log10={p_cutoff_log:.2f})", raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("combined-plot-raw", "figure"),
     Output("combined-raw-p-output", "children"),
     Output("combined-raw-p-input", "value"),
     Output("combined-raw-p-slider", "value")],
    Input("combined-raw-fc-slider", "value"),
    Input("combined-raw-cohen-slider", "value"),
    Input("combined-raw-p-slider", "value"),
    Input("combined-raw-p-input", "value"),
    Input("combined-raw-xaxis", "value")
)
def update_combined_raw(fc_cutoff, cohen_cutoff, p_cutoff_log, p_input_val, x_col):
    raw_p_cutoff = float(p_input_val) if dash.callback_context.triggered_id == "combined-raw-p-input" else 10**(-p_cutoff_log)
    raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
    p_cutoff_log = -np.log10(raw_p_cutoff)
    fig = make_combined(df_raw, fc_cutoff, cohen_cutoff, p_cutoff_log, raw_p_cutoff, x_col, "Raw")
    return fig, f"p ≤ {raw_p_cutoff:.5g} (-log10={p_cutoff_log:.2f})", raw_p_cutoff, p_cutoff_log


# --- Callbacks for Cr_Norm ---
@app.callback(
    [Output("fc-plot-norm", "figure"),
     Output("fc-norm-p-output", "children"),
     Output("fc-norm-p-input", "value"),
     Output("fc-norm-p-slider", "value")],
    Input("fc-norm-fc-slider", "value"),
    Input("fc-norm-p-slider", "value"),
    Input("fc-norm-p-input", "value")
)
def update_fc_norm(fc_cutoff, p_cutoff_log, p_input_val):
    raw_p_cutoff = float(p_input_val) if dash.callback_context.triggered_id == "fc-norm-p-input" else 10**(-p_cutoff_log)
    raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
    p_cutoff_log = -np.log10(raw_p_cutoff)
    fig = make_volcano(df_norm, "log2FC", fc_cutoff, p_cutoff_log, raw_p_cutoff,
                       "Cr_Norm log2(FC) vs -log10(p-value)", "log2(FC)", "-log10(p-value)")
    return fig, f"p ≤ {raw_p_cutoff:.5g} (-log10={p_cutoff_log:.2f})", raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("cohen-plot-norm", "figure"),
     Output("cohen-norm-p-output", "children"),
     Output("cohen-norm-p-input", "value"),
     Output("cohen-norm-p-slider", "value")],
    Input("cohen-norm-fc-slider", "value"),
    Input("cohen-norm-p-slider", "value"),
    Input("cohen-norm-p-input", "value")
)
def update_cohen_norm(cohen_cutoff, p_cutoff_log, p_input_val):
    raw_p_cutoff = float(p_input_val) if dash.callback_context.triggered_id == "cohen-norm-p-input" else 10**(-p_cutoff_log)
    raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
    p_cutoff_log = -np.log10(raw_p_cutoff)
    fig = make_volcano(df_norm, "log2CohenD", cohen_cutoff, p_cutoff_log, raw_p_cutoff,
                       "Cr_Norm log2(Cohen’s d) vs -log10(p-value)", "log2(Cohen’s d)", "-log10(p-value)")
    return fig, f"p ≤ {raw_p_cutoff:.5g} (-log10={p_cutoff_log:.2f})", raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("combined-plot-norm", "figure"),
     Output("combined-norm-p-output", "children"),
     Output("combined-norm-p-input", "value"),
     Output("combined-norm-p-slider", "value")],
    Input("combined-norm-fc-slider", "value"),
    Input("combined-norm-cohen-slider", "value"),
    Input("combined-norm-p-slider", "value"),
    Input("combined-norm-p-input", "value"),
    Input("combined-norm-xaxis", "value")
)
def update_combined_norm(fc_cutoff, cohen_cutoff, p_cutoff_log, p_input_val, x_col):
    raw_p_cutoff = float(p_input_val) if dash.callback_context.triggered_id == "combined-norm-p-input" else 10**(-p_cutoff_log
