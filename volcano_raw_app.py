import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash

# --- Load Excel ---
file_path = "Raw_Normalized_FC_Cohensd_MW_p.xlsx"
df = pd.read_excel(file_path, sheet_name="Raw")

# Precompute transforms
df["neg_log10_pval"] = -np.log10(df["MW_pvalue_ALN_vs_iSLE"])
df["log2FC"] = np.log2(df["FC_LN_Vs_iSLE"].replace(0, np.nan))
df["log2CohenD"] = np.sign(df["Cohen's_d_LN_Vs_iSLE"]) * np.log2(1 + abs(df["Cohen's_d_LN_Vs_iSLE"]))

# --- Dash app ---
app = Dash(__name__)

def slider_block(prefix, label_fc, label_cohen=None, default_fc=1.0, default_p=0.05, max_fc=5, combined=False):
    """Reusable block for sliders under each plot"""
    children = [
        html.Label(label_fc),
        dcc.Slider(
            id=f"{prefix}-fc-slider",
            min=0, max=max_fc, step=1, value=default_fc,
            marks={i: {"label": str(i), "style": {"font-size": "10px"}} for i in range(0, max_fc + 1)},
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
                marks={i: {"label": str(i), "style": {"font-size": "10px"}} for i in range(0, max_fc + 1)},
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
                    marks=None,  # no ticks
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
            ], style={"width": "25%", "display": "inline-block", "margin-left": "10px"})
        ], style={"display": "flex", "align-items": "center"}),

        html.Div(id=f"{prefix}-p-output", style={"margin-top": "5px", "font-weight": "bold"})
    ])

    return html.Div(children)


app.layout = html.Div([
    html.H2("Volcano Plots of Metabolites (ALN vs iSLE)", style={"text-align": "center"}),

    html.Div([
        # Left plot: log2FC
        html.Div([
            dcc.Graph(id="fc-plot"),
            slider_block("fc", "log(FC) cutoff", max_fc=5)
        ], style={"width": "48%", "display": "inline-block", "vertical-align": "top"}),

        # Right plot: log2CohenD
        html.Div([
            dcc.Graph(id="cohen-plot"),
            slider_block("cohen", "log(Cohen’s d) cutoff", max_fc=5)
        ], style={"width": "48%", "display": "inline-block", "vertical-align": "top", "margin-left": "2%"})
    ]),

    html.Div([
        dcc.Graph(id="combined-plot"),
        slider_block("combined", "log(FC) cutoff", label_cohen="log(Cohen’s d) cutoff", max_fc=5, combined=True)
    ], style={"width": "98%", "margin-top": "30px"})
])


def make_volcano(x_col, fc_cutoff, p_cutoff_log, raw_p_cutoff, title, xlabel, ylabel):
    df["Category"] = "Not Significant"
    df.loc[(df[x_col] >= fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log), "Category"] = "Up"
    df.loc[(df[x_col] <= -fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log), "Category"] = "Down"

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
        color_discrete_map={"Up": "red", "Down": "blue", "Not Significant": "grey"},
        hover_data={"Compounds": True, "Index": True,
                    "FC_LN_Vs_iSLE": True, "Cohen's_d_LN_Vs_iSLE": True,
                    "MW_pvalue_ALN_vs_iSLE": True},
        title=title,
        labels={x_col: xlabel, "neg_log10_pval": ylabel}
    )
    for trace in fig.data:
        trace.name = color_map.get(trace.name, trace.name)

    fig.add_hline(y=p_cutoff_log, line_dash="dash", line_color="black", annotation_text="p-value cutoff", annotation_position="top left")
    fig.add_vline(x=fc_cutoff, line_dash="dash", line_color="black", annotation_text="cutoff", annotation_position="top left")
    fig.add_vline(x=-fc_cutoff, line_dash="dash", line_color="black")

    return fig


def make_combined(fc_cutoff, cohen_cutoff, p_cutoff_log, raw_p_cutoff):
    df["Category"] = "Not Significant"
    sig_mask = (df["neg_log10_pval"] >= p_cutoff_log) & (
        (df["log2FC"].abs() >= fc_cutoff) | (df["log2CohenD"].abs() >= cohen_cutoff)
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
        df, x="log2FC", y="neg_log10_pval",
        color="Category",
        color_discrete_map={"Up": "red", "Down": "blue", "Not Significant": "grey"},
        hover_data={"Compounds": True, "Index": True,
                    "FC_LN_Vs_iSLE": True, "Cohen's_d_LN_Vs_iSLE": True,
                    "MW_pvalue_ALN_vs_iSLE": True},
        title="Combined Volcano Plot (log2FC + log2Cohen’s d cutoffs)",
        labels={"log2FC": "log2(FC)", "neg_log10_pval": "-log10(p-value)"}
    )
    for trace in fig.data:
        trace.name = color_map.get(trace.name, trace.name)

    # Add cutoff lines with labels OUTSIDE
    fig.add_hline(y=p_cutoff_log, line_dash="dash", line_color="black",
                  annotation_text="p-value cutoff", annotation_position="outside top")
    fig.add_vline(x=fc_cutoff, line_dash="solid", line_color="red",
                  annotation_text="FC cutoff", annotation_position="outside top")
    fig.add_vline(x=-fc_cutoff, line_dash="solid", line_color="red")
    fig.add_vline(x=cohen_cutoff, line_dash="dot", line_color="blue",
                  annotation_text="Cohen’s d cutoff", annotation_position="outside top")
    fig.add_vline(x=-cohen_cutoff, line_dash="dot", line_color="blue")

    return fig


# --- Callbacks ---
@app.callback(
    [Output("fc-plot", "figure"),
     Output("fc-p-output", "children"),
     Output("fc-p-input", "value"),
     Output("fc-p-slider", "value")],
    Input("fc-fc-slider", "value"),
    Input("fc-p-slider", "value"),
    Input("fc-p-input", "value")
)
def update_fc(fc_cutoff, p_cutoff_log, p_input_val):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "fc-p-input":
        try:
            raw_p_cutoff = float(p_input_val)
        except:
            raw_p_cutoff = 0.05
        raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
        p_cutoff_log = -np.log10(raw_p_cutoff)
    else:
        raw_p_cutoff = 10**(-p_cutoff_log)

    fig_fc = make_volcano("log2FC", fc_cutoff, p_cutoff_log, raw_p_cutoff,
                          "log2(FC) vs -log10(p-value)", "log2(FC)", "-log10(p-value)")
    text = f"Current cutoff: p ≤ {raw_p_cutoff:.5g} ( -log10 = {p_cutoff_log:.2f} )"
    return fig_fc, text, raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("cohen-plot", "figure"),
     Output("cohen-p-output", "children"),
     Output("cohen-p-input", "value"),
     Output("cohen-p-slider", "value")],
    Input("cohen-fc-slider", "value"),
    Input("cohen-p-slider", "value"),
    Input("cohen-p-input", "value")
)
def update_cohen(cohen_cutoff, p_cutoff_log, p_input_val):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "cohen-p-input":
        try:
            raw_p_cutoff = float(p_input_val)
        except:
            raw_p_cutoff = 0.05
        raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
        p_cutoff_log = -np.log10(raw_p_cutoff)
    else:
        raw_p_cutoff = 10**(-p_cutoff_log)

    fig = make_volcano("log2CohenD", cohen_cutoff, p_cutoff_log, raw_p_cutoff,
                       "log2(Cohen’s d) vs -log10(p-value)",
                       "log2(Cohen’s d)", "-log10(p-value)")
    text = f"Current cutoff: p ≤ {raw_p_cutoff:.5g} ( -log10 = {p_cutoff_log:.2f} )"
    return fig, text, raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("combined-plot", "figure"),
     Output("combined-p-output", "children"),
     Output("combined-p-input", "value"),
     Output("combined-p-slider", "value")],
    Input("combined-fc-slider", "value"),
    Input("combined-cohen-slider", "value"),
    Input("combined-p-slider", "value"),
    Input("combined-p-input", "value")
)
def update_combined(fc_cutoff, cohen_cutoff, p_cutoff_log, p_input_val):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "combined-p-input":
        try:
            raw_p_cutoff = float(p_input_val)
        except:
            raw_p_cutoff = 0.05
        raw_p_cutoff = max(min(raw_p_cutoff, 1.0), 1e-10)
        p_cutoff_log = -np.log10(raw_p_cutoff)
    else:
        raw_p_cutoff = 10**(-p_cutoff_log)

    fig_combined = make_combined(fc_cutoff, cohen_cutoff, p_cutoff_log, raw_p_cutoff)
    text = f"Current cutoff: p ≤ {raw_p_cutoff:.5g} ( -log10 = {p_cutoff_log:.2f} )"
    return fig_combined, text, raw_p_cutoff, p_cutoff_log


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
