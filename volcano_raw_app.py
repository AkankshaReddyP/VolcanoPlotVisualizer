import pandas as pd
import numpy as np
import plotly.express as px
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

def slider_block(prefix, default_fc=1.0, default_p=0.05):
    """Reusable block for sliders under each plot"""
    return html.Div([
        html.Label("Effect size cutoff:"),
        dcc.Slider(
            id=f"{prefix}-fc-slider",
            min=0.1, max=3, step=0.05, value=default_fc,
            marks=None,  # avoid clutter
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Br(),
        html.Label("p-value cutoff:"),
        html.Div([
            html.Div([
                dcc.Slider(
                    id=f"{prefix}-p-slider",
                    min=-np.log10(0.2), max=-np.log10(1e-6), step=0.05,
                    value=-np.log10(default_p),
                    marks={
                        -np.log10(0.2): {"label": "0.2", "style": {"font-size": "10px"}},
                        -np.log10(0.05): {"label": "0.05", "style": {"font-size": "10px"}},
                        -np.log10(0.01): {"label": "0.01", "style": {"font-size": "10px"}},
                        -np.log10(0.001): {"label": "0.001", "style": {"font-size": "10px"}},
                        -np.log10(1e-6): {"label": "1e-6", "style": {"font-size": "10px"}}
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={"width": "70%", "display": "inline-block"}),

            html.Div([
                dcc.Input(
                    id=f"{prefix}-p-input", type="number",
                    value=default_p, step=0.001,
                    min=1e-6, max=0.2
                )
            ], style={"width": "25%", "display": "inline-block", "margin-left": "10px"})
        ], style={"display": "flex", "align-items": "center"}),

        html.Div(id=f"{prefix}-p-output", style={"margin-top": "5px", "font-weight": "bold"})
    ])


app.layout = html.Div([
    html.H2("Volcano Plots of Metabolites (ALN vs iSLE)", style={"text-align": "center"}),

    html.Div([
        # Left plot: log2FC
        html.Div([
            dcc.Graph(id="fc-plot"),
            slider_block("fc")
        ], style={"width": "48%", "display": "inline-block", "vertical-align": "top"}),

        # Right plot: log2CohenD
        html.Div([
            dcc.Graph(id="cohen-plot"),
            slider_block("cohen")
        ], style={"width": "48%", "display": "inline-block", "vertical-align": "top", "margin-left": "2%"})
    ])
])


def make_volcano(x_col, fc_cutoff, p_cutoff_log, raw_p_cutoff, title, xlabel, ylabel):
    # Classify points
    df["Category"] = "Not Significant"
    df.loc[(df[x_col] >= fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log), "Category"] = "Up"
    df.loc[(df[x_col] <= -fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log), "Category"] = "Down"

    # Counts for legend labels
    up_count = (df["Category"] == "Up").sum()
    down_count = (df["Category"] == "Down").sum()
    notsig_count = (df["Category"] == "Not Significant").sum()
    color_map = {
        "Up": f"Up (n={up_count})",
        "Down": f"Down (n={down_count})",
        "Not Significant": f"Not Significant (n={notsig_count})"
    }

    # Volcano plot
    fig = px.scatter(
        df, x=x_col, y="neg_log10_pval",
        color="Category",
        color_discrete_map={"Up": "red", "Down": "blue", "Not Significant": "grey"},
        hover_data={
            "Compounds": True,
            "Index": True,
            "FC_LN_Vs_iSLE": True,
            "Cohen's_d_LN_Vs_iSLE": True,
            "MW_pvalue_ALN_vs_iSLE": True
        },
        title=title,
        labels={x_col: xlabel, "neg_log10_pval": ylabel}
    )

    # Relabel legend items with counts
    for trace in fig.data:
        trace.name = color_map.get(trace.name, trace.name)

    # Add cutoff lines
    fig.add_hline(y=p_cutoff_log, line_dash="dash", line_color="black")
    fig.add_vline(x=fc_cutoff, line_dash="dash", line_color="black")
    fig.add_vline(x=-fc_cutoff, line_dash="dash", line_color="black")

    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))

    return fig


# --- Callbacks for each plot ---
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
        raw_p_cutoff = p_input_val
        p_cutoff_log = -np.log10(raw_p_cutoff)
    else:
        raw_p_cutoff = 10**(-p_cutoff_log)

    fig = make_volcano("log2FC", fc_cutoff, p_cutoff_log, raw_p_cutoff,
                       "log2 Fold Change vs -log10(p-value)",
                       "log2 Fold Change", "-log10(p-value)")
    text = f"Current cutoff: p ≤ {raw_p_cutoff:.3g} ( -log10 = {p_cutoff_log:.2f} )"
    return fig, text, raw_p_cutoff, p_cutoff_log


@app.callback(
    [Output("cohen-plot", "figure"),
     Output("cohen-p-output", "children"),
     Output("cohen-p-input", "value"),
     Output("cohen-p-slider", "value")],
    Input("cohen-fc-slider", "value"),
    Input("cohen-p-slider", "value"),
    Input("cohen-p-input", "value")
)
def update_cohen(fc_cutoff, p_cutoff_log, p_input_val):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "cohen-p-input":
        raw_p_cutoff = p_input_val
        p_cutoff_log = -np.log10(raw_p_cutoff)
    else:
        raw_p_cutoff = 10**(-p_cutoff_log)

    fig = make_volcano("log2CohenD", fc_cutoff, p_cutoff_log, raw_p_cutoff,
                       "Cohen’s d (log2) vs -log10(p-value)",
                       "Cohen’s d (log2)", "-log10(p-value)")
    text = f"Current cutoff: p ≤ {raw_p_cutoff:.3g} ( -log10 = {p_cutoff_log:.2f} )"
    return fig, text, raw_p_cutoff, p_cutoff_log


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))  # default to 8050 locally
    app.run(host="0.0.0.0", port=port, debug=True)
