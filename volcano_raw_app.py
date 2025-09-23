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

app.layout = html.Div([
    html.H2("Interactive Volcano Plot (ALN vs iSLE)"),

    html.Div([
        # Left side: controls + counts
        html.Div([
            html.Label("X-axis choice:"),
            dcc.Dropdown(
                id="x-axis-choice",
                options=[
                    {"label": "log2 Fold Change", "value": "log2FC"},
                    {"label": "log2 Cohen's d", "value": "log2CohenD"}
                ],
                value="log2FC",
                clearable=False
            ),
            html.Br(),

            html.Label("Effect size (log2FC / log2d) cutoff:"),
            html.Div([
                dcc.Slider(
                    id="fc-slider",
                    min=0.1, max=3, step=0.1, value=1,
                    marks={i: str(i) for i in range(0, 4)},
                    tooltip={"placement": "bottom"}
                )
            ], style={"width": "90%"}),
            html.Br(),

            html.Label("p-value cutoff:"),
            html.Div([
                html.Div([
                    dcc.Slider(
                        id="p-slider",
                        min=-np.log10(0.2), max=-np.log10(1e-6), step=0.1,
                        value=-np.log10(0.05),
                        marks={
                            -np.log10(0.2): "0.2 (0.70)",
                            -np.log10(0.1): "0.1 (1.0)",
                            -np.log10(0.05): "0.05 (1.3)",
                            -np.log10(0.01): "0.01 (2.0)",
                            -np.log10(0.001): "0.001 (3.0)",
                            -np.log10(1e-6): "1e-6 (6.0)"
                        },
                        tooltip={"placement": "bottom"}
                    )
                ], style={"width": "70%", "display": "inline-block"}),

                html.Div([
                    dcc.Input(
                        id="p-input", type="number", value=0.05, step=0.001,
                        min=1e-6, max=0.2
                    )
                ], style={"width": "25%", "display": "inline-block", "margin-left": "10px"})
            ], style={"display": "flex", "align-items": "center"}),

            html.Div(id="p-slider-output", style={"margin-top": "5px", "font-weight": "bold"}),

            html.Br(),
            html.Div(id="counts-box", style={"border": "1px solid #ccc", "padding": "10px", "margin-top": "10px"})
        ], style={"width": "25%", "display": "inline-block", "vertical-align": "top"}),

        # Right side: plot
        html.Div([
            dcc.Graph(id="volcano-plot")
        ], style={"width": "70%", "display": "inline-block", "padding-left": "20px"})
    ])
])


@app.callback(
    [Output("volcano-plot", "figure"),
     Output("counts-box", "children"),
     Output("p-slider-output", "children"),
     Output("p-input", "value"),
     Output("p-slider", "value")],
    Input("x-axis-choice", "value"),
    Input("fc-slider", "value"),
    Input("p-slider", "value"),
    Input("p-input", "value")
)
def update_volcano(x_col, fc_cutoff, p_cutoff_log, p_input_val):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "p-input":
        # user typed raw p-value
        raw_p_cutoff = p_input_val
        p_cutoff_log = -np.log10(raw_p_cutoff)
    else:
        # user moved slider
        raw_p_cutoff = 10**(-p_cutoff_log)

    # Classify points
    df["Category"] = "Not Significant"
    df.loc[(df[x_col] >= fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log), "Category"] = "Up"
    df.loc[(df[x_col] <= -fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff_log), "Category"] = "Down"

    # Counts
    total = len(df)
    up_count = (df["Category"] == "Up").sum()
    down_count = (df["Category"] == "Down").sum()
    notsig_count = (df["Category"] == "Not Significant").sum()

    counts_text = html.Div([
        html.H4("Counts"),
        html.P(f"Total metabolites: {total}"),
        html.P(f"Significant Up (red): {up_count}"),
        html.P(f"Significant Down (blue): {down_count}"),
        html.P(f"Not Significant (grey): {notsig_count}")
    ])

    # Volcano plot
    fig = px.scatter(
        df,
        x=x_col, y="neg_log10_pval",
        color="Category",
        color_discrete_map={"Up": "red", "Down": "blue", "Not Significant": "grey"},
        hover_data={
            "Compounds": True,
            "Index": True,
            "FC_LN_Vs_iSLE": True,
            "Cohen's_d_LN_Vs_iSLE": True,
            "MW_pvalue_ALN_vs_iSLE": True
        },
        title=f"Volcano Plot ({x_col})"
    )

    # Add cutoff lines
    fig.add_hline(y=p_cutoff_log, line_dash="dash", line_color="black")
    fig.add_vline(x=fc_cutoff, line_dash="dash", line_color="black")
    fig.add_vline(x=-fc_cutoff, line_dash="dash", line_color="black")

    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))

    # Slider/output text
    p_slider_text = f"Current cutoff: p ≤ {raw_p_cutoff:.3g} ( -log10 = {p_cutoff_log:.2f} )"

    return fig, counts_text, p_slider_text, raw_p_cutoff, p_cutoff_log


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))  # default to 8050 locally
    app.run(host="0.0.0.0", port=port, debug=True)
