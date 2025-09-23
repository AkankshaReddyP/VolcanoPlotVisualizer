import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

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
    
    # Layout: controls + counts + plot
    html.Div([
        # Left side: controls and counts
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
            dcc.Slider(
                id="fc-slider",
                min=0.1, max=3, step=0.1, value=1,
                marks={i: str(i) for i in range(0, 4)},
                tooltip={"placement": "bottom"}
            ),
            html.Br(),
            html.Label("p-value cutoff:"),
            dcc.Slider(
                id="p-slider",
                min=-np.log10(0.2), max=-np.log10(1e-6), step=0.1,
                value=-np.log10(0.05),
                marks={-np.log10(0.05): "0.05", -np.log10(0.01): "0.01"},
                tooltip={"placement": "bottom"}
            ),
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
     Output("counts-box", "children")],
    Input("x-axis-choice", "value"),
    Input("fc-slider", "value"),
    Input("p-slider", "value")
)
def update_volcano(x_col, fc_cutoff, p_cutoff):
    # Classify points
    df["Category"] = "Not Significant"
    df.loc[(df[x_col] >= fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff), "Category"] = "Up"
    df.loc[(df[x_col] <= -fc_cutoff) & (df["neg_log10_pval"] >= p_cutoff), "Category"] = "Down"

    # Count categories
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

    # Define colors
    color_map = {"Up": "red", "Down": "blue", "Not Significant": "grey"}

    # Make volcano plot
    fig = px.scatter(
        df,
        x=x_col, y="neg_log10_pval",
        color="Category",
        color_discrete_map=color_map,
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
    fig.add_hline(y=p_cutoff, line_dash="dash", line_color="black")
    fig.add_vline(x=fc_cutoff, line_dash="dash", line_color="black")
    fig.add_vline(x=-fc_cutoff, line_dash="dash", line_color="black")

    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))

    return fig, counts_text

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))  # default to 8050 locally
    app.run(host="0.0.0.0", port=port, debug=True)




