import pandas as pd
import dash
from dash import dcc, html, callback, dash_table
import plotly.express as px
from dash.dependencies import Input, Output
import seaborn as sns

px.defaults.template = "presentation"
title_font = dict(size=30, family='Times New Roman', color='Blue')
legend_font = dict(size=30, family='Courier New', color='Red')
legend_title_font = dict(
    family='Courier New',
    size=30,
    color='green',
)
font_ticks = dict(tickfont=dict(
    family='Courier New',
    size=22,
    color='black',
))

sns.set_palette('Spectral')
dash.register_page(__name__, path='/preprocessing', name="Preprocessing 🔍")

df = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-full.csv', sep=';')
df.rename(columns={'y': 'subscribed'}, inplace=True)

na_counts = df.isna().sum().reset_index()
na_counts.rename(columns={'index': 'Feature'}, inplace=True)
na_counts.rename(columns={0: 'Number of missing values'}, inplace=True)

columns = ['age', 'balance', 'duration', 'day', 'campaign', 'pdays', 'previous']

layout = html.Div(
    children=[
        html.Br(),
        html.H2("Preprocessing", style={'text-align': 'center'}),
        html.Br(),
        html.H3("Missing Values"),
        html.Br(),
        dash_table.DataTable(
            data=na_counts.to_dict('records'),
            page_size=20,
            style_table={
                'overflowX': 'auto',
                'border': 'thin lightgrey solid',
                'borderRadius': '8px',
                'width': '25%',
                'margin': 'auto'
            },
            style_header={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': '1px solid white',
            },
            style_cell={
                'backgroundColor': 'rgb(240, 240, 240)',
                'color': 'black',
                'textAlign': 'left',
                'border': '1px solid white',
                'padding': '8px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'column_editable': True},
                    'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                    'color': 'black'
                }
            ]
        ),
        html.Br(),
        html.H3("Outliers"),
        html.P("Select Columns:"),
        dcc.Dropdown(id="feature-s",
                     options=[{'label': i, 'value': i} for i in columns],
                     value="balance",
                     clearable=False,
                     multi=True,
                     className='m-2'),
        dcc.Graph(id="graph-out")
    ])


@callback(Output("graph-out", "figure"),
          [Input("feature-s", "value")])
def update_fig(feature):
    fig = px.box(df,
                 y=feature,
                 height=800,
                 title=f'{feature} distribution')
    # fig.update_traces(line=dict(width=4))
    fig.update_layout(xaxis=font_ticks,
                      yaxis=font_ticks,
                      title_font=title_font,
                      legend_title_font=legend_title_font,
                      title_x=0.5,
                      font=legend_font)
    return fig
