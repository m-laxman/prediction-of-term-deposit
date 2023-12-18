import pandas as pd
import dash
from dash import dcc, html, callback
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
dash.register_page(__name__, path='/distribution', name="Distribution 📊")

df = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-full.csv', sep=';')
df.rename(columns={'y': 'subscribed'}, inplace=True)

columns = df.columns

layout = html.Div(
    children=[
        html.Br(),
        html.H2("Distribution", style={'text-align': 'center'}),
        html.Br(),
        html.P("Select Column:"),
        dcc.Dropdown(id="feature",
                     options=[{'label': i, 'value': i} for i in columns],
                     value="age",
                     clearable=False,
                     className='m-2'),
            dcc.Graph(id="graph")
    ])


@callback(Output("graph", "figure"),
          [Input("feature", "value")])
def update_fig(feature):
    fig = px.histogram(df,
                       x=feature,
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
