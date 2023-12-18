import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np

dash.register_page(__name__, path='/relationship', name="Relationship 📈")

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

df = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-full.csv', sep=';')
df.rename(columns={'y': 'subscribed'}, inplace=True)

categorical_features = []
for col in df.select_dtypes(include='object').columns:
    categorical_features.append(col)

columns = ['age', 'balance', 'duration', 'day', 'campaign', 'pdays', 'previous']
plots = ['pie', 'histogram', 'bar', 'strip', 'violin']
# numerical = ['age', 'balance', 'duration', 'day', 'campaign', 'pdays', 'previous']
# categorical = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'job', 'month', 'subscribed']

layout = html.Div([
    html.Br(),
    html.H2("Relationship", style={'text-align': 'center'}),
    html.Br(),
    html.Div(
        children=[
            html.Div(
                children=[

                    html.Br(),
                    html.Div(
                        [
                            html.H3("Select feature for X-Axis"),
                            dcc.RadioItems(
                                id="x_axis",
                                options=[{'label': i, 'value': i} for i in columns],
                                value="age",
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.H3("Select feature for Y-Axis"),
                            dcc.RadioItems(
                                id="y_axis",
                                options=[{'label': i, 'value': i} for i in columns],
                                value="age",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                dcc.Loading([
                    dcc.Graph(id="scatter"),
                ]),
                style={"flex": "50%", "padding": "10px"},
            ),
            html.Div(
                [
                    html.Br(),
                    html.H3("Heat Map - Correlation"),
                    dcc.Checklist(
                        id='feature',
                        options=[{'label': i, 'value': i} for i in columns],
                        value=columns,
                        labelStyle={'display': 'block'},
                    ),
                    dcc.Loading([
                        dcc.Graph(id="heatmap"),
                    ]),
                ],
                style={"flex": "50%", "padding": "10px"},
            ),
        ],
        style={"display": "flex", "flexDirection": "row", "justifyContent": "center"},
    ),
    html.Div(
        [
            html.Br(),
            html.H2("Distribution"),
            html.Br(),
            html.P("Select Plot:"),
            dcc.Dropdown(id="plot",
                         options=[{'label': i, 'value': i} for i in plots],
                         value="pie",
                         clearable=False,
                         className='m-2'),
            html.P("Select Numerical feature:"),
            dcc.Dropdown(id="f1",
                         options=[{'label': i, 'value': i} for i in columns],
                         value="age",
                         clearable=False,
                         className='m-2'),
            html.P("Select Categorical feature:"),
            dcc.Dropdown(id="f2",
                         options=[{'label': i, 'value': i} for i in categorical_features],
                         value="job",
                         clearable=False,
                         className='m-2'),
            dcc.Loading([
                dcc.Graph(id="plot_out")
            ]),
        ],
        style={"flex": "50%", "padding": "10px"},
    ),
])


def generate_hist(f1, f2):
    fig = fig = px.histogram(df, x=f1, color=f2)
    return fig


def generate_bar(f1, f2):
    fig = px.bar(df, x=f2, y=f1, color=f2)
    return fig


def generate_strip(f1, f2):
    fig = px.strip(df, x=f2, y=f1, color=f2)
    return fig


def generate_violin(f1, f2):
    fig = px.violin(df, x=f2, y=f1, color=f2)
    return fig


@callback(
    Output("scatter", "figure"),
    Output("heatmap", "figure"),
    Output("plot_out", "figure"),
    [Input("x_axis", "value"),
     Input("y_axis", "value"),
     Input("feature", "value"),
     Input("plot", "value"),
     Input("f1", "value"),
     Input("f2", "value")])
def update_fig(x_axis, y_axis, feature, plot, f1, f2):
    fig1 = px.scatter(data_frame=df,
                      x=x_axis,
                      y=y_axis,
                      height=800,
                      title=f'{x_axis} vs {y_axis} distribution')
    fig1.update_layout(xaxis=font_ticks,
                       yaxis=font_ticks,
                       title_font=title_font,
                       legend_title_font=legend_title_font,
                       title_x=0.5,
                       font=legend_font)
    correlation_matrix = df[feature].corr().round(2)

    fig2 = px.imshow(correlation_matrix, text_auto=True, height=565)
    fig2.update_layout(xaxis=font_ticks,
                       yaxis=font_ticks,
                       title_font=title_font,
                       legend_title_font=legend_title_font,
                       title_x=0.5,
                       font=legend_font)

    fig = px.pie(df, values=f1, names=f2, hole=.3)
    if plot == 'histogram':
        fig = generate_hist(f1, f2)
    if plot == 'bar':
        fig = generate_bar(f1, f2)
    if plot == 'strip':
        fig = generate_strip(f1, f2)
    if plot == 'violin':
        fig = generate_violin(f1, f2)

    return fig1, fig2, fig
