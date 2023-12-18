import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.stats import kstest, normaltest, shapiro, boxcox
from sklearn.preprocessing import StandardScaler, LabelEncoder

dash.register_page(__name__, path='/normality', name="Normality Test 📜")

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

columns = ['age', 'campaign']
tests = ['Kolmogorov–Smirnov test', 'Shapiro–Wilk test', "D'Agostino's K-squared test"]
# numerical = ['age', 'balance', 'duration', 'day', 'campaign', 'pdays', 'previous']
# categorical = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'job', 'month', 'subscribed']

layout = html.Div([
    html.Br(),
    html.H2("Normality test", style={'text-align': 'center'}),
    html.Br(),
    html.Div(
        children=[
            dcc.Dropdown(id="numerical",
                         options=[{'label': i, 'value': i} for i in columns],
                         value="age",
                         clearable=False,
                         className='m-2'),
            html.Div(
                    dcc.Graph(id="hist"),
            ),
            html.H3('Select the test: '),
            html.Div(
                dcc.Dropdown(id="test",
                             options=[{'label': i, 'value': i} for i in tests],
                             value="age",
                             clearable=False,
                             className='m-2'),
            ),
            html.H3(id="otp"),
            html.Div(
                    dcc.Graph(id="hist-trf"),
            ),
        ])
])


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    np.random.seed(5764)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    s1 = f'K-S test: {title} : statistics= {stats:.2f} p-value = {p:.2f}'
    if p > 0.01:
        type = 'Gaussian'
    else:
        type = 'Non Gaussian'
    s2 = f'K-S test: {title} looks {type}'
    return s1 + "\n" + s2


def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    s1 = f'da_k_squared test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}'
    if p > 0.01:
        type = 'Gaussian'
    else:
        type = 'Non Gaussian'
    s2 = f'da_k_squared test: {title} looks {type}'
    return s1 + "\n" + s2


def shapiro_test(x, title):
    stats, p = shapiro(x)
    s1 = f'Shapiro test : {title} : statistics = {stats:.2f} p-value of = {p:.2f}'
    if p > 0.01:
        type = 'Gaussian'
    else:
        type = 'Non Gaussian'
    s2 = f'Shapiro test: {title} looks {type}'
    return s1 + "\n" + s2


@callback(
    Output("hist", "figure"),
    Output("otp", "children"),
    Output("hist-trf", "figure"),
    [Input("numerical", "value"),
     Input("test", "value")])
def update_fig(numerical, test):
    fig = px.histogram(df,
                       x=numerical,
                       height=800,
                       title=f'{numerical} distribution')
    # fig.update_traces(line=dict(width=4))
    fig.update_layout(xaxis=font_ticks,
                      yaxis=font_ticks,
                      title_font=title_font,
                      legend_title_font=legend_title_font,
                      title_x=0.5,
                      font=legend_font)
    ot = ""
    fig1 = px.histogram(df,
                        x=numerical,
                        height=800,
                        title=f'{numerical} distribution')
    if test == 'Kolmogorov–Smirnov test':
        ot = ks_test(df[numerical], numerical)
    if test == 'Shapiro–Wilk test':
        ot = shapiro_test(df[numerical], numerical)
    if test == "D'Agostino's K-squared test":
        ot = da_k_squared_test(df[numerical], numerical)

    if 'Non Gaussian' in ot:
        transformed_data, best_lambda = boxcox(df[numerical])
        transformed_data = pd.DataFrame({numerical: transformed_data})
        fig1 = px.histogram(transformed_data,
                            x=numerical,
                            height=800,
                            title=f'{numerical} distribution after Boxcox transformation with best lambda {best_lambda}')
        fig1.update_layout(xaxis=font_ticks,
                           yaxis=font_ticks,
                           title_font=title_font,
                           legend_title_font=legend_title_font,
                           title_x=0.5,
                           font=legend_font)

    return fig, ot, fig1
