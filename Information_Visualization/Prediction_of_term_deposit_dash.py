import math
import numpy as np
from dash import Dash
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import seaborn as sns

title_font = dict(size=30, family='Times New Roman', color='Blue')
legend_font = dict(size=30, family='Courier New', color='Red')
legend_title_font = dict(
    family='Courier New',
    size=30,
    color='green',
)
font_ticks = dict(tickfont=dict(
    family='Courier New',
    size=30,
    color='black',
))

external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css']

my_app = Dash('Bank_dashboard', pages_folder='pages', use_pages=True, external_stylesheets=external_stylesheets)


# print(df)
def generate_buttons():
    page_buttons = []
    for page in dash.page_registry.values():
        button = dcc.Link(page['name'],
                          href=page['relative_path'],
                          className='btn btn-outline-danger m-2',
                          )
        page_buttons.append(button)
    return page_buttons


my_app.layout = html.Div([
    html.Br(),
    dcc.Loading([
        html.H1('Bank Marketing - Data Visualization', className='text-center'),
        html.Br(),
        html.Br(),
        html.Div(children=generate_buttons(), className='text-center'),

        dash.page_container
    ])
], className='px-5', style={'background-color': '#F8E7E7'})

my_app.run_server(
    port=8057,
    host='0.0.0.0'
)
