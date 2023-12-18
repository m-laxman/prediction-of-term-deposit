from dash import Dash
import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output
import pandas as pd
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-full.csv', sep=';')
# Renamed column 'y' to 'subscribed'
df.rename(columns={'y': 'subscribed'}, inplace=True)

sns.set_palette('Spectral')

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

dash.register_page(__name__, path='/', name="About ℹ️")

layout = html.Div(children=[
    html.H2("About", style={'text-align': 'center'}),
    html.Br(),
    html.Div(children=[
        html.H3("Bank Marketing Dataset Overview"),
        html.P(
            "The data is related with direct marketing campaigns of an European banking institution. The classification goal is to predict if the client would subscribe a term deposit (Yes or No) based on data from previous marketing."),
        html.P(
            "In this dashboard we will explore the dataset and reveal underlying information by visualizing the following features ")
    ]),
    html.Div(children=[
        html.Br(),
        html.H3("Feature description"),
        html.P("age: age of the person (numeric)"),
        html.P("job: type of job (categorical: 'admin','unemployed','management','housemaid','entrepreneur',"
               "'student','blue-collar','self-employed','retired','technician','services','unknown')"),
        html.P("marital: marital status (categorical: 'married','divorced','single'; note: 'divorced' means "
               "divorced or widowed)"),
        html.P("education: (categorical: 'unknown','secondary','primary','tertiary')"),
        html.P("default: has credit in default? (binary: 'yes','no')"),
        html.P("balance: average yearly balance, in euros (numeric)"),
        html.P("housing: has housing loan? (binary: 'yes','no')"),
        html.P("loan: has personal loan? (binary: 'yes','no')"),
        html.P("contact: contact communication type (categorical: 'unknown','telephone','cellular')"),
        html.P("day: last contact day of the month (numeric)"),
        html.P("month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')"),
        html.P("duration: last contact duration, in seconds (numeric)"),
        html.P("campaign: number of contacts performed during this campaign and for this client (numeric, "
               "includes last contact)"),
        html.P("pdays: number of days that passed by after the client was last contacted from a previous "
               "campaign (numeric, -1 means client was not previously contacted)"),
        html.P("previous: number of contacts performed before this campaign and for this client (numeric)"),
        html.P("poutcome: outcome of the previous marketing campaign (categorical: 'unknown','other','failure',"
               "'success')"),
        html.P(
            "y (renamed to subscribed): has the client subscribed a term deposit? (binary: 'yes' or 'no'). This is the target variable"),
        html.Br(),
        html.P("Observations: 45211"),
        html.P("Features: 17 (7 numerical, 10 categorical)"),

        html.Br(),
        html.H3('Dataset'),
        html.Br(),
        html.Button("Download CSV File",
                    id="download",
                    className='btn btn-primary'),
        html.Br(),
        dcc.Download(id="download-csv"),
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=20,
            style_table={
                'overflowX': 'auto',
                'border': 'thin lightgrey solid',
                'borderRadius': '8px'
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
        html.H3('Descriptive statistics'),
        html.Br(),
        html.Img(src=dash.get_asset_url('describe.jpg'), style={'display': 'block', 'margin': 'auto'})

    ])

], className="bg-light p-4 m-2")


@callback(
    Output("download-csv", "data"),
    Input("download", "n_clicks"),
    prevent_initial_call=True,
)
def update_download(n_clicks):
    return dcc.send_data_frame(df.to_csv, "bank-full.csv")
