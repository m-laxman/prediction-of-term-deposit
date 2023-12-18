import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

dash.register_page(__name__, path='/pca', name="Principal component analysis 🎯")

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

# df = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-full.csv', sep=';')
# df.rename(columns={'y': 'subscribed'}, inplace=True)
#
# numerical = []
# numerical_orig = []
# for col in df.select_dtypes(exclude='object').columns:
#     numerical.append(col)
#     numerical_orig.append(col)
#
# categorical_features = []
# for col in df.select_dtypes(include='object').columns:
#     categorical_features.append(col)
# categorical_features.remove('subscribed')
#
# X = df.drop('subscribed', axis=1)
# y = df['subscribed']
#
# # Encoding
# categorical_one_hot = []  # ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
# categorical_label = []  # ['job', 'month']
# for col in categorical_features:
#     if X[col].nunique() < 5:
#         categorical_one_hot.append(col)
#     else:
#         categorical_label.append(col)
# categorical_one_hot.remove('education')  # education ordinal must be label encoded
# categorical_label.append('education')
# numerical.extend(categorical_label)
#
# X_encoded = pd.get_dummies(X, columns=categorical_one_hot, dtype=int, drop_first=True)
# y_encoded = pd.get_dummies(y, columns='subscribed', dtype=int, drop_first=True)
#
# label_encoder = LabelEncoder()
# for col in categorical_label:
#     label_encoder = LabelEncoder()
#     X_encoded[col] = label_encoder.fit_transform(X_encoded[col])
#
# y_encoded.rename(columns={'yes': 'subscribed'}, inplace=True)
# encoded_features = ['job', 'month', 'education', 'marital_married', 'marital_single',
#                     'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone',
#                     'contact_unknown', 'poutcome_other', 'poutcome_success',
#                     'poutcome_unknown']
#
# categorical_all = ['day'] + encoded_features
#
# # Resampling
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)
# df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
#
# numerical = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
#
# # Standardization
# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(X_resampled[numerical])
# X_standardized = pd.DataFrame(X_standardized, columns=numerical)

X_std_res_enc = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-res-enc-std.csv')
# y_res_enc = y_resampled.copy()
#
# df_std_res_enc = pd.concat([X_std_res_enc, y_res_enc], axis=1)
#
#
# X_std_res_enc.to_csv('bank-res-enc-std.csv', index=False)

# Dimensionality reduction
# PCA
pca = PCA(n_components=X_std_res_enc.shape[1], svd_solver='full')
pca.fit(X_std_res_enc)
df_std_orig_pca = pca.transform(X_std_res_enc)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

threshold = 0.95
n_components = np.argmax(cumulative_explained_variance >= threshold) + 1

pca_reduced = PCA(n_components=n_components, svd_solver='full')
df_std_reduced_pca = pca_reduced.fit_transform(X_std_res_enc)

layout = html.Div(children=[
    html.Br(),
    html.H2("Principal Component Analysis", style={'text-align': 'center'}),
    html.Br(),
    html.P("Since the dataset is imbalanced w.r.t subscribed(target), oversampling was done using SMOTE."),
    html.P("Data is encoded and standardized."),
    html.Br(),
    html.Button("Download CSV File",
                id="download-csv-button",
                className='btn btn-primary'),
    html.Br(),
    dcc.Download(id="download-csv-data"),
    html.Br(),
    html.P("Adjust threshold:"),
    dcc.Slider(
        id='threshold_inp',
        min=85,
        max=99,
        step=1,
        value=95,
        marks={i: str(i) for i in range(85, 100)},
    ),
    html.Br(),
    dcc.Loading([
        dcc.Graph(id="pca_graph"),
    ])
])


@callback(Output("pca_graph", "figure"),
          [Input("threshold_inp", "value")])
def update_bar_chart(threshold_inp):
    n_components_t = np.argmax(cumulative_explained_variance * 100 >= threshold_inp) + 1
    fig = px.line(x=range(1, len(cumulative_explained_variance) + 1),
                  y=cumulative_explained_variance * 100,
                  markers=True,
                  labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
                  title='PCA Cumulative Explained Variance vs Number of Features',
                  height=800)

    fig.add_vline(x=n_components_t, line_color="blue", line_dash="dash", name=f'{n_components_t} features')

    fig.add_hline(y=threshold_inp, line_color="red", line_dash="dash", name=f'{threshold_inp}% threshold')

    fig.update_layout(xaxis=font_ticks,
                      yaxis=font_ticks,
                      title_font=title_font,
                      legend_title_font=legend_title_font,
                      title_x=0.5,
                      font=legend_font)
    # if download:
    #     return dcc.send_data_frame(X_std_res_enc.to_csv, "bank-full-clean.csv"), fig
    return fig


@callback(Output("download-csv-data", "data"),
          [Input("download-csv-button", "n_clicks")])
def download_csv(n_clicks):
    if n_clicks:
        return dcc.send_data_frame(X_std_res_enc.to_csv, "bank-full-clean.csv")
