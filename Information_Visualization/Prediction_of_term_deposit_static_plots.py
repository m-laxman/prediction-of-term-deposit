import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from prettytable import PrettyTable, ALL
import statsmodels.api as sm
import plotly.express as px
from statsmodels.graphics.gofplots import qqplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

sns.set(
    rc={
        'axes.labelcolor': 'darkred',
        'font.family': 'serif',
        'lines.linewidth': 2.5,
    },
    style='darkgrid',
    context='notebook',
    palette='Spectral'
)
title_fontdict = {"font": "serif", "color": "blue"}
font = {'fontname': 'serif', 'color': 'darkred'}
df_bank = pd.read_csv('https://raw.githubusercontent.com/m-laxman/dataset_term_project/main/bank-full.csv', sep=";")
df = df_bank.copy()
pd.set_option('display.max_columns', 17)

# Preprocessing
# Renamed column 'y' to 'subscribed'
df.rename(columns={'y': 'subscribed'}, inplace=True)
print(df.info())

print(df.head().round(2))

print(df.describe().round(2))

# Check for missing Values
print(df.isna().sum())
print(f'Missing values : {df.isna().sum().sum()}')

# Check for unwanted Columns
categorical = []
for col in df.select_dtypes(include='object').columns:
    print(col, ':', df[col].nunique())
    categorical.append(col)

# Check if any rows are duplicates
print(f'Duplicated rows : {df.duplicated().sum()}')

# EDA
# Explore the Categorical Features
for col in df.select_dtypes(include='object').columns:
    print(col, ':', df[col].unique())

# Categorical Feature Distribution
plt.figure(figsize=(22, 20))
i = 0
for j in range(0, len(categorical), 2):
    i += 1
    plt.subplot(3, 2, i)
    sns.countplot(data=df, x=categorical[j])
    plt.title(f'{categorical[j]} distribution', title_fontdict)
    plt.xticks(rotation=45)
plt.suptitle(f'Categorical Feature Distribution', fontdict=title_fontdict)
plt.show()

plt.figure(figsize=(22, 20))
i = 0
for j in range(1, len(categorical), 2):
    i += 1
    plt.subplot(3, 2, i)
    plt.pie(df[categorical[j]].value_counts(), labels=df[categorical[j]].value_counts().index, autopct='%1.2f%%')
    plt.title(f'{categorical[j]} ', title_fontdict)
    plt.legend()
plt.suptitle(f'Categorical Feature Distribution', fontdict=title_fontdict)
plt.show()

# Relationship between Categorical Features and Target
for j in range(0, len(categorical), 2):
    if (categorical[j] != 'subscribed'):
        plt.figure(figsize=(12, 12))
        sns.catplot(data=df, x=categorical[j], kind='count', hue='subscribed')
        plt.title(f'{categorical[j]} ', title_fontdict)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

for j in range(1, len(categorical), 2):
    if (categorical[j] != 'subscribed'):
        sns.histplot(data=df, x=categorical[j], hue='subscribed', multiple='stack')
        plt.title(f'{categorical[j]} ', title_fontdict)
        plt.tight_layout()
        plt.show()


for col in categorical:
    print(df.groupby(['subscribed', col]).size())

# Explore the Numerical Features
numerical = []
for col in df.select_dtypes(exclude='object').columns:
    print(col, ':', df[col].nunique())
    numerical.append(col)

# Distribution of Numerical Features
plt.figure(figsize=(22, 20))
for i, col in enumerate(numerical):
    ax = plt.subplot(4, 2, i + 1)
    sns.histplot(data=df, x=col, kde=True)
    plt.xlabel(col)
plt.suptitle(f'Numerical features distribution', fontdict=title_fontdict)
plt.show()

# Determine which month has highest duration on calls
month_phone_df = df[['month', 'duration']]
plt.plot(month_phone_df['month'].value_counts().sort_index())
plt.title(f'Month with highest duration of phone call', title_fontdict)
plt.xlabel('month')
plt.ylabel('duration')
plt.tight_layout()
plt.show()

# Relation between numerical Features and Labels
plt.figure(figsize=(22, 20), facecolor='white')
for i, col in enumerate(numerical):
    ax = plt.subplot(4, 2, i + 1)
    sns.boxplot(data=df, x='subscribed', y=df[col])
    plt.xlabel(col)

plt.suptitle(f'Numerical features vs subscribed', fontdict=title_fontdict)
plt.show()

# pair plot for numerical
# kind='kde'
sns.pairplot(df[numerical], diag_kind='kde')
plt.suptitle(f'Pair plot', fontdict=title_fontdict)
plt.tight_layout()
plt.show()

# Find Outliers in numerical features
plt.figure(figsize=(22, 20), facecolor='white')
for i, col in enumerate(numerical):
    ax = plt.subplot(4, 2, i + 1)
    sns.boxplot(data=df, x=col)
    plt.xlabel(col)
plt.suptitle(f'Outlier detection', fontdict=title_fontdict)
plt.show()

# Explore the Correlation between numerical features
correlation_matrix = df[numerical].corr()
fig = plt.figure(figsize=(15, 7))
sns.heatmap(correlation_matrix, annot=True)
plt.title(f'Heatmap for correlation matrix', title_fontdict)
plt.show()

# Age and subscribed
plt.figure(figsize=(12, 12))
sns.displot(data=df, x='age', hue='subscribed', stat='density', kde=True, alpha=0.6)
plt.title('Age and subscribed', title_fontdict)
plt.show()

# Check the Data set is balanced or not based wrt target values in classification
plt.figure(figsize=(12, 12))
sns.countplot(data=df, x='subscribed')
plt.title(f'Dataset balance w.r.t to Subscribed', title_fontdict)
plt.show()

# Campaign and Age
plt.figure(figsize=(12, 12))
sns.regplot(data=df, x='age', y='campaign', line_kws={'color': 'blue'})
plt.title(f'Age vs campaign', title_fontdict)
plt.show()

# Balance and age
plt.figure(figsize=(12, 12))
qqplot(df['age'], line='s')
plt.title(f'Age ', title_fontdict)
plt.show()

plt.figure(figsize=(12, 12))
qqplot(df['balance'], line='s')
plt.title(f'Balance ', title_fontdict)
plt.show()

# Campaign and subscribed
plt.figure(figsize=(12, 12))
sns.boxenplot(data=df, x='campaign', y='subscribed')
plt.title(f'Campaign and subscribed', title_fontdict)
plt.show()

# age vs last contact day
dim = (40, 20)
plt.figure(figsize=dim)
sns.boxplot(df, x='age', y='day')
plt.title(f'Age vs last contact day', title_fontdict)
plt.show()

# Education and balance
plt.figure(figsize=(12, 12))
sns.stripplot(data=df, x='education', y='balance')
plt.title(f'education and balance', title_fontdict)
plt.show()

# marital and balance
# When there is a lot of data and a lot of points overlap, swarm plot struggles, hence commented the code
# sns.swarmplot(data=df, x='job', y='balance')
# plt.title(f'Job and balance', title_fontdict)
# plt.show()

# age vs duration
plt.figure(figsize=(12, 12))
sns.scatterplot(df, x='duration', y='age', hue='subscribed')
sns.rugplot(df, x='duration', y='age', hue='subscribed')
plt.title(f'age and duration', title_fontdict)
plt.show()

# age and balance
plt.figure(figsize=(12, 12))
sns.jointplot(df, x='age', y='balance', kind='hex')
plt.title(f'age and balance', title_fontdict)
plt.show()

# campaign and marital
plt.figure(figsize=(12, 12))
sns.violinplot(df, x='campaign', y='marital', hue='subscribed')
plt.title(f'campaign and marital', title_fontdict)
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(df['age'], df['balance'], df['duration'], marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('Balance')
ax.set_zlabel('Duration')

plt.title('3D Scatter Plot of Age, Balance, and Duration', title_fontdict)
plt.show()

numerical = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
covariance_matrix = df[numerical].cov()

# Create a cluster map of the covariance matrix
plt.figure(figsize=(10, 8))
sns.clustermap(covariance_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.title('Cluster Map of Covariance Matrix')
plt.show()

# plt.show()
sns.kdeplot(data=df, x='balance', y='age', fill=True, hue='subscribed', alpha=0.6)
plt.title('balance and age', title_fontdict)
plt.tight_layout()
plt.legend()
plt.show()


numerical = []
numerical_orig = []
for col in df.select_dtypes(exclude='object').columns:
    numerical.append(col)
    numerical_orig.append(col)

categorical_features = []
for col in df.select_dtypes(include='object').columns:
    categorical_features.append(col)
categorical_features.remove('subscribed')

X = df.drop('subscribed', axis=1)
y = df['subscribed']

# Encoding
categorical_one_hot = []  # ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
categorical_label = []  # ['job', 'month']
for col in categorical_features:
    if X[col].nunique() < 5:
        categorical_one_hot.append(col)
    else:
        categorical_label.append(col)
categorical_one_hot.remove('education')  # education ordinal must be label encoded
categorical_label.append('education')
numerical.extend(categorical_label)

X_encoded = pd.get_dummies(X, columns=categorical_one_hot, dtype=int, drop_first=True)
y_encoded = pd.get_dummies(y, columns='subscribed', dtype=int, drop_first=True)

label_encoder = LabelEncoder()
for col in categorical_label:
    label_encoder = LabelEncoder()
    X_encoded[col] = label_encoder.fit_transform(X_encoded[col])

y_encoded.rename(columns={'yes': 'subscribed'}, inplace=True)
encoded_features = ['job', 'month', 'education', 'marital_married', 'marital_single',
                    'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone',
                    'contact_unknown', 'poutcome_other', 'poutcome_success',
                    'poutcome_unknown']

categorical_all = ['day'] + encoded_features

# Resampling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

numerical = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_resampled[numerical])
X_standardized = pd.DataFrame(X_standardized, columns=numerical)

X_std_res_enc = pd.concat([X_standardized, df_resampled[categorical_all]], axis=1)
y_res_enc = y_resampled.copy()

df_std_res_enc = pd.concat([X_std_res_enc, y_res_enc], axis=1)


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

plt.figure(figsize=(12, 12))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance * 100, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
ticks = np.arange(1, len(cumulative_explained_variance) + 1)
plt.xticks(ticks)
plt.axvline(x=n_components, label='4 features', color='blue', linestyle='dashed')
plt.axhline(y=95, label='95% threshold', color='red', linestyle='dashed')
plt.title('PCA Cumulative Explained Variance vs Number of Features', title_fontdict)
plt.show()

# print(f'PCA: condition number for original: {np.linalg.cond(X_std_res_enc):.2f}')
# print(f'condition number for reduced: {np.linalg.cond(df_std_reduced_pca):.2f}')

singular_values_formatted = ', '.join([f'{value:.2f}' for value in pca_reduced.singular_values_])
# displaying first 10
singular_values_formatted_orig = ', '.join([f'{value:.2f}' for value in pca.singular_values_[:10]])

pca_table = PrettyTable()
pca_table.field_names = ['', 'Original Dataset', 'Reduced Dataset']
pca_table.add_row(
    ['Condition number', f'{np.linalg.cond(X_std_res_enc):.2f}', f'{np.linalg.cond(df_std_reduced_pca):.2f}'])
pca_table.add_row(['Singular Values', f'{singular_values_formatted_orig}', f'{singular_values_formatted}'])
pca_table.add_row(['No. of components', X_std_res_enc.shape[1], n_components])
pca_table.float_format = ".2"
pca_table.hrules = ALL
print(pca_table.get_string(title='PCA comparison table'))
