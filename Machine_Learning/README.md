## Dataset
[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages.

```bash
pip install -r requirements.txt
```

## Usage

Run the "Prediction_of_term_deposit.ipynb" jupyter notebook. 


## Model Comparison

| Model                   | Confustion Matrix | Precision | Recall | Specificity | F1 Score | AUC  |
|-------------------------|-------------------|-----------|--------|-------------|----------|------|
| Pre-pruned              | [[6227 1722]      | 0.81      | 0.89   | 0.78        | 0.84     | 0.90 |
|                         | [ 893 7127]]      |           |        |             |          |      |
| Post-pruned             | [[6293 1656]      | 0.81      | 0.88   | 0.79        | 0.85     | 0.90 |
|                         | [ 927 7093]]      |           |        |             |          |      |
| Logistic                | [[6310 1639]      | 0.81      | 0.86   | 0.79        | 0.83     | 0.89 |
|                         | [1122 6898]]      |           |        |             |          |      |
| K Neighbors Classifier  | [[6243 1706]      | 0.81      | 0.89   | 0.79        | 0.85     | 0.91 |
|                         | [ 862 7158]]      |           |        |             |          |      |
| Support Vector Machine  | [[5214 2735]      | 0.73      | 0.92   | 0.66        | 0.82     | 0.89 |
|                         | [ 615 7405]]      |           |        |             |          |      |
| Naive Bayes             | [[5055 2894]      | 0.72      | 0.91   | 0.64        | 0.80     | 0.87 |
|                         | [ 717 7303]]      |           |        |             |          |      |
| Bagging Classifier      | [[6425 1524]      | 0.82      | 0.89   | 0.81        | 0.86     | 0.92 |
|                         | [ 877 7143]]      |           |        |             |          |      |
| Stacking      | [[6379 1570]      | 0.82      | 0.88   | 0.80        | 0.85     | 0.91 |
|                         | [ 967 7053]]      |           |        |             |          |      |
| AdaBoost Classifier     | [[6464 1485]      | 0.83      | 0.88   | 0.81        | 0.85     | 0.92 |
|                         | [ 935 7085]]      |           |        |             |          |      |
| Multi-layer Perceptron  | [[6229 1720]      | 0.80      | 0.88   | 0.78        | 0.84     | 0.90 |
|                         | [ 938 7082]]      |           |        |             |          |      |


### Result
In our comprehensive assessment of diverse classifiers, the 'Stacking' ensemble learning method showcased superior performance over other models, exhibiting remarkable precision, recall, specificity, and achieving an AUC score of 0.92. By adeptly merging the strengths of multiple models, this ensemble method effectively leverages their unique capabilities to ensure accurate predictions. With an 82% precision and 89% recall, the Stacking Classifier robustly identifies both positive and negative instances, maintaining a balanced approach between precision and recall. 