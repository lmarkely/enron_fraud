# Enron Fraud project

Enron was one of the largest companies in the US. It went into bankruptcy due to corporate fraud. A significant amount of Enron data (emails
and financial data) were entered into public record as a result of Federal
investigation. This project aims to build machine learning algorithm to
identify Enron employees who may have committed fraud based on the public Enron
financial and email dataset. More details about Enron scandal can be found on
[Wikipedia](https://en.wikipedia.org/wiki/Enron_scandal).

## Workflow
This project is divided into 4 main stages:
1. Feature selection and engineering
2. Data visualization
3. Algorithm selection
4. Model selection

Stage 1 will be performed with the help of stage 2. Stage 1 will determine the
features that will be included for machine learning. Here, various data
visualization techniques, including scatter plot matrix and dimensionality
reduction (Principal Component Analysis & Linear Discriminant Analysis) will be
used (stage 2).This visualization will help identify if there is any redundancy
among the features due to correlation, as well as provide some estimation on how
the data are distributed in high dimensional space. If there are correlations
among the features, we may return to stage 1 to reselect the feature.
After the features are selected and engineered, nested cross validation will
be used for algorithm selection in stage 3. Then, stage 4 identifies the
hyperparameter values that give the best result for the selected algorithm.
Three metrics, F1 score, precision, and recall will be used during the algorithm
and model selection.

## Feature Selection and Engineering
First, the row corresponding to 'TOTAL' is removed. In addition, all features
are included except 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', and 'from_this_person_to_poi'.
Furthermore, 'std_from_poi' and 'std_to_poi' are standardized features derived
as follows.

```
data_dict.pop('TOTAL')
for key in data_dict:
    if (type(data_dict[key]['from_poi_to_this_person']) == int and
        type(data_dict[key]['from_messages']) == int):
        data_dict[key]['std_from_poi'] = \
        (data_dict[key]['from_poi_to_this_person']/
         data_dict[key]['from_messages'])
    else:
        data_dict[key]['std_from_poi'] = 0
    if (type(data_dict[key]['from_this_person_to_poi']) == int and
        type(data_dict[key]['to_messages']) == int):
        data_dict[key]['std_to_poi'] = \
        (data_dict[key]['from_this_person_to_poi']/
         data_dict[key]['to_messages'])
    else:
        data_dict[key]['std_to_poi'] = 0
```

## Data Visualization
First, PCA and LDA was done on features standardized using StandardScaler from
scikit-learn. A quick data exploration shows the followings
```
### First, explore the dataset.
### Identify the total number of data points.
print 'Total number of data points:',np.shape(X)[0]
print 'Total number of features:', np.shape(X)[1]

X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_std,y)
print 'PCA explained_variance_ratio_', pca.explained_variance_ratio_
print 'LDA explained variance:', lda.explained_variance_ratio_
```

The output of the above code is as follows.
```
Total number of data points: 145
Total number of features: 17
PCA explained_variance_ratio_ [ 0.77628412  0.07772042]
LDA explained variance: [ 1.]
```
In addition, there is a warning message from running LDA that variables are
collinear. In other words, some of the features are correlated with each other.
This correlation should be avoided for LDA because it implies redundancy and
confuses the interpretation of the  LDA coefficients. More detailed explanation
can be found [here](https://stats.stackexchange.com/questions/29385/collinear-variables-in-multiclass-lda-training). This correlation is confirmed by the following scatterplot
matrix (Fig. 1)

![Plot](https://github.com/lmarkely/enron_fraud/blob/master/Fig%201.png)
