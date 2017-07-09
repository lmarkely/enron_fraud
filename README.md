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
visualization techniques, including scatter plot matrix and PCA will be
used (stage 2). This visualization will help identify if there is any redundancy
among the features due to correlation, as well as provide some estimation on how
the data are distributed in high dimensional space. If there are correlations
among the features, we may return to stage 1 to reselect the feature.
After the features are selected and engineered, nested cross validation will
be used for algorithm selection in stage 3. Then, stage 4 identifies the
hyperparameter values that give the best result for the selected algorithm.
Three metrics, F1 score, precision, and recall will be used during the algorithm
and model selection.

## Feature Selection and Engineering
First, the row corresponding to 'TOTAL' as we are interested in the data of
individuals. In addition, 'total_payments' and 'total_stock_values' are removed
as they are aggregate of other features. Moreover, 'to_messages',
'email_address', 'from_poi_to_this_person', 'from_messages', and 'from_this_person_to_poi' are not included. Instead, they are standardized to 'std_from_poi' and 'std_to_poi'. These steps are captured in the following
codes.

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
First, a quick data exploration shows the followings
```
### First, explore the dataset.
### Identify the total number of data points.
print 'Total number of data points:',np.shape(X)[0]
print 'Total number of features:', np.shape(X)[1]
```

The output of the above code is as follows.
```
Total number of data points: 144
Total number of features: 15

```
Pairplot of the dataset shows that there are some, but weak correlation among
the features (**Fig. 1**). Thus, all features will be used in the following
steps.

![Plot](https://github.com/lmarkely/enron_fraud/blob/master/Fig%201.png)

**Figure 1.** Pairplot of all features of Enron dataset.

Furthermore, PCA

```
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_std)
print 'PCA explained_variance_ratio_', pca.explained_variance_ratio_
```

Output:
```
PCA explained_variance_ratio_ [ 0.77628412  0.07772042]
```

This plot is generated using [Seaborn](http://seaborn.pydata.org/generated/seaborn.pairplot.html).
```
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(X_std)
pg = sns.PairGrid(df)
pg.map_diag(plt.hist)
pg.map_offdiag(plt.scatter)
plt.show()
```
