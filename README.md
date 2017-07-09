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
recall score, precision and recall will be used during the algorithm
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
First, a quick data exploration shows the followings.
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

Furthermore, PCA shows that though some Principal Components (PCs), capture more
variance than others, the highest explained ratio is only 0.34. Thus, we will
keep all the 15 features in the following stages. At the end of stage 4, we will
return to compare the performance of the algorithm with and without PCA.
As there is no significantly dominant PC, we will skip plotting the PCA here.

```
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_std)
print 'PCA explained_variance_ratio_', pca.explained_variance_ratio_
```

Output:
```
PCA explained_variance_ratio_ [ 0.34010581  0.12119602  0.104491    0.08764263  0.06768687  0.05239806
  0.0467082   0.04564431  0.03765439  0.03034863  0.02354492  0.01881022
  0.01624238  0.00752657  0.        ]
```
## Algorithm Selection
Here, we will use repeated nested cross validation to choose the machine
learning algorithm. We are using nested cross validation in order to test how
each algorithm perform towards unseen data (**Fig. 2**). Furthermore, we are
using repeated instead of unrepeated nested cross validation in order to avoid
any bias due to the different combination of training, validation, and test
sets. Detailed discussions on repeated cross validation can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3994246/pdf/1758-2946-6-10.pdf).

![Plot](https://github.com/lmarkely/enron_fraud/blob/master/Fig%202.png)

**Figure 2.** Schematic diagram of nested cross validation.

The following algorithms from scikit-learn will be evaluated.
1. Logistic Regression
2. Random Forest Classifier
3. K-Nearest Neighbors Classifier
4. Linear Support Vector Classifier
5. Kernel Support Vector Classifier
6. Multi-Layer Perceptron Classifier

### Logistic Regression
The nested cross validation for Logistic Regression is performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Logistic Regression
scores=[]
clf_lr = LogisticRegression(penalty='l2')
pipe_lr = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_lr]])
params_lr = {'clf__C':10.0**np.arange(-4,4)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = KFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = KFold(n_splits=5,shuffle=True,random_state=j)
        gs_lr = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                             cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_lr,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print 'CV Recall Score of Logistic Regression: %.3f +/- %.3f %s' %(np.mean(scores),
                                                               np.std(scores),
                                                               '%')
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV Recall Score of Logistic Regression: 0.240 +/- 0.204 %
Complete in 31.9 sec
```
### Logistic Regression
The nested cross validation for Logistic Regression is performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Logistic Regression
scores=[]
clf_lr = LogisticRegression(penalty='l2')
pipe_lr = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_lr]])
params_lr = {'clf__C':10.0**np.arange(-4,4)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = KFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = KFold(n_splits=5,shuffle=True,random_state=j)
        gs_lr = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                             cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_lr,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print 'CV Recall Score of Logistic Regression: %.3f +/- %.3f %s' %(np.mean(scores),
                                                               np.std(scores),
                                                               '%')
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV Recall Score of Logistic Regression: 0.240 +/- 0.204 %
Complete in 31.9 sec
```
### Logistic Regression
The nested cross validation for Logistic Regression is performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Logistic Regression
scores=[]
clf_lr = LogisticRegression(penalty='l2')
pipe_lr = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_lr]])
params_lr = {'clf__C':10.0**np.arange(-4,4)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = KFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = KFold(n_splits=5,shuffle=True,random_state=j)
        gs_lr = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                             cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_lr,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print 'CV Recall Score of Logistic Regression: %.3f +/- %.3f' %(np.mean(scores),
                                                             np.std(scores))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV Recall Score of Logistic Regression: 0.240 +/- 0.204
Complete in 32.3 sec
```

### Random Forest Classifier
The nested cross validation for Random Forest Classifier is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Random Forest Classifier
scores=[]
clf_rf = RandomForestClassifier(random_state=42)
pipe_rf = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_rf]])
params_rf = {'clf__n_estimators':np.arange(1,11)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = KFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = KFold(n_splits=5,shuffle=True,random_state=j)
        gs_rf = GridSearchCV(estimator=pipe_rf,param_grid=params_rf,
                             cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_rf,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print ('CV Recall Score of Random Forest Classifier: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV Recall Score of Random Forest Classifier: 0.170 +/- 0.231
Complete in 183.9 sec
```

### K-Nearest Neighbors Classifier
The nested cross validation for K-Nearest Neighbors Classifier is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

scores=[]
clf_knn = KNeighborsClassifier()
pipe_knn = Pipeline([['sc',StandardScaler()],
                     ['clf',clf_knn]])
params_knn = {'clf__n_neighbors':np.arange(1,6)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = KFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = KFold(n_splits=5,shuffle=True,random_state=j)
        gs_knn = GridSearchCV(estimator=pipe_knn,param_grid=params_knn,
                              cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_knn,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print ('CV Recall Score of KNN Classifier: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV Recall Score of KNN Classifier: 0.146 +/- 0.207
Complete in 24.6 sec
```
