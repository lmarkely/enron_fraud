# Enron Fraud Project

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
keep all the 15 features in the following stages. Since there is no
significantly dominant PC, we will skip plotting the PCA here.

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
6. Naive Bayes
7. Multi-Layer Perceptron Classifier
8. AdaBoost Classifier

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
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_lr = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                             cv=k_fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_lr,X,y,cv=k_fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of Logistic Regression: %.3f +/- %.3f'
       %(np.mean(scores),np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_lr = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                             cv=k_fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_lr,X,y,cv=k_fold_outer,
                                      scoring='precision'))
print ('CV Precision Score of Logistic Regression: %.3f +/- %.3f'
       %(np.mean(scores),np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_lr = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                             cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_lr,X,y,cv=k_fold_outer,
                                      scoring='recall'))

print ('CV Recall Score of Logistic Regression: %.3f +/- %.3f'
       %(np.mean(scores),np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV F1 Score of Logistic Regression: 0.260 +/- 0.199
Complete in 32.9 sec
CV Precision Score of Logistic Regression: 0.250 +/- 0.229
Complete in 31.0 sec
CV Recall Score of Logistic Regression: 0.283 +/- 0.238
Complete in 29.5 sec
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
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_rf = GridSearchCV(estimator=pipe_rf,param_grid=params_rf,
                             cv=k_fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_rf,X,y,cv=k_fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of Random Forest Classifier: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_rf = GridSearchCV(estimator=pipe_rf,param_grid=params_rf,
                             cv=k_fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_rf,X,y,cv=k_fold_outer,
                                      scoring='precision'))
print ('CV Precision Score of Random Forest Classifier: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
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
CV F1 Score of Random Forest Classifier: 0.219 +/- 0.226
Complete in 184.3 sec
CV Precision Score of Random Forest Classifier: 0.241 +/- 0.280
Complete in 366.0 sec
CV Recall Score of Random Forest Classifier: 0.233 +/- 0.268
Complete in 551.9 sec
```

### K-Nearest Neighbors Classifier
The nested cross validation for K-Nearest Neighbors Classifier is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#KNN Classifier
scores=[]
clf_knn = KNeighborsClassifier()
pipe_knn = Pipeline([['sc',StandardScaler()],
                     ['clf',clf_knn]])
params_knn = {'clf__n_neighbors':np.arange(1,6)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_knn = GridSearchCV(estimator=pipe_knn,param_grid=params_knn,
                              cv=k_fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_knn,X,y,cv=k_fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of KNN Classifier: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedStratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_knn = GridSearchCV(estimator=pipe_knn,param_grid=params_knn,
                              cv=k_fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_knn,X,y,cv=k_fold_outer,
                                      scoring='precision'))
print ('CV Precision Score of KNN Classifier: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
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
CV F1 Score of KNN Classifier: 0.203 +/- 0.208
Complete in 22.6 sec
CV Precision Score of KNN Classifier: 0.222 +/- 0.268
Complete in 24.4 sec
CV Recall Score of KNN Classifier: 0.221 +/- 0.250
Complete in 24.1 sec
```

### Linear SVC
The nested cross validation for Linear SVC is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Linear SVC
scores=[]
clf_svc = SVC()
pipe_svc = Pipeline([['sc',StandardScaler()],
                     ['clf',clf_svc]])
params_svc = {'clf__C':10.0**np.arange(-4,4)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_svc = GridSearchCV(estimator=pipe_svc,param_grid=params_svc,
                              cv=k_fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_svc,X,y,cv=k_fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of Linear SVC: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_svc = GridSearchCV(estimator=pipe_svc,param_grid=params_svc,
                              cv=k_fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_svc,X,y,cv=k_fold_outer,
                                      scoring='precision'))
print ('CV Precision Score of Linear SVC: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_svc = GridSearchCV(estimator=pipe_svc,param_grid=params_svc,
                              cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_svc,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print ('CV Recall Score of Linear SVC: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV F1 Score of Linear SVC: 0.157 +/- 0.186
Complete in 31.8 sec
CV Precision Score of Linear SVC: 0.169 +/- 0.223
Complete in 29.9 sec
CV Recall Score of Linear SVC: 0.169 +/- 0.215
Complete in 29.6 sec
```

### Kernel SVC
The nested cross validation for Kernel SVC is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Kernel SVC
scores=[]
clf_ksvc = SVC(kernel='rbf')
pipe_ksvc = Pipeline([['sc',StandardScaler()],
                     ['clf',clf_ksvc]])
params_ksvc = {'clf__C':10.0**np.arange(-4,4),'clf__gamma':10.0**np.arange(-4,4)}
t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_ksvc = GridSearchCV(estimator=pipe_ksvc,param_grid=params_ksvc,
                               cv=k_fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_ksvc,X,y,cv=k_fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of Kernel SVC: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_ksvc = GridSearchCV(estimator=pipe_ksvc,param_grid=params_ksvc,
                               cv=k_fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_ksvc,X,y,cv=k_fold_outer,
                                      scoring='precision'))
print ('CV Precision Score of Kernel SVC: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        k_fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_ksvc = GridSearchCV(estimator=pipe_ksvc,param_grid=params_ksvc,
                               cv=k_fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_ksvc,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print ('CV Recall Score of Kernel SVC: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV F1 Score of Kernel SVC: 0.208 +/- 0.215
Complete in 250.2 sec
CV Precision Score of Kernel SVC: 0.233 +/- 0.290
Complete in 499.9 sec
CV Recall Score of Kernel SVC: 0.231 +/- 0.269
Complete in 760.1 sec
```

### Naive Bayes
As there is no regularization parameter for Naive Bayes, we will simply use
cross validation.
```
#Set the number of repeats of the cross validation
N_outer = 5

#Naive Bayes
scores=[]
clf_nb = GaussianNB()
pipe_nb = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_nb]])
t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    scores.append(cross_val_score(pipe_nb,X,y,cv=k_fold_outer,
                                      scoring='f1'))
print 'CV F1 Score of Logistic Regression: %.3f +/- %.3f' %(np.mean(scores),
                                                               np.std(scores))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    scores.append(cross_val_score(pipe_nb,X,y,cv=k_fold_outer,
                                      scoring='precision'))
print 'CV Precision Score of Logistic Regression: %.3f +/- %.3f' %(np.mean(scores),
                                                               np.std(scores))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    k_fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    scores.append(cross_val_score(pipe_nb,X,y,cv=k_fold_outer,
                                      scoring='recall'))
print 'CV Recall Score of Logistic Regression: %.3f +/- %.3f' %(np.mean(scores),
                                                               np.std(scores))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV F1 Score of Naive Bayes: 0.257 +/- 0.062
Complete in 0.1 sec
CV Precision Score of Naive Bayes: 0.205 +/- 0.075
Complete in 0.1 sec
CV Recall Score of Naive Bayes: 0.430 +/- 0.339
Complete in 0.1 sec
```

### Multi-Layer Perceptron Classifier
The nested cross validation for Multi-Layer Perceptron is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#MLP Classifier
scores=[]
clf_mlp = MLPClassifier(solver='lbfgs')
pipe_mlp = Pipeline([['sc',StandardScaler()],
                     ['clf',clf_mlp]])
params_mlp = {'clf__activation':['logistic','relu'],'clf__alpha':10.0**np.arange(-4,4)}
t0 = time()
for i in range(N_outer):
    fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_mlp = GridSearchCV(estimator=pipe_mlp,param_grid=params_mlp,
                               cv=fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_mlp,X,y,cv=fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of MLP: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_mlp = GridSearchCV(estimator=pipe_mlp,param_grid=params_mlp,
                               cv=fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_mlp,X,y,cv=fold_outer,
                                      scoring='precision'))
print ('CV Precision of MLP: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_mlp = GridSearchCV(estimator=pipe_mlp,param_grid=params_mlp,
                               cv=fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_mlp,X,y,cv=fold_outer,
                                      scoring='recall'))
print ('CV Recall Score of MLP: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV F1 Score of MLP: 0.187 +/- 0.184
Complete in 760.5 sec
CV Precision of MLP: 0.204 +/- 0.219
Complete in 589.0 sec
CV Recall Score of MLP: 0.208 +/- 0.215
Complete in 871.1 sec
```
### AdaBoost Classifier
The nested cross validation for Multi-Layer Perceptron is
performed as follows.
```
#Set the number of repeats of the cross validation
N_outer = 5
N_inner = 5

#Kernel SVC
scores=[]
clf_ada = AdaBoostClassifier(random_state=42)
pipe_ada = Pipeline([['sc',StandardScaler()],
                     ['clf',clf_ada]])
params_ada = {'clf__n_estimators':np.arange(1,11)*10}
t0 = time()
for i in range(N_outer):
    fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_ada = GridSearchCV(estimator=pipe_ada,param_grid=params_ada,
                               cv=fold_inner,scoring='f1')
        scores.append(cross_val_score(gs_ada,X,y,cv=fold_outer,
                                      scoring='f1'))
print ('CV F1 Score of AdaBoost: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_ada = GridSearchCV(estimator=pipe_ada,param_grid=params_ada,
                               cv=fold_inner,scoring='precision')
        scores.append(cross_val_score(gs_ada,X,y,cv=fold_outer,
                                      scoring='precision'))
print ('CV F1 Score of AdaBoost: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)

t0 = time()
for i in range(N_outer):
    fold_outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=i)
    for j in range(N_inner):
        fold_inner = StratifiedKFold(n_splits=5,shuffle=True,random_state=j)
        gs_ada = GridSearchCV(estimator=pipe_ada,param_grid=params_ada,
                               cv=fold_inner,scoring='recall')
        scores.append(cross_val_score(gs_ada,X,y,cv=fold_outer,
                                      scoring='recall'))
print ('CV F1 Score of AdaBoost: %.3f +/- %.3f'
       %(np.mean(scores), np.std(scores)))
print 'Complete in %.1f sec' %(time()-t0)
```
Output:
```
CV F1 Score of AdaBoost: 0.269 +/- 0.222
Complete in 40.0 sec
CV F1 Score of AdaBoost: 0.224 +/- 0.227
Complete in 40.0 sec
CV F1 Score of AdaBoost: 0.233 +/- 0.226
Complete in 40.8 sec
```
The above results show that Logistic Regression has the best combination of F1
score, precision, and recall. Gaussian Naive Bayes has higher recall score than
Logistic Regression, but it has lower precision score. Since, precision and
recall scores of Logistic regression cover 0.3, minimum criterion for this
project, Logistic Regression is chosen for model selection. Moreover, similar
pipeline consisting of StandardScaler, PCA, and classifier was assessed in
parallel, but did not perform better than the current setup. These data are not
presented here, but can be found in the Jupyter Notebook
'Enron_fraud-PCA.ipynb'. Similarly, a pipeline consisting of MinMaxScaler,
SelectKBest, and classifier also did not perform better than the current setup.
These data can be found in the Jupyter Notebook 'Enron_fraud-SKB.ipynb'. As an
alternative to StratifiedKFold with shuffle, StratifiedShuffleSplit was
evaluated. Similar to the other alternatives, this option did not perform better
than the current setup. Thus, the current setup with Logistic Regression is
chosen for the next step.

## Model Selection
In model selection, repeated cross validation is used to select the optimum
hyperparameter value, 'C', based on F1 score, precision, and recall as shown
in the followings.
```
from IPython.core.display import display
#Model selection based on F1 Score
n_reps = 1000
best_params = []

clf_lr = LogisticRegression(penalty='l2')
pipe_lr = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_lr]])
params_lr = {'clf__C':10.0**np.arange(-4,4)}

for rep in np.arange(n_reps):
    k_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=rep)
    gs_lr_cv = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                            cv=k_fold,scoring='f1')
    gs_lr_cv = gs_lr_cv.fit(X,y)
    best_param = gs_lr_cv.best_params_
    best_param.update({'Best Score': gs_lr_cv.best_score_})
    best_params.append(best_param)

#DataFrame summarizing average of best scores, frequency for each best
#parameter value
best_params_df = pd.DataFrame(best_params)
best_params_df = best_params_df.rename(columns={'clf__C':'C'})
best_params_df = best_params_df.groupby('C')['Best Score'].describe()
best_params_df = \
np.round(best_params_df,decimals=2).sort_values(['mean','count'],axis=0,
                                                ascending=[False,False])
display(best_params_df)

# Model selection based on precision
n_reps = 1000
best_params = []

clf_lr = LogisticRegression(penalty='l2')
pipe_lr = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_lr]])
params_lr = {'clf__C':10.0**np.arange(-4,4)}

for rep in np.arange(n_reps):
    k_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=rep)
    gs_lr_cv = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,
                            cv=k_fold,scoring='precision')
    gs_lr_cv = gs_lr_cv.fit(X,y)
    best_param = gs_lr_cv.best_params_
    best_param.update({'Best Score': gs_lr_cv.best_score_})
    best_params.append(best_param)

#DataFrame summarizing average of best scores, frequency for each
#best parameter value
best_params_df = pd.DataFrame(best_params)
best_params_df = best_params_df.rename(columns={'clf__C':'C'})
best_params_df = best_params_df.groupby('C')['Best Score'].describe()
best_params_df = \
np.round(best_params_df,decimals=2).sort_values(['mean','count'],axis=0,
                                                ascending=[False,False])
display(best_params_df)

# Model selection based on recall
n_reps = 1000
best_params = []

clf_lr = LogisticRegression(penalty='l2')
pipe_lr = Pipeline([['sc',StandardScaler()],
                    ['clf',clf_lr]])
params_lr = {'clf__C':10.0**np.arange(-4,4)}

for rep in np.arange(n_reps):
    k_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=rep)
    gs_lr_cv = GridSearchCV(estimator=pipe_lr,param_grid=params_lr,cv=k_fold,
                            scoring='recall')
    gs_lr_cv = gs_lr_cv.fit(X,y)
    best_param = gs_lr_cv.best_params_
    best_param.update({'Best Score': gs_lr_cv.best_score_})
    best_params.append(best_param)

#DataFrame summarizing average of best scores, frequency for each best
#parameter value
best_params_df = pd.DataFrame(best_params)
best_params_df = best_params_df.rename(columns={'clf__C':'C'})
best_params_df = best_params_df.groupby('C')['Best Score'].describe()
best_params_df = \
np.round(best_params_df,decimals=2).sort_values(['mean','count'],axis=0,
                                                ascending=[False,False])
display(best_params_df)
```

The results of the hyperparameter C tuning is presented in **Table 1**.The first column corresponds to the best C values selected in 1000 permutations of data splitting. The second column corresponds to the number of times each C value was
selected as the best hyperparameter value. The third column, corresponds to the
mean of either F1 score, precision, or recall when the corresponding C value was
chosen. Similary, the other columns correspond to the standard deviation,
minimum, 25% quantile, 50% quantile, 75% quantile, and maximum of the metrics
scores when the corresponding C value was chosen. This summary shows
that C = 0.0001 gives the best F1 score and recall. Although, this
hyperparameter value does not give the best precision, the mean of precision is
above 0.3 and it is selected as the best parameters in significant number of
permutations of data splitting.

**Table 1.** Summary of model selection for Logistic Regression.
![Plot](https://github.com/lmarkely/enron_fraud/blob/master/Fig%203.png)

## Udacity Project Questions
**Q:** Summarize for us the goal of this project and how machine learning is useful
in trying to accomplish it. As part of your answer, give some background on the
dataset and how it can be used to answer the project question. Were there
any outliers in the data when you got it, and how did you handle those?

**A:** The goal is to build a classifier to predict the employees who are
involved in Enron fraud case given the financial and email data. The input data
are these dataset, and the output data is whether the employee may
be involved oin the fraud case. Some characteristics about the dataset is
provided by poi_id.py and is summarized below. There are a few outliers.
One of them is the data for 'Total', which sums the
data of everyone in the dataset. This was removed from the analysis. While the
other outliers are not removed as they may correspond to those who are involved
in the fraud.
```
Total number of data points: 144
Total number of features: 15
Total number of data points with all zeros: 1
Total number of missing feature values: 1352
Total number of POI: 18
Total number of non-POI 126
```
**Q:** What features did you end up using in your POI identifier, and what
selection process did you use to pick them? Did you have to do any scaling?
Why or why not? As part of the assignment, you should attempt to engineer your
own feature that does not come ready-made in the dataset -- explain what feature
you tried to make, and the rationale behind it. (You do not necessarily have to
use it in the final analysis, only engineer and test it.) In your feature
selection step, if you used an algorithm like a decision tree, please also give
the feature importances of the features that you use, and if you used an
automated feature selection function like SelectKBest, please report the feature
scores and reasons for your choice of parameter values.

**A:** The features used are 'salary', 'deferral_payments','loan_advances',
'bonus', 'restricted_stock_deferred', 'deferred_income', 'expenses',
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','shared_receipt_with_poi', 'std_from_poi','std_to_poi'. The
rationale is described in detail above. Briefly, features that are sum of other
features are removed to minimize redundancy and correlation among features. In
addition, email address is removed as we are interested in quantitative input
data. Furthermore, new features, 'std_from_poi' and 'std_to_poi', are created
by dividing received and sent email messages to poi by total received and sent
messages. Decision Tree was not used as it is prone to overfitting. Instead,
Random Forest, which is a bagging version of Decision Tree ([ref](https://sebastianraschka.com/faq/docs/bagging-boosting-rf.html)) was
used in an attempt to avoid overfitting. SelectKBest was not used as it did not
improved the performance of the algorithm in this case.

**Q:** What algorithm did you end up using? What other one(s) did you try?
How did model performance differ between algorithms?

**A:** Logistic regression was chosen. Other algorithms evaluated include
KNNClassifier, RandomForestClassifier, AdaBoostClassifier, Linear SVC, Kernel
SVC, MLPClassifier, and GaussianNB. Logistic Regression has the best combination
of F1 score, precision, and recall. GaussianNB has higher recall score than
Logistic Regression during algorithm selection, but it has lower precision
score. The other algorithms have lower F1 score, precision, and recall than
Logistic Regression for this dataset.

**Q:** What does it mean to tune the parameters of an algorithm, and what can
happen if you don’t do this well?  How did you tune the parameters of your
particular algorithm? What parameters did you tune? (Some algorithms do not
have parameters that you need to tune -- if this is the case for the one you
picked, identify and briefly explain how you would have done it for the model
that was not your final choice or a different model that does utilize parameter
tuning, e.g. a decision tree classifier).

**A:** The parameters tuned in this project are regularization parameters. These
parameters control the complexity of the algorithm. If we do not tune it well,
the algorithm may suffer from high bias (model is unable to capture and fit the
complexity of the data) or high variance (model is overly complicated and can't
generalize to unseen dataset). The parameters were tuned using nested cross
validation, in which the data set are split into training, validation, and test
set. Training and validation sets are used by GridSearchCV to obtain the best
parameters, and test set is used to test the generalization of the algorithm.
This tuning is iterated over different splitting of training, validation, and
test sets.

**Q:** What is validation, and what’s a classic mistake you can make if you do
it wrong? How did you validate your analysis?

**A:** In validation, we test the model to confirm that it can fit unseen data
at a reasonable rate. The is repeated nested cross validation as described in the
previous question, or using simpler methods, such as nested cross validation or
cross validation without repeat. There are also variation in the way the
algorithm splits the data, such as KFold, StratifiedShuffleSplit,
StratifiedKFold, etc. To validate our analysis, metrics, such as accuracy, F1
score, precision, recall, etc are used to obtain quantitative assessment on the
algorithm.

**Q:** Give at least 2 evaluation metrics and your average performance for each
of them.  Explain an interpretation of your metrics that says something
human-understandable about your algorithm’s performance.

**A:** F1 score, precision, and recall are used as the metrics. The average
performance for F1 score is 0.33, precision is 0.34, and recall is 0.38. Recall
of 0.38 means that given a group of suspects being evaluated,
the algorithm can correctly predict 0.38 of the total number of suspects who are
indeed POI (Person of Interest). Precision of 0.34 means that from all the
suspects that the algorithm predicts as POI, 0.34 of those people are indeed
POI. Algorithms that have high precision may tend to have low recall and vice
versa. To balance these two metrics, we can use F1 score, which is a harmonic
mean of these two metrics. Out of curiosity and just as a back of the envelope
calculation, I would like to assess how good this recall value compared with
random coin toss. If we were to predict the POI by random coin toss, the
probability that we will do better than this algorithm is approximately

![Plot](https://github.com/lmarkely/enron_fraud/blob/master/Calculation.png)

which is significantly lower than 0.38. 

## References
* Python Machine Learning by Sebasitan Raschka
([URL](https://sebastianraschka.com/books.html)):
reference for implementing nested cross validation and pipeline,
as well as Machine Learning in general.
* https://sebastianraschka.com/faq/docs/bagging-boosting-rf.html: reference for
bagging, bossting, and Random Forest.
* [Scikit-learn](http://scikit-learn.org/):
reference for implementing machine learning algorithms.
* [Seaborn](http://seaborn.pydata.org/generated/seaborn.pairplot.html):
reference for implementing Seaborn.
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3994246/pdf/1758-2946-6-10.pdf:
reference for repeated nested cross validation and repeated cross validation.
* (https://en.wikipedia.org/wiki/Enron_scandal): reference for Enron scandal.

**“I hereby confirm that this submission is my work.
I have cited above the origins of any parts of the submission that were taken
from Websites, books, forums, blog posts, github repositories, etc."**

## Notes
* poi_id.py should be used to generate my_classifier.pkl, my_dataset.pkl,
my_feature_list.pkl, as well as the answers to the questions listed in the
following section.
* poi_id_modified.py provides the codes for all the steps described in details
below.
* Environment file: enron_fraud.yaml
