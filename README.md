# Enron Fraud Project

![Plot](Enron_Complex.jpg)
Enron Complex in Houston - [Source](https://www.flickr.com/photos/23094783@N03)

Enron was one of the largest companies in the US. It went into bankruptcy due to corporate fraud. A significant amount of Enron data (emails
and financial data) were entered into public record as a result of Federal
investigation. This project aims to build a classifier that can
predict Enron employees involved fraud based on the public Enron
financial and email dataset. More details about Enron scandal can be found on
[Wikipedia](https://en.wikipedia.org/wiki/Enron_scandal).

## Workflow
This project is divided into 3 main stages:
1. Feature selection and engineering
2. Algorithm selection
3. Model selection

## Feature Selection and Engineering
First, the data are cleaned up; the data corresponding to 'TOTAL' and
'THE TRAVEL AGENCY IN THE PARK' are removed as we are interested in the data of individuals. In addition, 'LOCKHART EUGENE E' data have all zero feature values
and is also removed.

Some features are also removed. 'total_payments' and 'total_stock_values' are removed
as they are aggregate of other features. Moreover, 'to_messages',
'email_address', 'from_poi_to_this_person', 'from_messages', and 'from_this_person_to_poi' are replaced by two engineered features:
'std_from_poi' and 'std_to_poi', obtained from 'from_poi_to_this_person'/
'from_messages' and 'from_this_person_to_poi'/'to_messages', respectively.

In addition, features that have more than 70 zero values
are removed. These features
include 'deferral_payments', 'long_term_incentive', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', and 'director_fees'.
70 was chosen as the cut-off value because there are 141 data points in this
data set after the above feature selection and engineering. After these
feature selection and engineering, there are a total of 9 features.

Furthermore, PCA and SelectKBest are evaluated for dimensionality reduction and
feature selection methods. In this case, SelectKBest is chosen due to better
performance in algorithm selection. Detailed algorithm selection without PCA and
SelectKBest can be found in 'Enron_fraud.ipynb', with SelectKBest in
'Enron_fraud_SKB.ipynb', and with PCA in 'Enron_fraud_PCA.ipynb'. The F score
and p-value from SelectKBest suggest that only 'exercised_stock_options' and
'bonus' are significant, while others with p-value > 0.05 and F score
significantly lower than these two features will not be used for algorithm
selection and model selection.

![Plot](Fig%201.png)

**Figure 1.** F score and p-value from SelectKBest.

## Algorithm Selection
Repeated nested cross validation is used for algorithm selection (**Fig. 2**).
Furthermore, repeated instead of unrepeated nested cross validation is used
to minimize influence from different splitting of training, validation, and test
sets on algorithm selection as described in a [previous study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3994246/pdf/1758-2946-6-10.pdf).

![Plot](Fig%202.png)

**Figure 2.** Schematic diagram of nested cross validation.

The following algorithms from scikit-learn are evaluated.
1. Logistic Regression
2. Random Forest Classifier
3. K-Nearest Neighbors Classifier
4. Linear Support Vector Classifier
5. Kernel Support Vector Classifier
6. Naive Bayes
7. Multi-Layer Perceptron Classifier
8. AdaBoost Classifier

**Figs. 3-5** summarize the F1 score, precision, and recall of all algorithms.
There are 3 types of pipelines evaluated. The first includes StandardScaler and
the classifier (**Fig. 3**). The second includes StandardScaler, PCA, and the
classifier (**Fig. 4**). The third includes MinMaxScaler, SelectKBest, and the
clasifier. Overall, KNN Classifier with the third pipeline has the best average
F1 score (0.35), precision (0.43), and recall (0.4) in the repeated nested
cross validation.

![Plot](Fig%203.png)

**Figure 3.** F1 score, precision, and recall in algorithm selection without
PCA or SelectKBest.

![Plot](Fig%204.png)

**Figure 4.** F1 score, precision, and recall in algorithm selection with
PCA.

![Plot](Fig%205.png)

**Figure 5.** F1 score, precision, and recall in algorithm selection with
SelectKBest.

## Model Selection

The hyperparameter of KNN Classifier is the number of neighbors. This parameter
is tuned in the model selection (**Table 1**).
The first column corresponds to the best number of neighbors selected in 1000
permutations of data splitting. The second column corresponds to the number of
times each N value was selected as the best value. The third column corresponds
to the mean of either F1 score, precision, or recall when the corresponding N
value was chosen. Similary, the other columns correspond to the standard
deviation, minimum, 25% quantile, 50% quantile, 75% quantile, and maximum of the
metrics scores when the corresponding N value was chosen. This summary shows
that N = 5 gives the optimum combination of F1 score (0.41), precision (0.65),
and recall (0.36).

**Table 1.** Summary of KNN Classifier model selection.
![Plot](Table%201.png)

## Comparison with Dummy Classifier
To assess how significant these results are, the performance is compared with
Dummy Classifier with 'uniform' strategy from sklearn. The Dummy Classifier has
overall F1 score, precision, and recall around 0.16 - 0.27. So, the performance
of the KNN is significantly better than random guessing.

## Udacity Project Questions
**Q:** Summarize for us the goal of this project and how machine learning is useful
in trying to accomplish it. As part of your answer, give some background on the
dataset and how it can be used to answer the project question. Were there
any outliers in the data when you got it, and how did you handle those?

**A:** The goal is to build a classifier to predict the employees
involved in Enron fraud case (POI) given the financial and email data.
Some characteristics about the dataset is provided by poi_id.py and is
summarized below.

```
Total number of data points: 141
Total number of features: 9 before SelectKBest, 2 after SelectKBest
Total number of data points with all zeros: 3 (after feature selection & engineering)
Total number of missing feature values: 1334 (after feature selection & engineering)
Total number of POI: 18
Total number of non-POI 123
```

This summary shows that the dataset is imbalanced; there are much less POI than
non-POI. Due to this imbalance, StratifiedKFold is used to ensure similar
distribution of POI vs non-POI in training, validation, and test sets. In
addition, a few outliers that were removed
include 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK', and 'LOCKHART EUGENE E'. The
first is the sum of all data, the second is not a person, and the third has all
zero values. Moreover, the data also have features with a lot of missing values.
The followings are summary of number of missing values for each feature.

```
'loan_advances': 141
'director_fees': 128
'restricted_stock_deferred': 127
'deferral_payments': 106
'deferred_income': 96
'long_term_incentive': 79
'bonus': 63
'shared_receipt_with_poi': 58
'from_poi_to_this_person': 58
'to_messages': 58
'from_messages': 58
'from_this_person_to_poi': 58
'salary': 50
'other': 53
'expenses': 50
'exercised_stock_options': 43
'restricted_stock': 35
'email_address': 33
'total_payments': 21
'total_stock_value': 19
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

**A:** The features used are 'exercised_stock_options' and 'bonus'.
In feature selection process, features that are sum of other
features are removed to minimize redundancy and correlation among features. In
addition, email address is removed as we are interested in quantitative input
data. Furthermore, new features, 'std_from_poi' and 'std_to_poi', are created
by dividing received and sent email messages to poi by total received and sent
messages.

Feature scaling was used because each feature has different value ranges. Without
scaling, features with high value and/or variance may dominate over features
with low value and/or variance. For SelectKBest, MinMaxScaler was used for
feature scaling. Feature scores from SelectKBest is also provided in **Fig.1**.
Features with p-value < 0.05 were selected for algorithm and model selection.
The figure also shows that the new engineered features 'std_from_poi' and
'std_to_poi' have significantly lower F Score (1.7 and 1.2) than the top two
features (6.7 and 5.0). Moreover, the p-value of these engineered features are
very high at 0.19 and 0.27, suggesting that these features may not be
significant in the classification. Decision Tree was not used as it is prone to
overfitting. Instead,Random Forest, which is a bagging version of Decision Tree ([ref](https://sebastianraschka.com/faq/docs/bagging-boosting-rf.html)) and
AdaBoostClassifier with Decision Tree as the base classifier, a boosting version of
Decision Tree were used in an attempt to avoid overfitting.

**Q:** What algorithm did you end up using? What other one(s) did you try?
How did model performance differ between algorithms?

**A:** KNN Classifier was chosen. Other algorithms evaluated include
Logistic Regression, RandomForestClassifier, AdaBoostClassifier, Linear SVC, Kernel
SVC, MLPClassifier, and GaussianNB. KNN Classifier has the highest F1 score,
precision, and recall among all algorithms evaluated. There is no clear pattern
among different algorithm performance. One interesting observation is that
implementing SelectKBest significantly improves the performance of most of the
algorithms, except Logistic Regression and Linear SVC. The performance of KNN
is significantly improved by SelectKBest.

**Q:** What does it mean to tune the parameters of an algorithm, and what can
happen if you don’t do this well?  How did you tune the parameters of your
particular algorithm? What parameters did you tune? (Some algorithms do not
have parameters that you need to tune -- if this is the case for the one you
picked, identify and briefly explain how you would have done it for the model
that was not your final choice or a different model that does utilize parameter
tuning, e.g. a decision tree classifier).

**A:** The parameters tuned in this project are regularization parameters. These
parameters control the complexity of the algorithm. If we do not tune it well,
the algorithm may suffer from high bias (underfitting) or high variance
(overfitting). The parameters were tuned using repeated nested cross
validation, in which the data set are split into training, validation, and test
set. Training and validation sets are used by GridSearchCV to obtain the best
parameters, and test set is used to test the generalization of the algorithm.
This tuning is iterated over different splitting of training, validation, and
test sets.

**Q:** What is validation, and what’s a classic mistake you can make if you do
it wrong? How did you validate your analysis?

**A:** In cross validation, we split the dataset into test set and training set.
The model is trained on the training set and then tested on test set. This
splitting is done so that the model never sees the test set while training. It
is critical that there is no leaking of information from the test set to the
training of the model. Otherwise, we will have an overfitting model. The classic
mistake is to train and test the model on the same set of data. We may get high
performance score, but poor performance score when we use the model on a
completely new dataset. This is an example of overfitting problem. To validate
our analysis, we use metrics, such as accuracy, F1
score, precision, recall, etc to obtain quantitative assessment on the
algorithm performance.

**Q:** Give at least 2 evaluation metrics and your average performance for each
of them.  Explain an interpretation of your metrics that says something
human-understandable about your algorithm’s performance.

**A:** F1 score, precision, and recall are used as the metrics. The average
performance for F1 score is 0.41, precision is 0.65, and recall is 0.36. Recall
of 0.36 means that given a group of suspects being evaluated,
the algorithm can correctly predict 0.36 of the total number of suspects who are
indeed POI (Person of Interest). Precision of 0.65 means that from all the
suspects that the algorithm predicts as POI, 0.65 of those people are indeed
POI. Algorithms that have high precision may tend to have low recall and vice
versa. To balance these two metrics, we can use F1 score, which is a harmonic
mean of these two metrics.

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
* The Jupyter notebooks provide the codes for all the steps described in details
above.
* Environment file: enron_fraud.yaml
