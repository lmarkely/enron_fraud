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

The first stage will determine the features that will be included for machine
learning. Here, various data visualization techniques, including scatter plot
matrix and dimensionality reduction (Principal Component Analysis & Linear
Discriminant Analysis) will be used (stage 2). This visualization will help
identify if there is any redundancy among the features due to correlation,
as well as provide some estimation on how the data are distributed in
high dimensional space. After the features are selected and engineered, nested
cross validation will be used for algorithm selection in stage 3. Then, stage 4
identifies the hyperparameter values that give the best result for the selected
algorithm. Three metrics, F1 score, precision, and recall will be used during
the algorithm and model selection.
