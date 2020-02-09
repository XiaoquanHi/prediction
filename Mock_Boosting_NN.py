# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.tree import export_graphviz
import pydot
import pydotplus
from IPython.display import Image
import os
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt


f = open("mock_Boosting_NN", 'w+')

# Read in data and display first 5 rows
MockData = pd.read_csv('mock.csv', skiprows=3)
MockData.head(5)

print('The shape of the MockData is:', MockData.shape,file=f)

# To show the proportion of readmission, whether it is unbalanced
print('The proportion of readmission is: ', MockData['readm indicator'].mean(),file=f)

# Labels are the values we want to predict
labels = np.array(MockData['readm indicator'])

# Remove the labels from the features; axis 1 refers to the columns
features= MockData.drop(['mock ID','readm indicator','time to readmission (-1 for no readmitters)'], axis=1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
MockData = np.array(MockData)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)

print('Training Features Shape:', train_features.shape,file=f)
print('Training Labels Shape:', train_labels.shape,file=f)
print('Testing Features Shape:', test_features.shape,file=f)
print('Testing Labels Shape:', test_labels.shape,file=f)

# instantiate the model (using the default parameters)
ada = AdaBoostClassifier(n_estimators=100)

# fit the model with data
ada.fit(train_features, train_labels)
train_pred = ada.predict(train_features)
test_pred = ada.predict(test_features)

print('\n********AdaBoosting_Performance on the Training Set********',file=f)
print(confusion_matrix(train_labels,train_pred),file=f)
print(classification_report(train_labels,train_pred),file=f)
print(accuracy_score(train_labels,train_pred),file=f)

print('\n********AdaBoosting_Performance on the Test Set********',file=f)
print(confusion_matrix(test_labels,test_pred),file=f)
print(classification_report(test_labels,test_pred),file=f)
print(accuracy_score(test_labels,test_pred),file=f)


test_pred = ada.fit(train_features, train_labels).predict(test_features)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:], test_pred[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), test_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', label='ROC curve (area = %0.2f)' %(roc_auc[0]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of AdaBoosting')
plt.legend(loc="lower right")
plt.show()



gra = GradientBoostingClassifier(n_estimators=100)

# fit the model with data
gra.fit(train_features, train_labels)
train_pred = gra.predict(train_features)
test_pred = gra.predict(test_features)

print('\n********GradientBoosting_Performance on the Training Set********',file=f)
print(confusion_matrix(train_labels,train_pred),file=f)
print(classification_report(train_labels,train_pred),file=f)
print(accuracy_score(train_labels,train_pred),file=f)

nn = GradientBoostingClassifier(n_estimators=100)

# fit the model with data
gra.fit(train_features, train_labels)
train_pred = gra.predict(train_features)
test_pred = gra.predict(test_features)



print('\n********Gradient Boosting_Performance on the Training Set********',file=f)
print(confusion_matrix(train_labels,train_pred),file=f)
print(classification_report(train_labels,train_pred),file=f)
print(accuracy_score(train_labels,train_pred),file=f)

print('\n******** Gradient Boosting_Performance on the Test Set********',file=f)
print(confusion_matrix(test_labels,test_pred),file=f)
print(classification_report(test_labels,test_pred),file=f)
print(accuracy_score(test_labels,test_pred),file=f)

test_pred = gra.fit(train_features, train_labels).predict(test_features)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:], test_pred[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), test_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', label='ROC curve (area = %0.2f)' %(roc_auc[0]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Gradient Boosting')
plt.legend(loc="lower right")
plt.show()


nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1,)
# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#               beta_1=0.9, beta_2=0.999, early_stopping=False,
#               epsilon=1e-08, hidden_layer_sizes=(5, 2),
#               learning_rate='constant', learning_rate_init=0.001,
#               max_iter=200, momentum=0.9, n_iter_no_change=10,
#               nesterovs_momentum=True, power_t=0.5, random_state=1,
#               shuffle=True, solver='lbfgs', tol=0.0001,
#               validation_fraction=0.1, verbose=False, warm_start=False)

# fit the model with data
nn.fit(train_features, train_labels)
train_pred = nn.predict(train_features)
test_pred = nn.predict(test_features)

print('\n********Neural Network_Performance on the Training Set********',file=f)
print(confusion_matrix(train_labels,train_pred),file=f)
print(classification_report(train_labels,train_pred),file=f)
print(accuracy_score(train_labels,train_pred),file=f)

print('\n********Neural Network_Performance on the Test Set********',file=f)
print(confusion_matrix(test_labels,test_pred),file=f)
print(classification_report(test_labels,test_pred),file=f)
print(accuracy_score(test_labels,test_pred),file=f)


test_pred = nn.fit(train_features, train_labels).predict(test_features)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:], test_pred[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), test_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', label='ROC curve (area = %0.2f)' %(roc_auc[0]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Neural Network')
plt.legend(loc="lower right")
plt.show()



