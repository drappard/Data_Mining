import quandl, math
import numpy as np
import pandas as pd
import sklearn
import graphviz
import pydotplus

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#import dataset
df = pd.read_csv("dataset.csv")
print(df.head(), '\n')

#print the dataset information
print ("Number of Instances: ", len(df), '\n')
print ("Number of Attributes: ", len(df.columns), '\n')

#use built in functions to get dataset details
print(df.describe(), '\n')
print(df.info(), '\n')

obj_df = df.select_dtypes(include=['object']).copy()

#use dictionary to encode binary-attribute data
encode = {"Student":  {"No": 0, "Yes": 1},
"Credit":   {"Fair": 0, "Excellent": 1},
"Buy_Computer":     {"No": 0, "Yes": 1}}

#use one hot encoding to encode multi-attribute data
obj_df = pd.get_dummies(obj_df, columns=["Age","Income"])
obj_df.replace(encode, inplace=True)

print(obj_df, '\n')

#create variable for features, x, and labels, y
X = np.array(obj_df.drop(['Buy_Computer'], 1))
y = np.array(obj_df['Buy_Computer'])

#split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state = 100)

#build the entropy decision tree
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 4, min_samples_leaf = 1)
clf_entropy.fit(X_train, y_train)

#build the gini decision tree
clf_gini = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 4, min_samples_leaf = 1)
clf_gini.fit(X_train, y_train)

#print the predicted class values
y_predict = clf_gini.predict(X_test)
print("Predicted values:",y_predict, '\n')

print("Class Train Set:",y_train, '\n')
print("Class Test Set:",y_test, '\n')

#confusion matrix and accuracy information for the build
print("Confusion Matrix: ", '\n', confusion_matrix(y_test,y_predict), '\n')
print("Accuracy : ", accuracy_score(y_test,y_predict)*100, "%", '\n')
print("Classification Report : ", '\n', classification_report(y_test, y_predict), '\n')

#view and export the entropy decision tree to a pdf
dot_data = tree.export_graphviz(clf_gini, out_file=None,
    feature_names=['Age_Youth','Age_Adult','Age_Senior','Income_High','Income_Medium','Income_Low','Student','Credit'],
    class_names=['BUY_COMPUTER:NO','BUY_COMPUTER:YES'], filled=True,
    rounded=True,special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("Entropy", view = True)

#view and export the gini decision tree to a pdf
dot_data = tree.export_graphviz(clf_gini, out_file=None,
    feature_names=['Age_Youth','Age_Adult','Age_Senior','Income_High','Income_Medium','Income_Low','Student','Credit'],
    class_names=['BUY_COMPUTER:NO','BUY_COMPUTER:YES'], filled=True,
    rounded=True,special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("Gini Index", view = True)
