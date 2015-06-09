#!/usr/local/bin/python

import pandas
import re
import numpy as np
from scipy.stats import mode
from sklearn import tree
import pydot


############################ Data cleaning methods ############################
def clean_data_embarked(df):
  """Clean the embarked column. Replace missing values with the most common."""
  
  df['Embarked'] = df['Embarked'].fillna(mode(df['Embarked'])[0][0])


def clean_data_fare(df):
  """Clean the fare column. Replace null values with the average fare."""

  df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


def clean_data_age(df):
  """Clean the age column. Replace null values with the average age.""" 

  df['Age'] = df['Age'].fillna( df['Age'].mean())


############################ Data factoring methods ############################
def factorize_data_sex(df):
  """Clean the sex column. Replace male = 0 and female = 1."""

  df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)  


def factorize_data_embarked(df):
  """Replace the letters with numbers."""

  embarked_codes = {'S': 0, 'C': 1, 'Q': 2}
  df['Embarked'] = df['Embarked'].apply(lambda x: embarked_codes[x])


############################ Utility methods ############################
def write_image_of_decision_tree(filename, decision_tree, columns):
  """Create an image visualising a decision tree."""

  dot_filename = str(filename) + '.dot'
  dotfile = open(dot_filename, 'w')
  tree.export_graphviz(decision_tree, out_file = dotfile, feature_names = columns)
  dotfile.close()
  graph = pydot.graph_from_dot_file(dot_filename)
  graph.write_png(str(filename) + '.png')


############################ Main method ############################

# Load the training data
train_data = pandas.read_csv('train.csv', header = 0)
test_data = pandas.read_csv('test.csv', header = 0)

# Clean the training data
clean_data_embarked(train_data)
clean_data_embarked(test_data)
clean_data_age(train_data)
clean_data_age(test_data)
clean_data_fare(train_data)
clean_data_fare(test_data)

# Make categorical values into factors
factorize_data_sex(train_data)
factorize_data_sex(test_data)
factorize_data_embarked(train_data)
factorize_data_embarked(test_data)

# Now make a decision tree
columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
labels = train_data['Survived'].values
features = train_data[list(columns)].values

# Fit our training data to a decision tree model
dtree = tree.DecisionTreeClassifier(random_state=415, min_samples_leaf = 10, max_depth = 5)
dtree = dtree.fit(features, labels)

# Write the decision tree as an image so we can see how it was done
write_image_of_decision_tree('submission-1-decision-tree', dtree, columns)

# Remove unnecessary columns from the test data
test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1) 

# Run the prediction model on the test data
output = dtree.predict(test_data)

# Write the predicted data to a CSV file
result = np.c_[test_data['PassengerId'].astype(int), output.astype(int)]
df_result = pandas.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('output-model.csv', columns=['PassengerId', 'Survived'], index=False)

