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
  """Clean the fare column.

  The strategy here is to replace 0 fares with the average of fares from the relevant 
  class and embarcation point.
  The reason to consider class in combination with embarcation point is that the fare
  amount would probably be different based on embarcation point.
  """

  # Determine combinations of unique class and embarked values
  unique_class_embark_combinations = []
  unique_classes = df['Pclass'].unique()
  unique_embarks = df['Embarked'].unique()

  for i in unique_classes:
    for j in unique_embarks:
      unique_class_embark_combinations.append((i,j))

  mean_fare_per_class_and_embark = {}

  for (i,j) in unique_class_embark_combinations:
    x = df[ (df['Pclass'] == i) & (df['Embarked'] == j) ]['Fare'] 
    mean_fare_per_class_and_embark[(i,j)] = df[ (df['Pclass'] == i) & (df['Embarked'] == j) ]['Fare'].mean()

  # Assign the mean fares for each class and embarcation point to the 0 fares
  df['Fare'] = df[['Fare','Pclass','Embarked']].apply(lambda x : mean_fare_per_class_and_embark[(x['Pclass'],x['Embarked'])] if x['Fare'] == 0 or pandas.isnull(x['Fare']) else x['Fare'], axis=1)


def clean_data_age(df):
  """Clean the age column.

  The strategy here is to replace null values with the median value for other
  passengers with the same title. This should give an age value closer to reality
  than the overall using the mean of the entire set.
  """ 

  # Determine median ages for each class of Title
  mean_ages = {}
  titles = df['Title'].unique()
  for i in titles:
    mean_ages[str(i)] = df[df['Title'] == i]['Age'].dropna().mean()

  # hard-code edge case of Ms instead of Mrs (sometimes used by unmarried adult females)
  if pandas.isnull(mean_ages['Ms']):
    mean_ages['Ms'] = mean_ages['Mrs']

  # Assign the mean ages for each title to the null ones
  df['Age'] = df[['Age','Title']].apply(lambda x : mean_ages[x['Title']] if pandas.isnull(x['Age']) else x['Age'], axis=1)


############################ Data factoring methods ############################
def factorize_data_sex(df):
  """Clean the sex column. Replace male = 0 and female = 1."""

  df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)  


def factorize_data_embarked(df):
  """Replace the letters with numbers."""

  embarked_codes = {'S': 0, 'C': 1, 'Q': 2}
  df['Embarked'] = df['Embarked'].apply(lambda x: embarked_codes[x])


def factorize_data_title(df):
  """Replace the letters with numbers."""
  unique_title = np.sort(df['Title'].unique())
  title_codes = {}
  j = 0
  for i in unique_title:
    title_codes[i] = j
    j = j + 1
  df['Title'] = df['Title'].apply(lambda x: title_codes[x])


############################ Feature engineering methods ############################

def engineer_title(df):
  """Extract each passenger's title (Mr, Mrs, etc.) and add it to the dataframe."""
  
  df['Title'] = df['Name'].map( lambda x: re.match('^[^,]+, ([^.]+).*$', x).group(1))


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

# Feature engineering
engineer_title(test_data)
engineer_title(train_data)

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
factorize_data_title(train_data)
factorize_data_title(test_data)

# Now make a decision tree
columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
labels = train_data['Survived'].values
features = train_data[list(columns)].values

# Fit our training data to a decision tree model
dtree = tree.DecisionTreeClassifier(random_state=415, min_samples_leaf = 10, max_depth = 5)
dtree = dtree.fit(features, labels)

# Write the decision tree as an image so we can see how it was done
write_image_of_decision_tree('submission-2-decision-tree', dtree, columns)

# Remove unnecessary columns from the test data
test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1) 

# Run the prediction model on the test data
output = dtree.predict(test_data)

# Write the predicted data to a CSV file
result = np.c_[test_data['PassengerId'].astype(int), output.astype(int)]
df_result = pandas.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('output-model.csv', columns=['PassengerId', 'Survived'], index=False)

