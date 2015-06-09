This is a first shot at [Kaggle titanic](https://www.kaggle.com/c/titanic).

#Method
- Use a decision tree to predict survival or not
- Minimal data cleaning
  - embarked: replace null values with the mode (most recently occuring value)
  - fare: replace null values with the mean of non-null fares
  - age: replace null values with the mean age
- Data factorization
  - sex: replace male with 0 and female with 1
  - embarked: replace S with 0, C with 1 and Q with 2
- Decision Tree
  - random seed of 415
  - min_samples_leaf of 10
  - max_depth of 5

#Outcomes and observations
- Score for prediction: 0.77990
- See decision tree at submission-1-decision-tree.png
- The presence of passenger Id in the model is silly, since you can't really tell anythgin from that
- Males survive if
  - Age <= 6.5 AND at most 1 sibling
  - Age > 6.5 and 1st class
- Females die if
  - 3rd class AND embarked from Southampton AND fare <= 10
