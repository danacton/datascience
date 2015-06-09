This is a third attempt at [Kaggle titanic](https://www.kaggle.com/c/titanic).

#Method
- Use a decision tree to predict survival or not
- Some feature engineering based on outcome of submission 2
  - Don't use title from submission 2
  - add a new column, IsAdultMale, which is 1 if (Male and Age is > 13) and 0 if not (since males only survive if they're <= 13 and one expects women and children to get preference and adult males to help get others to safety first)
  - add a new column IsSouthamptonThirdClassCheapTickets which is 1 if Female and 3rd class and fare of 10 or less and embarked at Southampton.
- Data cleaning
  - As per submission 2
- Data factorization
  - sex: replace male with 0 and female with 1
  - embarked: replace S with 0, C with 1 and Q with 2
  - title: replace
- Decision Tree
  - random seed of 415
  - min_samples_leaf of 10
  - max_depth of 5

#Outcomes and observations
- Score is 0.80383
- See decision tree at submission-3-decision-tree.png
- The presence of passenger Id in the model is silly, since you can't really tell anything from that
- Interestingly, the prediction is more accurate with the addition of these 2 columns, maybe because the model is simpler
- Males survive if 
  - (not adult) and have 0, 1 or 2 siblings
- Females die if
  - 3rd class and embarked Southampton and fare <= 10
