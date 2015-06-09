This is a second attempt at [Kaggle titanic](https://www.kaggle.com/c/titanic).

#Method
- Use a decision tree to predict survival or not
- Some feature engineering based on outcome of submission 1
  - Add a title column, extracted from the name
- Data cleaning
  - embarked: replace null values with the mode (most recently occuring value)
  - age: instead of mean of the whole sample, use mean for the title
    - I'm assuming that the average age of Masters, for example, (unmarried males) would be a better average for males with that title and no stated age)
  - fare: instead of mean fare, use the mean fare per class per point of embarcation
    - I'm assuming that the fares would differ based on where you embarked and the class of travel
- Data factorization
  - sex: replace male with 0 and female with 1
  - embarked: replace S with 0, C with 1 and Q with 2
  - title: replace 
- Decision Tree
  - random seed of 415
  - min_samples_leaf of 10
  - max_depth of 5

#Outcomes and observations
- Score is 0.76077
- See decision tree at submission-2-decision-tree.png
- The presence of passenger Id in the model is silly, since you can't really tell anything from that
- Males survive if
  - Age <= 13 and 0, 1 or 2 siblings
- Females die if
  - 3rd class AND fare of 10 or less AND embarked Southampton

