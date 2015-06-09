## My solutions to the  [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) problem.
The idea is to use machine learning to predict survivors of the Titanic disaster.
The contest gives you a training data set and indicates who survived along with a whole bunch of attributes for each passenger. You then need to use that to predict from another data set who would survive.

## Some ideas about contributing factors
- I looked at the deck plan of the ship: http://www.encyclopedia-titanica.org/titanic-deckplans/profile.html to see where life boats were and if there was an advantage to being 1st or 3rd class based on proximity to lifeboats. I couldn't detect anything obvious.
- People closer to the lifeboats would have a better chance of getting to the lifeboats
- Women and children *should* have been given preference to the lifeboats, so Sex and Age may be factors to consider.

I did 4 solutions, tweaking a bit each time. I left PassengerId in the Decision Tree submissions, and it had a big impact in the decision, which tells me either I should work to find factors more important than PassengerId, or I should remove PassengerId from the dataset.

##Submission 1
- Use decision trees with little data cleaning and see what factors contribute to survival
- Score when submitted to Kaggle: 0.77990

##Submission 2
- Use decision trees with more data cleaning and some feature engineering
- Add a title column and make age and fare more "local" averages
- Score when submitted to Kaggle: 0.76077

##Submission 3
- Use decision trees with same data cleaning and some more feature engineering
- Add a column IsAdultMale which seems to contribute well (male over 13)
- Add a column IsSouthamptonThirdClassCheapTickets (a female that embarked at Southampton and had a 3rd class ticket that cost 10 or less)
- Score when submitted to Kaggle: 0.80383

##Submission 4
- Use exactly the same as submission 3 except with a random forest
- Score when submitted to Kaggle: 0.72727

