# Medical-Cost-Personal-Datasets

This project aims to analyze the medical cost personal datasets to uncover insights and patterns.
My final goal was to build a predictive model to estimate medical charges based on various features.

1. Initial Exploration
Libraries
- numpy
- pandas
- seaborn
- matplotlib
  
2. Exploratory Data Analysis (EDA) - Checking for Missing Values
3. Visualizations - A distribution plot of the charges variable is created to visualize the distribution of medical charges.
4. BMI vs. Charges - I created a scatter plot to visualize the relationship between BMI and charges ( + the smoker variable to distinguish between smokers and non-smokers).
5. Heatmap/Checking Correlations - I created a heatmap to visualize the correlation matrix of the dataset. It helps to identify any strong correlations between variables.

------
### Second part - Model Building

Preparing the Data: I converted the categorical variables into dummy variables using pd.get_dummies(). The data is then split into features (x) and (y). 
The data is divided into training and testing sets.

Important part: Training the Model - A Random Forest Regressor is initialized and trained on the training data. Predictions are made on the test data.
Evaluating the Model
- The performance of the model is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics. These metrics provide insight into the accuracy and reliability of the predictive model.
  
- Classification tree - It turns out to be too complicated tree.
  (With a maximum depth of 2 we already have quite a good classification accuracy (over 93%). For larger depths, Gini splits seem to be better than entropy splits).
  
![Classification Tree](https://github.com/tercasaskova311/pictures/blob/main/tree.plot.png)

- Extremely simple classifier: I noted that the last level of the tree (Classification tree) is not really separating anything (both labels in the split are equal). Then, I  further prune the tree.
  
![Classification Tree](https://github.com/tercasaskova311/pictures/blob/main/tree_plot_tree.png)

  -----
### Results:

Mean Absolute Error (MAE): Provides the average magnitude of errors in a set of predictions, without considering their direction.
Mean Squared Error (MSE): Measures the average of the squares of the errors, giving higher weight to larger errors.
Root Mean Squared Error (RMSE): The square root of MSE, providing an error metric in the same units as the target variable.
R-squared (R²): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
