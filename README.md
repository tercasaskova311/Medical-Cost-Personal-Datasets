# Medical-Cost-Personal-Datasets

This project aims to analyze the medical cost personal datasets to uncover insights and patterns.
My final goal was to build a predictive model to estimate medical charges based on various features.

## 1. Initial Exploration

**Libraries Used**:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
  
## 2. Exploratory Data Analysis (EDA)

**Checking for Missing Values**:  
An initial check was performed to identify and handle any missing values in the dataset.
**Visualizations**:
- **Distribution Plot**: A distribution plot of the `charges` variable was created to visualize the distribution of medical charges.
**BMI vs. Charges**:
- A scatter plot was created to visualize the relationship between BMI and charges, with the `smoker` variable to distinguish between smokers and non-smokers.
**Heatmap/Checking Correlations**:
- I created a heatmap to visualize the correlation matrix of the dataset. It helps to identify any strong correlations between variables.

------
### Second part - Model Building

**Preparing the Data**:
- Categorical variables were converted into dummy variables using `pd.get_dummies()`.
- The data was then split into features (`X`) and the target variable (`y`).
- The data was divided into training and testing sets.

**Training the Model**:A Random Forest Regressor 
- initialized and trained on the training data.
- predictions were made on the test data.

**Evaluating the Model**:
- The performance of the model was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics. These metrics provide insight into the accuracy and reliability of the predictive model.

**Classification Tree**:
- Initially, the classification tree turned out to be too complicated.
  - With a maximum depth of 2, the classification accuracy was quite good (over 93%). For larger depths, Gini splits seemed to be better than entropy splits.

![Classification Tree](https://github.com/tercasaskova311/pictures/blob/main/tree.plot.png)

- **Extremely Simple Classifier**: It was noted that the last level of the classification tree did not really separate anything (both labels in the split were equal). Therefore, the tree was further pruned.

![Classification Tree](https://github.com/tercasaskova311/pictures/blob/main/tree_plot_tree.png)

----

### Results:

- **Mean Absolute Error (MAE)**: Provides the average magnitude of errors in a set of predictions, without considering their direction.
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors, giving higher weight to larger errors.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing an error metric in the same units as the target variable.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
