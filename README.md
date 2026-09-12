# House Price Prediction

## Project Overview

This project predicts house prices using the California Housing dataset and a Linear Regression model.

The project includes data exploration, preprocessing, feature engineering, model training, evaluation, visualization, and house price prediction.

## Objective

To predict the median house value based on features such as:

* Median income
* House age
* Average rooms
* Average bedrooms
* Population
* Average occupancy
* Latitude
* Longitude

## Dataset

The project uses the **California Housing dataset** available through Scikit-learn.

* Total records: **20,640**
* Features: **8**
* Target: **MedHouseVal**
* Missing values: **0**

The target value is represented in units of **$100,000**.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Machine Learning

The project uses **Linear Regression** with feature engineering.

### Preprocessing

* Train/test split
* Log transformation for skewed features
* Polynomial features
* StandardScaler normalization

### Model

**Linear Regression**

## Model Evaluation

The model was evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Results

| Metric | Training | Testing |
| ------ | -------: | ------: |
| MSE    |   0.3680 |  0.3786 |
| RMSE   |   0.6066 |  0.6153 |
| R²     |   0.7247 |  0.7111 |

The testing R² score of **0.7111** means the model explains approximately **71.11% of the variation** in house values.

## Visualizations

### House Price Distribution



### Actual vs Predicted House Values


## Example Prediction

The model predicted:

* **Predicted House Price:** $309,136.27
* **Actual House Price:** $309,100.00
* **Prediction Error:** $36.27


## Conclusion

The House Price Prediction project successfully demonstrates how Linear Regression can be used to predict house values after preprocessing and feature engineering. The model achieved a testing R² score of **0.7111**.
