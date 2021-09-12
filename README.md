# E-Commerce Transaction Prediction


## Project Introduction
In this project, we analyze online shopper behavior through exploratory data analysis (EDA) and K-Means Clustering algorithm, and propose a system to predict the customer’s likelihood to abandon the session without any purchase. The first module of the system is a Stay Duration Prediction Model, which gives a conservative estimate on the session duration when a customer starts a session. When the session durations exceeds the estimate produced by the first module, the second module of the system, a Transaction Prediction Model, will be triggered to predict the likelihood of purchasing for the customer using clickstream data kept track during the session.

## Project Description
In the first module, the data is fed to Linear Regression, Decision Tree Regressor and Gradient Boosting Regressor as input. The results show that the Gradient Boosting Regressor has the lowest Root Mean Squared Error (RMSE).

In the second module, the data is fed to Logistic Regression, Decision Tree (DT), Random Forest (RF) and XGBoost Decision Tree (XGB) as input. We use oversampling and feature selection for data preprocessing to improve the performance and scalability of the classifiers. The results show that XGB with a strong L2 regularization has the highest Area Under the Curve of Receiver Operating Characteristic (AUC ROC).

The modules are used together to determine customers who are likely to leave the session without purchasing. Actions could be taken accordingly on the identified customers to improve purchase conversion rates. Our findings show the feasibility of an accurate and scalable system for predicting purchase likelihood using clickstream data.

## Results
Summary of training results on the dataset.

Module 1:
| Model                     | RMSE     | MAE      |
|---------------------------|----------|----------|
| DecisionTreeRegressor     | 2.606    | 1.302    |
| LinearRegression          | 2.587    | 1.282    |
| GradientBoostingRegressor | 2.582    | 1.268    |

Module 2:
| Model                     | ROC-AUC  | F1       |
|---------------------------|----------|----------|
| Logistic Regression       | 0.820    | 0.819    |
| Simple Decision Tree      | 0.873    | 0.873    |
| Gradient Boosted Tree     | 0.895    | 0.895    |

## Project Structure
```
├── data
│   └── online_shoppers_intention.csv       <-- full original dataset
│   └── train_origin.csv                    <-- original dataset used for 
│   └── train_bal.csv                       <-- rebalanced dataset used for 
│   └── test.csv                            <-- dataset used for testing
├── notebooks
│   └── 01_data_preprocessing.ipynb
│   └── 02_k_means_clustering.ipynb
│   └── 03_stay_duration_prediction.ipynb
│   └── 04_transaction_prediction.ipynb
└──
```

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling

### Libraries
* matplotlib 
* numpy 
* pandas
* sklearn
* xgboost

### Members (Group 7):

|Name                   |Matriculation Number   | 
|-----------------------|-----------------------|
|Chen Yilin             |                       |
|Chong Ying Qi          |                       |
|Chua Mint Sheen, Grace |                       |
|Ong Jing Long          |                       |
|Wang Hanbo             |                       |
|Zhang Simian           |                       |
|Tew Shu Rui            |                       |
