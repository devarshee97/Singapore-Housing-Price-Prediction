# Singapore Housing Price Prediction
The objective in this project is to predict the possible resale price of a flat in Singapore based on data collected from 2017-21. The features relevant in this task include number of rooms, flat model, floor area size etc.
# Workflow

## Data collection: 
The dataset is obtained from https://data.gov.sg/dataset/resale-flat-prices through the available API, and after the necessary format changes, it is prepared as a data frame. The data is supposed to be updated within a period of four months when new records are entered
# Data preprocessing:
* There are no null values present in the dataset
* Certain categorical features like number of rooms and date of registration are processed into numerical format
* A key point worth mentioning is that in my project, I have not explicitly utilised the location and street name features, however, it has an important role in determining the distance to necessary accessories nearby and thus becomes a significant factor for price prediction. The analysis involving street name and locality is done in extensive detail in 
 https://towardsdatascience.com/predicting-singapore-hdb-resale-price-eda-and-modeling-94af7d26798d by Tian Jie.
 * Another idea borrowed from the above blog is to intriduce the features describing the difference between overall median price of flats and the mdeian price of a flat in a particular category or model. These factors are also noticed to have high importance in determining the resale price
 * The outliers are analysed using boxplot and imputed throguh IQR technique.
 # Model fitting:
 * Three different models i.e XGBoost, CatBoost and Random Forest reressor were tried out, among which RandomForest gave the best performance with 93.7 % accuracy.
 * Performance metric used is Root Mean Squared Error
 * A slight improvement in performance was noticed after scaling features with standard scaler
 # Deployment(incomplete):
 * **Back end** developed using **Flask**
 * 
