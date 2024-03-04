import pyreadr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

#load miami house price dataset
result = pyreadr.read_r("model_data.rds")
mydf = result[None]

#DATA CLEANING
#drop missing values
mydf = mydf.dropna()

#data exploration
summary = mydf.describe()
summary = summary.transpose()
print(summary)
print(mydf.head())

mystandardized_df = mydf.copy()

#create training (60%), validation (20%) and test sets (20%)
train, validate, test = np.split(mystandardized_df.sample(
    frac = 1, random_state =42), [int(.6*len(mystandardized_df)),int(.8*len(mystandardized_df))])
train_feature = train.loc[:,train.columns != "sale_prc"]
train_labels = train['sale_prc']
test_features = test.loc[:,train.columns != "sale_prc"]
test_labels = test['sale_prc']



#Let's start with a simple linear model
lm_model = linear_model.LinearRegression()
lm_model.fit(train_feature,train_labels)
lm_predictions = lm_model.predict(test_features)

for coef_name, value in zip(list(train_feature.columns),lm_model.coef_):
    print(coef_name, "coef value:", round(value,2))

print("Linear Model Mean Abs Error", sklearn.metrics.mean_absolute_error(test_labels,lm_predictions))
print("Linear Model Explained Variance", sklearn.metrics.explained_variance_score(test_labels,lm_predictions),"\n")

explained_variance_comparisons = {}
explained_variance_comparisons['linear_regression'] = sklearn.metrics.explained_variance_score(test_labels,lm_predictions)



#Random Forest Modeling -- can we achieve a model with higher explained variance?

#instatiate random forest with 500 trees
my_rf = RandomForestRegressor(n_estimators=500,random_state =123)
#fit model
my_rf.fit(train_feature,train_labels)
#make predictions on test set
rf_preds = my_rf.predict(test_features)

#evaluate model results
print('Random Forest Explained Variance Score:', metrics.explained_variance_score(test_labels, rf_preds))
explained_variance_comparisons['random_forest'] = metrics.explained_variance_score(test_labels, rf_preds)

#how important are each of our predictors?
importances = list(my_rf.feature_importances_)
feature_importance = [(feature,round(important,2)) for feature, important in zip(list(test_features.columns),importances)]
feature_importance = sorted(feature_importance,key=lambda x: x[1], reverse= True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance]
print()




#KNN Regression: can we build a simpler model that still beats the linear regression model but is easier to interpret?
#load new dataset containing geospatial coordinates of data and use neighboring house prices to predict
knn_df_raw = pd.read_csv('miami-housing.csv')
knn_df = knn_df_raw.copy().loc[:,['LATITUDE','LONGITUDE','SALE_PRC']]
print(knn_df.head())

#create datasplits
knn_train, knn_validate, knn_test = np.split(knn_df.sample(
     frac = 1, random_state =42), [int(.6*len(knn_df)),int(.8*len(knn_df))])

k_list = [i for i in range(1,21)]
knn_list = []

for i in k_list:
    #need to use haversine distance formula for lat and long sphereical distance
    knn = KNeighborsRegressor(n_neighbors = i,metric = 'haversine')
    model_knn = knn.fit(knn_train.iloc[:,0:2],knn_train.iloc[:,2])
    pred_knn = model_knn.predict(knn_validate.iloc[:,0:2])
    mse = mean_squared_error(knn_validate.iloc[:,2],pred_knn)
    knn_list.append(mse)

#plot elbow plot to see best k
plt.plot(k_list,knn_list)
plt.xlabel('Values of K')
plt.ylabel('MSE')
plt.title('The Elbow Method using MSE')
plt.show()

#from the elbow plot it seems like using 5 is best for our model
knn_final = KNeighborsRegressor(n_neighbors = 5,metric = 'haversine')
final_model_knn = knn.fit(knn_train.iloc[:,0:2],knn_train.iloc[:,2])
test_pred_knn = final_model_knn.predict(knn_test.iloc[:,0:2])
print("KNN Mean Abs Error", sklearn.metrics.mean_absolute_error(knn_test.iloc[:,2],test_pred_knn))
print("KNN Explained Variance", sklearn.metrics.explained_variance_score(knn_test.iloc[:,2],test_pred_knn))
explained_variance_comparisons['KNN_reg'] = sklearn.metrics.explained_variance_score(knn_test.iloc[:,2],test_pred_knn)

#visualize comparison of model performance
models = list(explained_variance_comparisons.keys())
explained_variance = list(explained_variance_comparisons.values())
fig = plt.figure(figsize=(7,7))
plt.bar(models,explained_variance)
plt.xlabel("Different Price Models")
plt.ylabel("Variance Explained by Model")
plt.title("Comparison of Model Performances to Predict Miami House Prices")
plt.show()