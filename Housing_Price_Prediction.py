#!/usr/bin/env python
# coding: utf-8

# In[67]:


import requests
import pandas as pd
import json
import csv
import numpy as np
import seaborn as sns

from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[68]:


url = "https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit=200000"

    
response = requests.get(url)


# In[70]:


response.status_code


# In[71]:


print(type(response))


# In[72]:


temp = response.json
json_obj = json.dumps(temp())


# In[73]:


with open("response.json","w") as outfile:
    outfile.write(json_obj)


# In[74]:


with open("response.json") as json_file:
    data = json.load(json_file)
    
raw_data = data["result"]["records"]

data_file = open("data_file.csv", "w")
csv_writer = csv.writer(data_file)

count = 0

for data in raw_data:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count+=1
        
    csv_writer.writerow(data.values())
    
data_file.close()    


# In[75]:


df = pd.read_csv("data_file.csv")
df.head()


# In[76]:


df.shape


# In[78]:


df["town"].value_counts()


# In[79]:


df["flat_type"].value_counts()


# In[80]:


df['flat_model'].value_counts()


# In[81]:


df["storey_range"].value_counts()


# In[82]:


df["flat_type"].str.split()


# In[83]:


df.replace(to_replace= ["EXECUTIVE","MULTI-GENERATION"], value=["6 ROOM", "7 ROOM"], inplace= True)


# In[84]:


df["flat_type"].value_counts()


# In[85]:


df["flat_type"].str.split().str[0]


# In[86]:


df["Total_rooms"] = df["flat_type"].str.split().str[0]


# In[87]:


df["Min_storeys"] = df["storey_range"].str.split().str[0]
df["Max_storeys"] = df["storey_range"].str.split().str[2]


# In[88]:


df["remaining_lease"].str.split().str[0]


# In[89]:


df["remaining_years_of_lease"] = df["remaining_lease"].str.split().str[0]


# In[90]:


df.isnull().sum()


# In[91]:


df.head()


# In[92]:


town_lst = np.sort(df.town.unique()).tolist()
len(town_lst)


# In[93]:


df["resale_price"].median()


# In[94]:


df.groupby("town")["resale_price"].median()


# In[95]:


ser = df["resale_price"].median() - df.groupby("town")["resale_price"].median()


temp_df = pd.DataFrame({"town": ser.index, "town_premium": ser.values})
temp_df


# In[96]:


df = pd.merge(df, temp_df, left_on = "town", right_on = "town")
df


# In[97]:


df.groupby("flat_model")["resale_price"].median()


# In[98]:


ser_2 = df["resale_price"].median() - df.groupby("flat_model")["resale_price"].median()


temp_df2 = pd.DataFrame({"flat_model": ser_2.index, "flat_premium": ser_2.values})
temp_df2


# In[99]:


df = pd.merge(df, temp_df2, left_on = "flat_model", right_on = "flat_model")
df.head()


# In[100]:


df["Registration_year"] = df["month"].str.split("-").str[0]


# In[101]:


y = df["resale_price"]


# In[102]:


cols = ["town", "_id", "storey_range", "street_name","flat_type", "flat_model","remaining_lease",
        "block", "month"]

df.drop(cols, axis=1, inplace= True)


# In[103]:


df.info()


# In[104]:


df.drop("resale_price", axis=1, inplace=True)


# In[105]:


to_num_cols = ["Total_rooms", "Min_storeys", "Max_storeys", "remaining_years_of_lease", "Registration_year"]
df[to_num_cols] = df[to_num_cols].apply(pd.to_numeric)


# In[106]:


df.info()


# In[107]:


all_data = df.copy()
all_data["resale_price"] =  y

corrmat = all_data.corr(method = "spearman")
plt.figure(figsize=(15,15))
#plot heat map
g=sns.heatmap(corrmat,annot=True)


# In[108]:


df.drop("Min_storeys", axis=1, inplace= True)


# In[109]:


all_data.describe()


# In[110]:


col_lst = df.columns.tolist()


# In[111]:


from scipy.stats import skew

for fea in col_lst:
    plt.figure(figsize=(10,10))
    sns.set_style('whitegrid')
    sns.distplot(df[fea], kde=True, bins= 50)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(fea)
    plt.show()
    print("The skewness is ", skew(df[fea]))


# In[49]:


for fea in col_lst:
    plt.figure(figsize=(10,10))
    sns.boxplot(df[fea])
    plt.show()


# In[112]:


Q1_fa =  df["floor_area_sqm"].quantile(0.25)
Q3_fa =  df["floor_area_sqm"].quantile(0.75)

iqr_fa = Q3_fa - Q1_fa

up_lim_fa = Q3_fa + 1.5*iqr_fa
low_lim_fa = Q1_fa - 1.5*iqr_fa


Q1_tr =  df["Total_rooms"].quantile(0.25)
Q3_tr =  df["Total_rooms"].quantile(0.75)

iqr_tr = Q3_tr - Q1_tr

up_lim_tr = Q3_tr + 1.5*iqr_tr
low_lim_tr = Q1_tr - 1.5*iqr_tr

Q1_ms =  df["Max_storeys"].quantile(0.25)
Q3_ms =  df["Max_storeys"].quantile(0.75)

iqr_ms = Q3_ms - Q1_ms

up_lim_ms = Q3_ms + 1.5*iqr_ms
low_lim_ms = Q1_ms - 1.5*iqr_ms

Q1_fp =  df["flat_premium"].quantile(0.25)
Q3_fp =  df["flat_premium"].quantile(0.75)

iqr_fp = Q3_fp - Q1_fp

up_lim_fp = Q3_fp + 1.5*iqr_fp
low_lim_fp = Q1_fp - 1.5*iqr_fp

c1 = df["floor_area_sqm"] > up_lim_fa
c2 = df["floor_area_sqm"] < low_lim_fa
c3 = df["Total_rooms"] > up_lim_tr
c4 = df["Total_rooms"] < low_lim_tr
c5 = df["Max_storeys"] > up_lim_ms
c6 = df["Max_storeys"] < low_lim_ms
c7 = df["flat_premium"] > up_lim_fp
c8 = df["flat_premium"] < low_lim_fp


# In[113]:


df.loc[c1, "floor_area_sqm"] = up_lim_fa
df.loc[c2, "floor_area_sqm"] = low_lim_fa
df.loc[c3, "Total_rooms"]  = up_lim_tr
df.loc[c4, "Total_rooms"]  = low_lim_tr
df.loc[c5, "Max_storeys"]  = up_lim_ms
df.loc[c6, "Max_storeys"]  = low_lim_ms
df.loc[c7, "flat_premium"] = up_lim_fp
df.loc[c8, "flat_premium"] = low_lim_fp


# In[52]:


for fea in col_lst:
    print(fea)
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[fea].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[fea], dist="norm", plot=plt)
    plt.show()


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=1)


# In[118]:


sc= StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[119]:


xgb_reg = XGBRegressor()
rf_reg = RandomForestRegressor()

model_xgb = xgb_reg.fit(X_train_sc, y_train)

model_rf = rf_reg.fit(X_train_sc, y_train)


# In[120]:


cv_xgb = cross_val_score(xgb_reg, X_train_sc, y_train, scoring = "neg_mean_squared_error", cv=5)

xgb_scores = np.sqrt(-cv_xgb.mean())
print(xgb_scores)


# In[121]:


cv_rf = cross_val_score(rf_reg, X_train_sc, y_train,scoring = "neg_mean_squared_error", cv=5)

rf_scores = np.sqrt(-cv_rf.mean())
print(rf_scores)


# In[122]:


cb = CatBoostRegressor()
model_cb = cb.fit(X_train_sc, y_train)
y_pred_cb = model_cb.predict(X_test_sc)


# In[123]:


y_pred_xgb = model_xgb.predict(X_test_sc)


# In[124]:


y_pred_rf = model_rf.predict(X_test_sc)


# In[125]:


from sklearn.metrics import r2_score


# In[154]:





# In[ ]:





# In[126]:


from sklearn.metrics import mean_squared_error
xgb_mse = mean_squared_error(y_test, y_pred_xgb, squared= False)
print("XGB accuracy", model_xgb.score(X_test_sc, y_test))
print("XGB RMSE", mean_squared_error(y_test, y_pred_xgb, squared = False))


# In[127]:


print("RandomForest accuracy", model_rf.score(X_test_sc, y_test))
print("Random Forest test performance", mean_squared_error(y_test, y_pred_rf, squared = False))


# In[128]:


sns.distplot(y_test-y_pred_rf)


# In[129]:


plt.scatter(y_test, y_pred_rf)


# In[131]:


import pickle

file = open("randomforest_regression.pkl", "wb")
pickle.dump(model_rf,file)


# In[ ]:




