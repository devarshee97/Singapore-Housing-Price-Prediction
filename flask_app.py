import os

from flask import Flask, request, render_template
from flask_cors import cross_origin
import jsonify
import requests
import pickle
import sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

flask_app = Flask(__name__)
infile = open("rf.pkl", 'rb')
model = pickle.load(infile)
infile.close()


sc = StandardScaler()

df_train = pd.read_csv("train_data.csv")
y_train  = pd.read_csv("y_train.csv")

df_train_sc = sc.fit_transform(df_train)




@flask_app.route("/")
@cross_origin()
def home():
    return render_template("index.html")




@flask_app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        Town = request.form["Town"]
        Flat_model = request.form["Flat_model"]
        Total_rooms = int(request.form['Total_rooms'])
        floor_area_sqm = int(request.form['floor_area_sqm'])
        lease_commence_date = int(request.form['lease_commence_date'])
        Max_storeys = int(request.form["Max_storeys"])
        remaining_years_of_lease= int(request.form['remaining_years_of_lease'])
        registration_year = int(request.form['registration_year'])

        df_town_prem = pd.read_csv("town_prem.csv")
        df_flat_prem = pd.read_csv("flat_prem.csv")

        town_dic = dict(zip(df_town_prem["town"], df_town_prem["town_premium"]))
        flat_dic = dict(zip(df_flat_prem["flat_model"], df_flat_prem["flat_premium"]))

        Town_premium = town_dic[Town]
        Flat_premium = flat_dic[Flat_model]




        input_data = [[floor_area_sqm, lease_commence_date, Total_rooms,
                                      Max_storeys, remaining_years_of_lease,
                                      Town_premium, Flat_premium, registration_year]]

        input_data_scaled = sc.transform(input_data)

        Housing_price = model.predict(input_data_scaled)

        return render_template("index.html", prediction_text="Your cost for the house will be {} Singaporean Dollar".format(Housing_price[0]))


    return render_template("index.html")

if __name__=="__main__":
    flask_app.run(debug=True)