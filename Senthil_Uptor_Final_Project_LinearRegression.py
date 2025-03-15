import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

print("Manufacturing Segment with Time series data for Machine Learning using Supervised and Unsupervised Algorithms\n"
      "Features data> DateTime, OperationMode, Temperature, Vibration,PowerConsumption, NetworkLatency, PacketLoss\n "
      "Target > ProductionSpeed->Linear Regression\n"
      "Deployment and evaluate the model with user input\n")

""" Loading the Manufacturing dataset """
Manuf_df = pd.read_csv("Senthil_Uptor_Final_Project.csv", parse_dates=['Timestamp'])
# print(Manuf_df.columns)
# print(Manuf_df.dtypes)
Manuf_df.set_index('Timestamp', inplace=True)
print(Manuf_df)

""" Checking for Nan and forward fill """
FEATURES = ['Temperature', 'Vibration', 'PowerConsumption', 'NetworkLatency','PacketLoss']
TARGET = ['QCDefectRate','ProductionSpeed', 'PM_Score', 'ErrorRate']
LiR_TARGET = 'ProductionSpeed'

"""Fixing X and Y datasets"""
X = Manuf_df[FEATURES]
y1 = Manuf_df.ProductionSpeed

print(""" ---------Linear Regression Algorithm for y1 -> ProductionSpeed-------------""")
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test = train_test_split( X,y1,test_size=0.3,random_state=42)

LRmodel = LinearRegression()
LRmodel.fit(x_train,y_train)

model_x_prediction = LRmodel.predict(x_test)
print("Linear Regression Prediction of Production Speed:",model_x_prediction)

LRmodel_accuracy = r2_score(y_test, model_x_prediction)
print(" LR Model accuracy of Predicted data",LRmodel_accuracy)

print(""" -----Deployment of the model using Pickle and SteamLit libraries------""")

import pickle
with open ("Senthil_Final_linear_model_pickling.pkl","wb") as obj:
    pickle.dump(LRmodel, obj)