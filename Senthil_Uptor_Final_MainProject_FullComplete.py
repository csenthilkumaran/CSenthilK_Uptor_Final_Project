import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns


print("Manufacturing Segment with Time series data for Machine Learning using Supervised and Unsupervised Algorithms\n"
      "Features data> DateTime, OperationMode, Temperature, Vibration,PowerConsumption, NetworkLatency, PacketLoss\n "
      "K-Means for all Target parameters\n"
      "Target > ProductionSpeed->Linear Regression\n"
      "Target > PredictiveScore PM_Score ->XgB\n"
      "Target > Efficiency-> Decision Tree \n"
      "Target > QCDefectRate-> PCA and Linear Regression\n")

""" Loading the Manufacturing dataset """
Manuf_df = pd.read_csv("Senthil_Uptor_Final_Project.csv", parse_dates=['Timestamp'])
print(Manuf_df.columns)
print(Manuf_df.dtypes)
Manuf_df.set_index('Timestamp', inplace=True)
# print(Manuf_df)

""" Checking for Nan and forward fill """
FEATURES = ['Temperature', 'Vibration', 'PowerConsumption', 'NetworkLatency','PacketLoss']
TARGET = ['QCDefectRate','ProductionSpeed', 'PM_Score', 'ErrorRate']
LiR_TARGET = 'ProductionSpeed'
XgB_Target = 'PM_Score'
DT_TARGET = 'Efficiency'
PCA_Target = 'QCDefectRate'
#
# finding_NaN_column = Manuf_df[FEATURES].isna().sum()  # Finding the total sum of nan
# print(finding_NaN_column)
# Manuf_df[FEATURES] = Manuf_df[FEATURES].ffill()

"""Fixing X and Y datasets"""
X = Manuf_df[FEATURES]
y1 = Manuf_df.ProductionSpeed
y2 = Manuf_df.PM_Score
y3 = Manuf_df.Efficiency
y4 = Manuf_df.QCDefectRate

""" Plot Style Setting"""
# color_pal = sns.color_palette()
# Manuf_df.plot(style='.',
#         figsize=(15, 5),
#         color=color_pal[0],
#         title='Manufacturing Dataset')

# train = Manuf_df.loc[Manuf_df.index < '15-02-2024 00:00']
# test = Manuf_df.loc[Manuf_df.index >= '15-02-2024 00:00']
""" Since the data set is huge > i have attached the Image screenshot of the below output plot"""
# from statsmodels.tsa.seasonal import seasonal_decompose
# result_temp = seasonal_decompose(Manuf_df['Temperature'], model='additive', period=3600) #assuming minute data
# result_temp.plot()
# plt.show()

"""Split dataset into train and test sets where the Input features are all common 
for predicting the multiple Targets using different algorithms.  """

print(""" ------DecisionTree Algorithm for y3 -> Efficiency-------------\n""")

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
DTmodel = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
DTmodel.fit(X_train, y_train)

# Test the model
predictions = DTmodel.predict(X_test)
print("Predictions:", predictions)
print("actual:\n", y_test)

print("-----Confusion Matrix Output -----------\n")
metric_confusionmatrix_output = confusion_matrix (y_test, predictions)   # it forms a matrix with AP, PP, AN, PN
print(metric_confusionmatrix_output)

print(""" -------------------Classification report --------------------\n""")
from sklearn.metrics import classification_report
metric_CR_output = classification_report (y_test, predictions)
print(metric_CR_output)

print(""" --------Linear Regression Algorithm for y1 -> ProductionSpeed---------------\n""")
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test = train_test_split( X,y1,test_size=0.3,random_state=42)

LRmodel = LinearRegression()
LRmodel.fit(x_train,y_train)

model_x_prediction = LRmodel.predict(x_test)
print("Prod Speed for Test Data:",model_x_prediction)

LRmodel_accuracy = r2_score(y_test, model_x_prediction)
print("Model Accuracy of Prediction:",LRmodel_accuracy)

print(""" ----------XG Boost Algorithm for y3 -> PM_Score and Error Checking\n-------------""")
import xgboost as xgb
from sklearn.metrics import mean_squared_error

train = Manuf_df.loc[Manuf_df.index < '15-02-2024']
test = Manuf_df.loc[Manuf_df.index >= '15-02-2024']

X_train = train[FEATURES]
y_train = train[XgB_Target]
X_test = test[FEATURES]
y_test = test[XgB_Target]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

"""Forecast on Test"""
test['prediction'] = reg.predict(X_test)
"""Score (RMSE)"""
score = np.sqrt(mean_squared_error(test[XgB_Target], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')


print(""" ---------K-Means Algorithm for Manufacturing dataset on Targets------------------\n""")
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

min_max_scaler_object = MinMaxScaler()
y_scaled = min_max_scaler_object.fit_transform(Manuf_df[TARGET])
y = pd.DataFrame(y_scaled, columns = [Manuf_df[TARGET].columns])
print("Target data after Fit & Transform:\n", y.head())

kmean_object = KMeans(n_clusters=2, random_state=42)
fitted_model = kmean_object.fit_predict(y)
#print(kmean_object.labels_)
print("K-Mean Prediction:",fitted_model)

# plt.scatter(y.iloc[:,0],y.iloc[:,3], c=fitted_model, cmap="viridis", marker= "+", label="kmeans")
# plt.grid()
# plt.legend()
# plt.show()


""" PCA on y4- Quality control Defect Rate as Target"""
print("""-------------Create and train linear regression model for 
                       Quality_Defect rate% based on PCA transformed data------------\n""")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=5)  # Select the number of components to retain
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Create and train linear regression model on PCA transformed data
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Make predictions on the test set
y_predict = model.predict(X_test_pca)

# Evaluate model performance
mse = mean_squared_error(y_test, y_predict)
print("Quality Defect Rate % -> Mean Squared Error:", mse)

""" Deploying the model to find the Prodcution speed based on Linear Regression"""