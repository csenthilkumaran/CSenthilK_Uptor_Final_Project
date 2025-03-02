import pandas as pd
# from sklearn.datasets import load_boston  # Sample dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

""" Loading the Boston Housing dataset """

boston = pd.read_csv("boston_house_prices.csv")
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# y = boston.target

""" Printing the Columns of the Boston Dataset"""
print(boston.columns)

# X =
# y =

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Apply PCA to reduce dimensionality
# pca = PCA(n_components=10)  # Select the number of components to retain
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)
#
# # Create and train linear regression model on PCA transformed data
# model = LinearRegression()
# model.fit(X_train_pca, y_train)
#
# # Make predictions on the test set
# y_pred = model.predict(X_test_pca)
#
# # Evaluate model performance
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)