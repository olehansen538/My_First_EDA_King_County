import statsmodels.api as sms
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd


df = pd.read_csv("King_County_House_prices_dataset.csv")

# Adjusting data types and cleaning
df.price = df.price.astype("int") 
df.bedrooms = df.bedrooms.astype("int8")
df.bathrooms = df.bathrooms.astype("float32")
df.sqft_living = df.sqft_living.astype("int32")
df.sqft_lot = df.sqft_lot.astype("int32")
df.floors = df.floors.astype("float16")
df.waterfront = df.waterfront.fillna(0).astype("int8")
df.view = df.view.fillna(0).astype("int8")
df.grade = df.grade.astype("int8")
df.condition = df.condition.astype("int8")
df.grade = df.grade.astype("int8")
df.sqft_above = df.sqft_above.astype("int32")
df.sqft_basement = pd.to_numeric(df.sqft_basement, errors='coerce').fillna(0).astype("int32")
df.yr_built = df.yr_built.astype("int16")
df.yr_renovated = df.yr_renovated.fillna(0).astype("int16")
df.zipcode = df.zipcode.astype("int32")
df.sqft_living15 = df.sqft_living.astype("int32")
df.sqft_lot15 = df.sqft_lot.astype("int32")

df = df[df.bedrooms < 33]
df = df[df.sqft_living15 < 12000]

# Feature engineering
Y = df["price"]
X = df[["sqft_living15", "sqft_living", "floors", "waterfront", "yr_renovated",
        "yr_built", "view", "grade", "bedrooms", "bathrooms"]]

print("-----  Splitting the data in train and test ----")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Adding intercept
X_train = sms.add_constant(X_train) # adding a constant
X_test = sms.add_constant(X_test) # adding a constant

# Training the model
print("-----  Training the model ----")
ridge = Ridge(alpha=0.1, normalize=True)
model = ridge.fit(X_train, y_train)
#print_model = model.summary()

print("-----  Evaluating the model ----")
ridge_pred = ridge.predict(X_train)
err_train = np.sqrt(mean_squared_error(y_train, ridge_pred))
ridge_test = model.predict(X_test)
err_test = np.sqrt(mean_squared_error(y_test, ridge_test))

#print(print_model)
print ("-------------")
print (f"RMSE on train data: {err_train}")
print (f"RMSE on test data: {err_test}")

