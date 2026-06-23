import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("weather_data.csv")

X = data[['temperature','rainfall','humidity','wind_speed']]
y = data['risk']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier()
model.fit(X,y_encoded)

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(encoder,open("encoder.pkl","wb"))

print("Model trained successfully!")