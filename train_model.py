import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/crime_dataset_india.csv')
df = df.dropna(subset=['Crime Domain'])

X = df[['City', 'Crime Description', 'Victim Age', 'Victim Gender', 'Weapon Used']].copy()
y = df['Crime Domain']

encoders = {}
for col in ['City', 'Crime Description', 'Victim Gender', 'Weapon Used']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

X['Victim Age'] = pd.to_numeric(X['Victim Age'], errors='coerce').fillna(X['Victim Age'].mean())

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/crime_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("models/target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)
