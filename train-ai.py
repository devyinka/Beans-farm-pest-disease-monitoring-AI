import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#Opening beans_farm_data.csv...
print("1. Opening beans_farm_data.csv...")
df = pd.read_csv('beans_farm_data.csv')

print("2. Isolating the features and the target...")
X = df.drop('Target_Threat', axis=1)
y = df['Target_Threat']

print("3. Slicing data into 80% Training and 20% Testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training the Random Forest AI...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

accuracy = rf_model.score(X_test, y_test)
print(f"5. Training Complete! Final Accuracy: {accuracy * 100:.2f}%")

print("6. Saving the model to beans_rf_model.pkl...")
with open('beans_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("SUCCESS: The .pkl file is ready.")