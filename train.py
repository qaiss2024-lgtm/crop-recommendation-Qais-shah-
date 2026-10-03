import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Dataset load karna
df = pd.read_csv('Crop_recommendation.csv')

# 2. Features (Inputs) aur Target (Output) alag karna
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Data ko Train aur Test mein split karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Random Forest Model ko train karna
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Trained model ko save karna
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully train ho gaya aur 'crop_model.pkl' file ban gayi!")