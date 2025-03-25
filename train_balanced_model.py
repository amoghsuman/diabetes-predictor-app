# train_balanced_model.py
#Train a balanced model using SMOTE + class weights

#Save the architecture and weights in your root folder:
#diabetes_model.json
#diabetes_model.weights.h5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# 1. Load and clean data
data = pd.read_csv("data/diabetes.csv", names=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'PedigreeFunction', 'Age', 'Outcome'])

# Replace 0s with NaNs for key features
for col in ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Glucose']:
    data[col].replace(0, np.nan, inplace=True)

# Impute with medians
for col in ['BloodPressure', 'BMI', 'SkinThickness', 'Insulin']:
    data[col].fillna(data[col].median(), inplace=True)

# Predict missing Glucose using regression (optional — can skip if very few)
data['Glucose'].fillna(data['Glucose'].median(), inplace=True)

# 2. Log transform skewed features
for col in ['SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction', 'Age']:
    data[col] = np.log(data[col] + 1)

# 3. Feature scaling
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Use SMOTE to balance dataset
sm = SMOTE(random_state=42)
X_resampled, Y_resampled = sm.fit_resample(X[:, [0,1,5,7]], Y)  # Use selected 4 features

# 5. Train/test split
X_train, X_val, Y_train, Y_val = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

# 6. One-hot encode labels
Y_train_oh = pd.get_dummies(Y_train).values
Y_val_oh = pd.get_dummies(Y_val).values

# 7. Build model
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 8. Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i : class_weights[i] for i in range(len(class_weights))}

# 9. Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train_oh, epochs=50, batch_size=32, validation_data=(X_val, Y_val_oh), class_weight=class_weight_dict, verbose=1)

# 10. Save model and weights
with open("model.json", "w") as json_file:
    json_file.write(model.to_json())

model.save_weights("model.weights.h5")
print("✅ Model saved as 'model.json' and 'model.weights.h5'")
