import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔹 Load dataset
data_dir = r"C:\Users\BIT\Downloads\patient_medical_history_50k.csv"
raw_df = pd.read_csv(data_dir)

# 🔹 Drop missing values
raw_df = raw_df.dropna()

# 🔹 Encode binary categories manually
raw_df['Take_Medication'] = raw_df['Take_Medication'].replace({'Yes': 1, 'No': 0})
raw_df['Gender'] = raw_df['Gender'].replace({'Male': 1, 'Female': 0})

# 🔹 Separate features and target
X = raw_df[['Age', 'Gender', 'Past_Medical_History', 'Condition', 'Prescribed_Antibiotic']]
y = raw_df[['Take_Medication', 'Suggested_Antibiotic']]

# 🔹 Initialize LabelEncoders
encoders = {}

# ✅ Encode Categorical Columns in X
encoders = {}  # Dictionary to store encoders
categorical_cols = ['Past_Medical_History', 'Condition', 'Prescribed_Antibiotic']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Transform column
    encoders[col] = le
    joblib.dump(le, f'{col}_encoder.pkl')  # Save the encoder
print("✅ Feature Encoding Completed!")

# ✅ Encode Categorical Target Column in y
le_suggested_antibiotic = LabelEncoder()
y['Suggested_Antibiotic'] = le_suggested_antibiotic.fit_transform(y['Suggested_Antibiotic'])
joblib.dump(le_suggested_antibiotic, 'suggested_antibiotic_encoder.pkl')
print("✅ Target Encoding Completed!")

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Model Training
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
param_grid = {
    'estimator__n_estimators': [50, 100],         
    'estimator__max_depth': [None, 10, 20],         
    'estimator__min_samples_split': [2, 5]          
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
predictions = grid_search.predict(X_test)
predictions_df = pd.DataFrame(predictions, columns=y.columns)
print("\nPredictions:\n", predictions_df)

# 🔹 Save trained model
joblib.dump(grid_search.best_estimator_, 'final_model.pkl')

# 🔹 Predictions on test set
predictions = grid_search.predict(X_test)

# 🔹 Convert to DataFrame
predictions_df = pd.DataFrame(predictions, columns=y.columns)

# 🔹 Decode categorical predictions
suggested_antibiotic_encoder = joblib.load('Suggested_Antibiotic_encoder.pkl')
predictions_df['Suggested_Antibiotic'] = suggested_antibiotic_encoder.inverse_transform(predictions_df['Suggested_Antibiotic'])

# 🔹 Accuracy
accuracy1 = accuracy_score(y_test["Take_Medication"], predictions_df["Take_Medication"])
accuracy2 = accuracy_score(y_test["Suggested_Antibiotic"], y_test["Suggested_Antibiotic"])

print("\n✅ Accuracy for Take_Medication:", accuracy1)
print("✅ Accuracy for Suggested_Antibiotic:", accuracy2)

# 🔹 Save final predictions
predictions_df.to_csv("predictions_output.csv", index=False)
print("\n✅ Model and Encoders Saved Successfully!")
# Load Model and Encoders
model = joblib.load('model.pkl')
le_suggested_antibiotic = joblib.load('suggested_antibiotic_encoder.pkl')

# Load Encoders for Features
encoders = {
    'Past_Medical_History': joblib.load('Past_Medical_History_encoder.pkl'),
    'Condition': joblib.load('Condition_encoder.pkl'),
    'Prescribed_Antibiotic': joblib.load('Prescribed_Antibiotic_encoder.pkl')
}

# Example New Data (Make sure to encode it before prediction)
new_data = pd.DataFrame([{
    'Age': 45,
    'Gender': 1,  # Male -> 1
    'Past_Medical_History': encoders['Past_Medical_History'].transform(['Diabetes'])[0],
    'Condition': encoders['Condition'].transform(['Flu'])[0],
    'Prescribed_Antibiotic': encoders['Prescribed_Antibiotic'].transform(['Amoxicillin'])[0]
}])

# Make Prediction
prediction = model.predict(new_data)
predicted_suggested_antibiotic = le_suggested_antibiotic.inverse_transform([prediction[0][1]])[0]

print("\n✅ Predicted Take_Medication:", prediction[0][0])
print("✅ Predicted Suggested Antibiotic:", predicted_suggested_antibiotic)
