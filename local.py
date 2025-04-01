import joblib
import pandas as pd
# Load Model and Encoders
model = joblib.load('final_model.pkl')
le_suggested_antibiotic = joblib.load('suggested_antibiotic_encoder.pkl')

# Load Encoders for Features
encoders = {
    'Past_Medical_History': joblib.load('Past_Medical_History_encoder.pkl'),
    'Condition': joblib.load('Condition_encoder.pkl'),
    'Prescribed_Antibiotic': joblib.load('Prescribed_Antibiotic_encoder.pkl')
}

# Example New Data (Make sure to encode it before prediction)
new_data = pd.DataFrame([{
    'Age': 55,
    'Gender': 1,  # Male -> 1
    'Past_Medical_History': encoders['Past_Medical_History'].transform(['Hypertension'])[0],
    'Condition': encoders['Condition'].transform(['Bronchitis'])[0],
    'Prescribed_Antibiotic': encoders['Prescribed_Antibiotic'].transform(['Doxycycline'])[0]
}])

# Make Prediction
prediction = model.predict(new_data)
predicted_suggested_antibiotic = le_suggested_antibiotic.inverse_transform([prediction[0][1]])[0]

print("\n✅ Predicted Take_Medication:", prediction[0][0])
print("✅ Predicted Suggested Antibiotic:", predicted_suggested_antibiotic)
