from flask import Flask, request, jsonify
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and symptoms DataFrame
model = joblib.load('trained_model.joblib')
with open('symptoms_df.pkl', 'rb') as symptoms_file:
    symptoms_df = pickle.load(symptoms_file)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    try:
        # Get user input as JSON data
        user_input = request.get_json()

        # Convert user input to a DataFrame with symptom names and values
        user_symptoms = pd.DataFrame(user_input, columns=['Symptom'])
        user_symptoms['Number'] = user_symptoms['Symptom'].apply(lambda x: symptoms_df[symptoms_df['Symptom'] == x]['Number'].values[0])

        # Prepare the feature vector for prediction
        symptom_numbers = user_symptoms['Number'].values
        user_input_feature_vector = [1 if num in symptom_numbers else 0 for num in range(1, len(symptoms_df) + 1)]

        # Make predictions using the model
        predicted_disease = model.predict([user_input_feature_vector])

        # Get the predicted disease name
        predicted_disease_name = predicted_disease[0]

        # Return the prediction as a JSON response
        response = {
            'predicted_disease': predicted_disease_name
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
