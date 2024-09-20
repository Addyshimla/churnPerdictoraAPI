import pandas as pd
from flask import Flask, request, jsonify,send_from_directory
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('churn_pred_model_poc 1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def documentation():
    return send_from_directory('static', ' documentation.html')

@app.route('/predict', methods=['POST'])
def predict():
    required_columns = [
        'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
        'PreferredPaymentMode', 'Gender', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'PreferredOrderCat',
        'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount'
    ]

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # Validate columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({"error": f"Missing columns in the CSV file: {', '.join(missing_columns)}"}), 400

            # Ensure 'CustomerID' is preserved for output
            customer_ids = df.get('CustomerID', None)  # Optional, if present

            # Preprocess the DataFrame (same as before)
            # (Include preprocessing code here as shown in the previous response)

            # Drop unnecessary columns for prediction
            df = df.drop(columns=['Unnamed: 20', 'CouponUsed'], errors='ignore')
            x = df.drop(columns='Churn', errors='ignore')

            # Make predictions
            predictions = model.predict(x)
            prediction_probabilities = model.predict_proba(x)

            # Prepare the response with CustomerID
            response = []
            churn_count = 0
            non_churn_count = 0

            for index in range(len(predictions)):
                # Map predicted_class to meaningful labels
                predicted_label = "Churn" if predictions[index] == 1 else "Not Churn"

                response.append({
                    "CustomerID": int(customer_ids.iloc[index]) if customer_ids is not None else None,  # Convert to int
                    "predicted_class": predicted_label,  # Use the label instead of int
                    "probability": prediction_probabilities[index].tolist()  # This should be fine as it returns a list
                })

                # Count churn and non-churn records
                if predictions[index] == 1:
                    churn_count += 1
                else:
                    non_churn_count += 1

            # Include total counts in the response
            summary = {
                "total_records": len(predictions),
                "total_churn": churn_count,
                "total_non_churn": non_churn_count
            }

            return jsonify({"summary": summary, "predictions": response}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
