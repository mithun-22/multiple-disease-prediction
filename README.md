##🩺 Multiple Disease Prediction System
  
  This project is a Streamlit web application that allows users to predict the likelihood of Kidney Disease, Liver Disease, and Parkinson’s Disease using pre-trained Machine Learning models. The models are built with RandomForestClassifier, scaled using StandardScaler, and saved as .pkl files for quick inference.

##📌 Features

🧠 Parkinson's Disease Prediction (based on voice measurements)

🩸 Liver Disease Prediction (based on standard liver function tests)

🧪 Kidney Disease Prediction (based on clinical inputs and blood/urine values)

📈 Pre-trained models loaded dynamically

✅ Real-time results with a clean UI in Streamlit

##🧰 Tech Stack
Python 3.x

Scikit-learn

Pandas & NumPy

Joblib for saving/loading models

Streamlit for web interface

##📁 Project Structure
bash
Copy
Edit
multiple-disease-predictor/
│
├── liver_rf_model.pkl               # Pre-trained liver model
├── scaler_liver.pkl                 # Scaler used for liver features
├── kidney_rf_model.pkl              # Pre-trained kidney model
├── kidney_scaler.pkl                # Scaler used for kidney features
├── parkinsons_rf_model (1).pkl     # Pre-trained parkinson's model
├── scaler (2).pkl                   # Scaler used for parkinson's features
├── multi_disease_predictor.py      # Streamlit app file
└── README.md                        # Project documentation (this file)
