#!  C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe

from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import time
import joblib

app = Flask(__name__)

# Global variable to check if SMOTE is available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        for col in self.columns:
            self.encoders[col] = LabelEncoder().fit(X[col].astype(str))
        return self

    def transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].transform(X[col].astype(str))
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

def preprocess_data(data):
    # Feature engineering
    data['TCGA_PATHWAY'] = data['TCGA_DESC'] + '_' + data['PATHWAY_NAME']
    
    features = ['CELL_LINE_NAME', 'TCGA_DESC', 'PATHWAY_NAME', 'TCGA_PATHWAY', 'PUTATIVE_TARGET', 'MIN_CONC', 'MAX_CONC', 'LN_IC50', 'AUC', 'RMSE', 'Z_SCORE']
    target = 'DRUG_NAME'
    
    X = data[features]
    y = data[target]
    
    # Encode categorical features
    categorical_features = ['CELL_LINE_NAME', 'TCGA_DESC', 'PATHWAY_NAME', 'TCGA_PATHWAY', 'PUTATIVE_TARGET']
    feature_encoder = MultiColumnLabelEncoder(columns=categorical_features)
    X_encoded = feature_encoder.fit_transform(X)
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X_encoded, y_encoded, feature_encoder, target_encoder, features

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    selector = SelectKBest(f_classif, k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train_resampled)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, selector, accuracy, training_time

# Global variables to store model and related objects
# Global variables to store model and related objects
model = None
scaler = None
selector = None
feature_encoder = None
target_encoder = None
data = None
features = None

@app.route('/')
def home():
    global data
    if data is None:
        data = pd.read_csv("GDSC_data.csv")
    
    return render_template_string('''
        <h1>Cancer Drug Prediction App</h1>
        <form action="/train" method="post">
            <input type="submit" value="Train Model">
        </form>
    ''')

@app.route('/train', methods=['POST'])
def train():
    global model, scaler, selector, feature_encoder, target_encoder, data, features
    
    if data is None:
        data = pd.read_csv("GDSC_data.csv")
    
    sample_size = min(50000, len(data))
    data_sample = data.sample(n=sample_size, random_state=42)
    
    X, y, feature_encoder, target_encoder, features = preprocess_data(data_sample)
    model, scaler, selector, accuracy, training_time = train_model(X, y)
    
    # Save the model and related objects
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(selector, 'selector.joblib')
    joblib.dump(feature_encoder, 'feature_encoder.joblib')
    joblib.dump(target_encoder, 'target_encoder.joblib')
    
    return render_template_string('''
        <h1>Cancer Drug Prediction App</h1>
        <p>Model training completed in {{training_time}} seconds. Accuracy: {{accuracy}}</p>
        <form action="/predict" method="post">
            <h2>Predict Drug</h2>
            <label>Cell Line Name: <input type="text" name="CELL_LINE_NAME" required></label><br>
            <label>TCGA Description: <input type="text" name="TCGA_DESC" required></label><br>
            <label>Pathway Name: <input type="text" name="PATHWAY_NAME" required></label><br>
            <label>Putative Target: <input type="text" name="PUTATIVE_TARGET" required></label><br>
            <label>Minimum Concentration: <input type="number" step="any" name="MIN_CONC" required></label><br>
            <label>Maximum Concentration: <input type="number" step="any" name="MAX_CONC" required></label><br>
            <label>LN IC50: <input type="number" step="any" name="LN_IC50" required></label><br>
            <label>AUC: <input type="number" step="any" name="AUC" required></label><br>
            <label>RMSE: <input type="number" step="any" name="RMSE" required></label><br>
            <label>Z Score: <input type="number" step="any" name="Z_SCORE" required></label><br>
            <input type="submit" value="Predict">
        </form>
    ''', training_time=f"{training_time:.2f}", accuracy=f"{accuracy:.2f}")
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, selector, feature_encoder, target_encoder, data, features
    
    if model is None or scaler is None or selector is None or feature_encoder is None or target_encoder is None:
        return "Please train the model first."
    
    # Get input data from form
    input_data = {
        'CELL_LINE_NAME': request.form.get('CELL_LINE_NAME'),
        'TCGA_DESC': request.form.get('TCGA_DESC'),
        'PATHWAY_NAME': request.form.get('PATHWAY_NAME'),
        'TCGA_PATHWAY': f"{request.form.get('TCGA_DESC')}_{request.form.get('PATHWAY_NAME')}",
        'PUTATIVE_TARGET': request.form.get('PUTATIVE_TARGET'),
        'MIN_CONC': float(request.form.get('MIN_CONC')),
        'MAX_CONC': float(request.form.get('MAX_CONC')),
        'LN_IC50': float(request.form.get('LN_IC50')),
        'AUC': float(request.form.get('AUC')),
        'RMSE': float(request.form.get('RMSE')),
        'Z_SCORE': float(request.form.get('Z_SCORE'))
    }
    input_df = pd.DataFrame([input_data])
    
    # Ensure input_df has all necessary features
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # or some appropriate default value
    
    input_df = input_df[features]  # Reorder columns to match training data
    
    input_encoded = feature_encoder.transform(input_df)
    input_selected = selector.transform(input_encoded)
    input_scaled = scaler.transform(input_selected)
    prediction = model.predict(input_scaled)
    predicted_drug = target_encoder.inverse_transform(prediction)[0]
    
    return render_template_string('''
        <h1>Cancer Drug Prediction App</h1>
        <p>Predicted Drug: {{predicted_drug}}</p>
        <a href="{{url_for('home')}}">Back to Home</a>
    ''', predicted_drug=predicted_drug)

if __name__ == '__main__':
    app.run(debug=True)