from flask import Flask, request, jsonify
import joblib
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the saved model and tokenizer
rf_model = joblib.load('rf_model.joblib')
tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def extract_bert_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Extract BERT features
    features = extract_bert_features(text)
    
    # Make prediction
    prediction = rf_model.predict(features.reshape(1, -1))[0]
    
    return jsonify({'proficiency_level': prediction})

@app.route('/retrain', methods=['POST'])
def retrain():
    # Load the dataset
    data = pd.read_csv('data/language_proficiency_data.csv')
    
    # Split the data
    X = data['text']
    y = data['proficiency_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract BERT features
    X_train_bert = np.array([extract_bert_features(text) for text in X_train])
    X_test_bert = np.array([extract_bert_features(text) for text in X_test])
    
    # Train Random Forest model
    global rf_model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_bert, y_train)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test_bert)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the updated model
    joblib.dump(rf_model, 'rf_model.joblib')
    
    return jsonify({'message': 'Model retrained successfully', 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
