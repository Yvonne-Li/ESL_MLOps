import pytest
from app import app, extract_bert_features
import json
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict(client):
    response = client.post('/predict', json={'text': 'This is a test sentence.'})
    assert response.status_code == 200
    assert 'proficiency_level' in json.loads(response.data)

def test_retrain(client):
    response = client.post('/retrain')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'accuracy' in data

def test_extract_bert_features():
    features = extract_bert_features('This is a test sentence.')
    assert isinstance(features, np.ndarray)
    assert features.shape == (768,)  # BERT base model output dimension
