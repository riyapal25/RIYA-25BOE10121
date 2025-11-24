# tests/test_api.py
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_root():
    res = client.get('/')
    assert res.status_code in (200, 404, 200)

def test_models():
    res = client.get('/models')
    assert res.status_code == 200

def test_predict_missing():
    res = client.post('/predict/spam', json={})
    assert res.status_code in (422, 400)
