# src/models/predictor.py
import os
import joblib
from typing import Dict, Any
from src.preprocessing.text_cleaner import clean_text

MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), 'models'))

class ModelPredictor:
    def __init__(self):
        self.models = {}
        spam_path = os.path.join(MODELS_DIR, 'spam_model.pkl')
        fake_path = os.path.join(MODELS_DIR, 'fake_model.pkl')
        if os.path.exists(spam_path):
            self.models['spam'] = joblib.load(spam_path)
        if os.path.exists(fake_path):
            self.models['fakenews'] = joblib.load(fake_path)

    def available_models(self):
        return list(self.models.keys())

    def predict(self, text: str, task: str = 'spam') -> Dict[str, Any]:
        if task not in self.models:
            return {'error': f"No model for task '{task}'. Train the model first."}
        model = self.models[task]
        cleaned = clean_text(text)
        result = {}
        try:
            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([cleaned])[0].tolist()
            label = model.predict([cleaned])[0]
            result.update({
                'task': task,
                'label': str(label),
                'probability': probs,
                'input': text
            })
        except Exception as e:
            result = {'error': str(e)}
        return result
