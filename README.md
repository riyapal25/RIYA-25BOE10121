# Fake News & Spam Message Classifier (Complete)

This repository contains a complete starter project for detecting fake news and spam messages using ML.

## Quickstart
1. python -m venv venv
2. source venv/bin/activate   # Windows: venv\Scripts\activate
3. pip install -r requirements.txt
4. python -m nltk.downloader punkt stopwords
5. python src/models/trainer.py --spam data/raw/sms_spam.csv --fake data/raw/fake_news.csv
6. uvicorn src.app.main:app --reload
7. Open http://localhost:8000

## Notes
- Replace sample datasets in `data/raw/` with full datasets for real experiments.
- Transformer fine-tuning is optional; the trainer includes hooks/comments.
