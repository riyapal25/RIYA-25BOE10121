# src/models/trainer.py
import os
import joblib
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.models.dataloader import load_csv_dataset

MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

def train_baseline(df, outpath, test_size=0.2, random_state=42):
    X = df['text'].values
    y = df['label'].values
    if len(set(y)) < 2:
        print("Not enough classes to train")
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    joblib.dump(pipeline, outpath)
    print(f"Saved model to {outpath}")
    return pipeline

def main(spam_csv, fake_csv):
    if spam_csv:
        print("Training spam model from:", spam_csv)
        sp_df = load_csv_dataset(spam_csv)
        train_baseline(sp_df, os.path.join(MODELS_DIR, 'spam_model.pkl'))
    if fake_csv:
        print("Training fake-news model from:", fake_csv)
        fa_df = load_csv_dataset(fake_csv)
        train_baseline(fa_df, os.path.join(MODELS_DIR, 'fake_model.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spam', default='data/raw/sms_spam.csv', help='Path to spam CSV')
    parser.add_argument('--fake', default='data/raw/fake_news.csv', help='Path to fake news CSV')
    args = parser.parse_args()
    main(args.spam, args.fake)
