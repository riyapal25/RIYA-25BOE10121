# src/preprocessing/text_cleaner.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure these are downloaded before running:
# python -m nltk.downloader punkt stopwords

try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {'the','a','and','is','in','it','of','to'}

def clean_text(text: str) -> str:
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)
