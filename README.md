# Fake News & Spam Classifier (AI & ML Project)

## 📌 Project Title
**Fake News & Spam Detection System using Machine Learning & FastAPI**

---

## 📌 Overview of the Project
The Fake News & Spam Classifier is an end-to-end machine learning system that detects:

- Whether a **news article is Fake or Real**
- Whether an **SMS message is Spam or Ham**

This project uses **NLP**, **TF-IDF vectorization**, and **Logistic Regression** models trained on two datasets:

- `sms_spam.csv` (Spam/Ham messages)
- `fake_news.csv` (Fake/Real news dataset)

A **FastAPI backend** serves predictions via REST APIs, and a **simple Web UI** allows users to test the system easily.  
The application also supports **batch prediction**, **labeled probability outputs**, and clean ML pipelines.

---

## 📌 Features

### 🔹 Machine Learning
- Text Preprocessing Pipeline  
- TF-IDF Vectorization  
- Logistic Regression Models  
- Clean labeled probabilistic output:
```json
"probabilities": {
  "ham": 0.21,
  "spam": 0.79
}
```
---

## 📌Technologies / Tools Used

### 🔹 Programming & ML Tools

- Python 3.10+
- scikit-learn
- pandas
- numpy
- NLTK (tokenization & stopwords)
- joblib (model saving/loading)

### 🔹 Backend / Deployment

- FastAPI
- Uvicorn
- CORS Middleware
- Dockerfile for deployment

### 🔹 Development Tools

- VS Code
- Virtual Environment (venv)
- Swagger UI (/docs)

---

## 📌 Steps to Install & Run the Project

### 1️⃣ Clone the project
```
git clone <your-repository-url>
cd fakespamclassifier
```

### 2️⃣ Create and activate virtual environment

```
Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate
```

```
Windows CMD:

venv\Scripts\activate.bat
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Download required NLTK resources
```
python -m nltk.downloader punkt punkt_tab stopwords
```

### 5️⃣ Train the models
```
python -m src.models.trainer --spam data/raw/sms_spam.csv --fake data/raw/fake_news.csv
```


This creates:
```
models/spam_model.pkl
models/fake_model.pkl
```

### 6️⃣ Run the application
```
python -m uvicorn src.app.main:app --reload
```

### 7️⃣ Open in browser
```
Web UI:
http://127.0.0.1:8000
```
---

## 📌 Instructions for Testing

### ✔ Test Spam/Ham Prediction

In Swagger → POST /predict/spam
Body example:
```
{
  "text": "You have won a free prize!"
}
```

### ✔ Test Fake / Real News Prediction

Swagger → POST /predict/fakenews
Body:
```
{
  "text": "Local council approves new park"
}
```

### ✔ Batch Prediction

Swagger → POST /predict/{task}/batch
Body:
```
{
  "texts": [
    "Win a free iPhone now!",
    "Government launches new policy"
  ]
}
```

### ✔ Test from PowerShell
```
$body = @{ text = "You have won a free prize!" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/spam" -Method Post -ContentType "application/json" -Body $body
```
---

## 📌Screenshot

<img width="1178" height="572" alt="image" src="https://github.com/user-attachments/assets/b37efa9c-2896-41af-9788-0893ca1c7905" />
<img width="1178" height="572" alt="Screenshot 2025-11-24 202229" src="https://github.com/user-attachments/assets/b2e24d5e-1021-46fa-945a-beebf3d97281" />


