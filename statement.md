# Fake News & Spam Classification System  
## Project Statement

---

## 📌 1. Problem Statement

With the rapid increase in online communication, users are exposed to massive amounts of information — including **fake news**, **misleading articles**, **spam messages**, and **phishing attempts**.  
These harmful or deceptive messages can lead to misinformation, financial losses, fraud, and reduced trust in digital platforms.

The goal of this project is to design an **automated, machine-learning powered classification system** that can:

- Detect whether a **news article** is *real or fake*
- Detect whether an **SMS/message** is *spam or ham*
- Provide predictions through a **FastAPI-based REST API**
- Support both **single and batch predictions**

This system helps users make informed decisions and enhances message reliability in communication platforms.

---

## 📌 2. Scope of the Project

### **In Scope**
- Preprocessing of raw text (cleaning, tokenizing, removing stopwords)
- Feature extraction using TF-IDF vectorization
- Training ML models for:
  - Fake News Detection  
  - SMS Spam Detection
- Creating a prediction pipeline with labeled probability outputs
- FastAPI backend with multiple endpoints:
  - `/predict/spam`
  - `/predict/fakenews`
  - `/predict/{task}/batch`
  - `/models`
- Simple frontend UI (`index.html`)
- Dockerfile for container-based deployment
- LIME explainability module (optional)

### **Out of Scope**
- Deep learning transformer models (BERT, GPT, RoBERTa)
- Multilingual fake news detection
- Image-based misinformation detection
- Database integration or authentication system
- Real-time scraping from news websites

---

## 📌 3. Target Users

This system is designed for:

### **1. General Users**
People who want to quickly check if a message is spam or if a news snippet is trustworthy.

### **2. Students & Researchers**
Useful for studying:
- Machine learning pipelines  
- NLP preprocessing  
- Classification models  
- API development  

### **3. Content Moderators / Journalists**
To verify news credibility before publishing or sharing.

### **4. Educational Institutions**
For teaching ML concepts, text classification, API deployment, and project structuring.

### **5. Developers / Data Enthusiasts**
For integrating a lightweight fake-news/spam classifier into other applications.

---

## 📌 4. High-Level Features

### **🔹 1. Machine Learning Models**
- TF-IDF + Logistic Regression for both tasks  
- Clean probability output:
  ```json
  "probabilities": { "ham": 0.21, "spam": 0.79 }
  ```
   
### 🔹 2. FastAPI REST Endpoints

- `/predict/spam` → classify SMS  
- `/predict/fakenews` → classify news  
- `/predict/{task}/batch` → classify multiple texts at once  
- `/models` → list loaded models  


### 🔹 3. Batch Prediction Support

- Accepts multiple messages/news in a single request.


### 🔹 4. Web UI

- Simple HTML interface (`/web/index.html`) that interacts with the API.


### 🔹 5. Modular Architecture

- Clean project structure: 
  - `src/models` → ML code
  - `src/app` → FastAPI application
  - `src/preprocessing` → NLP cleaning and text preprocessing
  - `models/` → trained .pkl model files
 

### 🔹 6. Explainability (Optional)

- LIME wrapper (`lime_wrapper.py`) to generate feature importance for predictions.


### 🔹 7. Deployment Ready

- Dockerfile included for containerized deployment  
- Uvicorn server for production and API hosting
 
  ---
   

