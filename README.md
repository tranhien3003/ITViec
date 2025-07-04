# ITViec
Sentiment Analysis and Information Clustering for ITviec

📊 Project Overview
This project is designed to analyze user-generated reviews on ITviec.com — a leading job platform for IT professionals in Vietnam. The goal is to extract insights from these reviews to help partner companies understand their public perception and improve their employer branding.

🎯 Objectives
Perform sentiment analysis on reviews to classify them as positive, negative, or neutral.
Cluster companies based on review characteristics to identify patterns and benchmarking opportunities.

🧩 Requirements
1. Sentiment Analysis
Analyze textual reviews from candidates and employees to determine the sentiment:
Positive
Negative
Neutral
This helps companies monitor feedback trends and respond accordingly.

2. Information Clustering
Group companies into clusters based on the nature of their reviews. This enables:
Identification of peer groups
Benchmarking against similar companies
Strategic insights for improvement

🛠 Technologies Used
Python 3
Natural Language Processing (NLP)
Scikit-learn
Pandas, NumPy
Matplotlib / Seaborn (for visualization)
Jupyter Notebook

🚀 How to Run
# 📊 ITviec Sentiment Analysis and Clustering Project

This repository contains all necessary files to run a **Streamlit application** that performs **sentiment analysis** and **topic clustering** on employee reviews collected from ITviec.

---

## 📁 Project File Structure

| File Name                             | Description |
|--------------------------------------|-------------|
| **Companies_Clean.xlsx**             | Cleaned company data used for linking with reviews. |
| **Reviews_Cluster.xlsx**             | Reviews that have been clustered into topics using LDA. |
| **Reviews_User_Web.xlsx**            | Reviews submitted by users through the Streamlit web interface. |
| **sentiment_prediction_results.xlsx**| Exported results including predicted sentiment and cluster label. |
| **Logistic_Regression_sentiment_model.pkl** | Pre-trained Logistic Regression model for sentiment prediction. |
| **xgboost_sentiment_model.pkl**      | Pre-trained XGBoost model for more advanced sentiment analysis. |
| **lda_model.pkl**                    | Trained LDA model for topic modeling and clustering of reviews. |
| **vectorizer.pkl**                   | Vectorizer used to transform review texts for LDA clustering. |
| **text_preprocessor.pkl**            | Text transformer pipeline for preprocessing user input. |
| **function.py**                      | Python script with helper functions for preprocessing and translation. |
| **streamlit_app.py**                 | Main Streamlit application to launch the dashboard. |
| **requirements.txt**                 | Python package requirements for installing dependencies. |
| **Procfile**                         | Deployment configuration file (e.g., for Heroku). |
| **setup.sh**                         | Shell script for environment setup during deployment. |
| **sentiment_Cover.jpg**             | Visual cover image used in the Streamlit UI. |
| **emojicon.txt**                     | Dictionary file to convert emojis to text for better processing. |
| **teencode.txt**                     | Vietnamese teen slang dictionary for normalization. |
| **wrong-word.txt**                   | List of frequently misspelled words used for correction. |

---

## 🚀 Features

- 🔍 **Sentiment Analysis**: Supports both Logistic Regression and XGBoost models.
- 🧠 **Topic Clustering**: Applies LDA model to assign reviews to dominant topics.
- ✍️ **User Input Review**: Allows real-time sentiment and cluster analysis.
- 📊 **Interactive Dashboard**: Built with Streamlit for visualization and insights.

---

## 🛠️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📌 Notes

- Models are pre-trained and saved in `.pkl` format for faster execution.
- The app preprocesses reviews using slang, emoji, and spelling correction dictionaries before inference.
- Clustering is based on LDA and can be used to suggest improvements for employers.

---


