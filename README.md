# Twitter Sentiment Analysis

## Overview
This project implements a Natural Language Processing (NLP) pipeline to classify the sentiment of tweets as either positive or negative. The model relies on **Term Frequency-Inverse Document Frequency (TF-IDF)** for text vectorization and a **Logistic Regression** classifier to make its predictions. 

## Tech Stack
* **Language:** Python
* **Machine Learning:** `scikit-learn` (Logistic Regression, TF-IDF, Model Evaluation)
* **Data Manipulation:** `pandas`, `numpy`
* **Text Preprocessing:** `nltk` or `re` (Regex)
* **Data Visualization:** `matplotlib`, `seaborn` (for confusion matrix and metrics)

## Pipeline Architecture

1. **Data Preprocessing:** 
   Raw tweets are noisy. The preprocessing step cleans the text by removing URLs, user mentions (`@user`), hashtags, special characters, and punctuation. The text is lowercased, and common English stop words are removed to reduce noise.
   
2. **Feature Extraction (TF-IDF):** 
   The cleaned text data is passed through a `TfidfVectorizer`. This converts the text into numerical vectors, assigning importance to words based on how frequently they appear in a specific tweet versus how common they are across the entire dataset.

3. **Classification:** 
   The vectorized text is fed into a `LogisticRegression` model. Logistic Regression is highly efficient for binary classification tasks (Positive/Negative) and performs exceptionally well on sparse, high-dimensional data like TF-IDF matrices.


