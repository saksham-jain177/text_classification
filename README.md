# Customer Review Sentiment Classification

## Overview
This project implements a machine learning pipeline for classifying customer reviews from the IMDb dataset as positive or negative. The solution covers data loading, text preprocessing, TF-IDF feature extraction, model training, evaluation, and an interactive Streamlit interface for real-time predictions.**The training pipeline also uses caching and persistent model saving to avoid retraining on every run.**


## Objectives
- **Data Collection:** Load reviews from the [aclImdb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
- **Preprocessing:** Clean and tokenize reviews.
- **Feature Extraction:** Convert text to TF-IDF features.
- **Model Training:** Train a classifier (Logistic Regression) to predict sentiment.
- **Evaluation:** Assess model performance using standard metrics.
- **User Interface:** Provide an interactive UI for evaluation and review classification.

## Components

### Data Collection & Preprocessing
- Load the [aclImdb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (organized into train/test with positive and negative reviews).
- Clean and tokenize review texts.

### Feature Extraction
- Use TF-IDF vectorization to convert reviews into numerical features.

### Model Training & Evaluation
- Split the data into training and test sets.
- Train a Logistic Regression model.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

### User Interface
- A Streamlit app to run the entire pipeline, display evaluation metrics, and classify new reviews.
- Caching and model persistence to avoid retraining on every run.

## How to Run

1. **Clone the Repository:**

   ```
   git clone https://github.com/saksham-jain177/text_classification.git
   cd text-classification
   ```
2. **Install Dependencies:**
    ```
   pip install -r requirements.txt
    ```
3. **Run the Application:**
   ```
   streamlit run app.py
   ```
   
## Directory Structure
    text_classification/
    ├── app.py                     # Main application file (Streamlit interface)
    ├── data/
    │   └── aclImdb/               # IMDb dataset organized into train/test with pos/neg reviews
    ├── evaluation.py              # metrics and visualization
    ├── feature_extraction.py      # TF-IDF feature extraction
    ├── model.py                   # model training 
    ├── preprocessing.py           # data loading and text preprocessing
    ├── requirements.txt           
    └── README.md                  
## Challenges and Insights
- Balancing data cleaning and feature extraction to capture meaningful signals.
- Tuning the TF-IDF vectorizer for effective text representation.
- Achieving robust model performance given the variability in customer reviews.

## Future Improvements
- Experimenting with alternative classifiers (e.g., Naive Bayes, SVM) and ensemble methods.
- Integrate hyperparameter tuning for optimized performance.
- Enhance the UI with additional visualizations and batch prediction capabilities.