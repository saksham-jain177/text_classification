import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_imdb_data  # Ensure your load_imdb_data function is in preprocessing.py
from feature_extraction import extract_features
from model import train_model
from evaluation import evaluate_model

def main():
    # Set the path to the aclImdb dataset folder (inside your task_b/data directory)
    data_dir = os.path.join("data", "aclImdb")
    print("Loading data from:", data_dir)
    
    # Load the IMDb data (assumes load_imdb_data returns a DataFrame with columns "review" and "sentiment")
    df = load_imdb_data(data_dir)
    print("Total reviews loaded:", len(df))
    
    # If desired, you can limit the dataset size for faster experimentation (e.g., sample 2000 reviews)
    # df = df.sample(n=2000, random_state=42)
    
    # Extract the review texts and the sentiment labels.
    # Note: The dataset contains "pos" and "neg" labels. Map them to numeric values (e.g., pos=1, neg=0).
    corpus = df['review'].tolist()
    y = df['sentiment'].map({'pos': 1, 'neg': 0}).tolist()
    
    # Extract TF-IDF features from the review texts.
    print("Extracting TF-IDF features...")
    X, vectorizer = extract_features(corpus)
    print("Feature matrix shape:", X.shape)
    
    # Split the dataset into training and testing sets (e.g., 80/20 split).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classification model.
    print("Training the model...")
    model = train_model(X_train, y_train)
    
    # Evaluate the model on the test set.
    print("\nModel Evaluation:")
    evaluate_model(model, X_test, y_test)
    
if __name__ == "__main__":
    main()
