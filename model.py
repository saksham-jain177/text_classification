from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """
    Train a Logistic Regression model on the provided training data.
    
    Args:
        X_train: TF-IDF feature matrix for the training set.
        y_train: Training labels (e.g., positive or negative).
    
    Returns:
        model: The trained Logistic Regression model.
    """
    # Initialize the Logistic Regression model.
    # Increase max_iter if needed to ensure convergence.
    model = LogisticRegression(max_iter=1000)
    # Fit the model on the training data.
    model.fit(X_train, y_train)
    return model

# For testing purposes (can be removed in production)
if __name__ == "__main__":
    # Import additional necessary libraries for testing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import pandas as pd
    
    # Create a small sample DataFrame for demonstration purposes.
    data = {
        'review': [
            "I loved this movie, it was fantastic!",
            "The film was terrible and boring.",
            "An excellent performance by the lead actor.",
            "I did not enjoy the movie at all."
        ],
        'sentiment': [1, 0, 1, 0]  # Let's assume 1=positive, 0=negative
    }
    df = pd.DataFrame(data)
    
    # For feature extraction, import the extract_features function from feature_extraction.py
    from feature_extraction import extract_features
    
    # Extract features from the review texts.
    X, vectorizer = extract_features(df['review'].tolist())
    y = df['sentiment']
    
    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train the model.
    model = train_model(X_train, y_train)
    
    # Evaluate the model.
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
