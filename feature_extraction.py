from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(corpus, max_features=5000):
    """
    Convert a collection of text documents to a TF-IDF feature matrix.
    
    Args:
        corpus (list of str): List of review texts.
        max_features (int): Maximum number of features (vocabulary size) to keep.
    
    Returns:
        X: The TF-IDF feature matrix.
        vectorizer: The fitted TfidfVectorizer object.
    """
    # Initialize the vectorizer. We also remove English stop words.
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    # Fit the vectorizer on the corpus and transform the documents into a feature matrix.
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# For testing purposes (can be removed in production)
if __name__ == "__main__":
    # Example corpus
    sample_corpus = [
        "I loved this movie, it was fantastic!",
        "The film was terrible and boring.",
        "An excellent performance by the lead actor.",
        "I did not enjoy the movie at all."
    ]
    X, vec = extract_features(sample_corpus)
    print("Feature matrix shape:", X.shape)
