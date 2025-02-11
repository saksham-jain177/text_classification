from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data and print evaluation metrics.

    Args:
        model: The trained classifier.
        X_test: The TF-IDF feature matrix for the test set.
        y_test: The true labels for the test set.

    Returns:
        y_pred: The model predictions for the test set.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return y_pred

# For testing purposes (can be removed in production)
if __name__ == "__main__":
    # This block is only for a quick test.
    # In production, evaluation will be performed via main.py.
    import numpy as np
    # Fake test data for demonstration:
    y_test = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
