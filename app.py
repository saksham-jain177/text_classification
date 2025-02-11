import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_imdb_data   # Your function to load the aclImdb dataset
from feature_extraction import extract_features
from model import train_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#############################################
# Persistent Model Saving/Loading Functions #
#############################################

def load_saved_model():
    """
    Load the saved model, vectorizer, report, and confusion matrix from disk.
    Returns:
        model, vectorizer, report, cm if available; otherwise, None for each.
    """
    if (os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl") and 
        os.path.exists("report.pkl") and os.path.exists("cm.pkl")):
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        with open("report.pkl", "rb") as rep_file:
            report = pickle.load(rep_file)
        with open("cm.pkl", "rb") as cm_file:
            cm = pickle.load(cm_file)
        return model, vectorizer, report, cm
    else:
        return None, None, None, None

def save_model(model, vectorizer, report, cm):
    """
    Save the model, vectorizer, report, and confusion matrix to disk.
    """
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
    with open("report.pkl", "wb") as rep_file:
        pickle.dump(report, rep_file)
    with open("cm.pkl", "wb") as cm_file:
        pickle.dump(cm, cm_file)

############################################################
# Cached Training Pipeline (to avoid retraining every time) #
############################################################

@st.cache_resource(show_spinner=False)
def get_trained_model(data_dir):
    # Load the dataset
    df = load_imdb_data(data_dir)
    corpus = df['review'].tolist()
    # Map sentiment labels: pos -> 1, neg -> 0
    y = df['sentiment'].map({'pos': 1, 'neg': 0}).tolist()
    
    # Extract TF-IDF features
    X, vectorizer = extract_features(corpus)
    
    # Split the dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classification model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, vectorizer, report, cm

##########################################
# Pipeline Runner (with Persistent Model)#
##########################################

def run_pipeline():
    # Define the path to the aclImdb dataset folder (relative to your task_b/ directory)
    data_dir = os.path.join("data", "aclImdb")
    st.write("**Loading data from:**", data_dir)
    
    # Check if a saved model exists; if so, load it.
    saved_model, saved_vectorizer, saved_report, saved_cm = load_saved_model()
    if saved_model is not None and saved_vectorizer is not None:
        st.write("**Loaded saved model and vectorizer from disk!**")
        return saved_model, saved_vectorizer, saved_report, saved_cm
    else:
        # Otherwise, train the model (this is cached to avoid retraining repeatedly)
        model, vectorizer, report, cm = get_trained_model(data_dir)
        st.write("**Model training complete!**")
        # Save the model for future runs
        save_model(model, vectorizer, report, cm)
        return model, vectorizer, report, cm

######################
# Display Metrics UI #
######################

def display_metrics(report, cm):
    st.subheader("Classification Report")
    # Convert the report dictionary to a DataFrame and display it
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    st.subheader("Confusion Matrix")
    # Create a heatmap of the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

################
# Main Program #
################

def main():
    st.title("Customer Review Sentiment Classification")
    st.write("This application classifies customer reviews (from the IMDb dataset) as positive or negative.")
    
    # Button to run the sentiment classification pipeline
    if st.button("Run Sentiment Classification Pipeline"):
        with st.spinner("Running pipeline... This may take a few moments."):
            model, vectorizer, report, cm = run_pipeline()
        st.success("Pipeline execution complete!")
        
        # Save the trained model, vectorizer, report, and confusion matrix in session state
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.report = report
        st.session_state.cm = cm
        
        # Display evaluation metrics
        display_metrics(report, cm)
    
    # Allow the user to test a new review if the model is already trained or loaded
    if "model" in st.session_state and "vectorizer" in st.session_state:
        st.subheader("Test a New Review")
        user_review = st.text_area("Enter a review below:")
        if st.button("Classify Review"):
            if user_review:
                # Transform the user review using the saved vectorizer
                user_features = st.session_state.vectorizer.transform([user_review])
                prediction = st.session_state.model.predict(user_features)
                sentiment = "Positive" if prediction[0] == 1 else "Negative"
                st.write("**Predicted Sentiment:**", sentiment)
            else:
                st.warning("Please enter a review!")
    
    # Optionally, show a detailed classification report in an expander
    if "report" in st.session_state:
        with st.expander("Show Detailed Classification Report"):
            st.dataframe(pd.DataFrame(st.session_state.report).transpose())

if __name__ == "__main__":
    main()
