import os
import pandas as pd
from tqdm import tqdm  # Import tqdm

def load_imdb_data(data_dir):
    """
    Load the IMDb data from the aclImdb folder.
    
    The data_dir should point to the aclImdb folder (e.g., 'task_b/data/aclImdb').
    It reads from both the 'train' and 'test' directories, each containing 'pos' and 'neg' subdirectories.
    
    Returns:
        A pandas DataFrame with columns:
          - 'review': The review text.
          - 'sentiment': The label ('pos' or 'neg').
    """
    reviews = []
    sentiments = []
    
    # Iterate over the train and test sets
    for subset in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            folder_path = os.path.join(data_dir, subset, sentiment)
            # Check if the folder exists (in case of any unexpected structure)
            if not os.path.exists(folder_path):
                continue
            # Use tqdm to wrap the list of filenames for a progress bar
            for filename in tqdm(os.listdir(folder_path), desc=f"Reading {subset}/{sentiment} files"):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            review = f.read()
                            reviews.append(review)
                            sentiments.append(sentiment)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        
    df = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments
    })
    return df

# For quick testing:
if __name__ == "__main__":
    data_directory = "data/aclImdb"  # Adjust the path if necessary (relative to where you run the script)
    df = load_imdb_data(data_directory)
    print(df.head())
    print("Total reviews loaded:", len(df))
