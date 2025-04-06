import os
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import tiktoken

# Change the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the dataset
df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
# Get all possible characters
full_text = "\n".join(df["conversation"].tolist() + test_df["conversation"].tolist())
# Build a vocabulary (character-level)
chars = sorted(list(set(full_text)))  # Get unique characters

stoi = { ch:i for i,ch in enumerate(chars) }  # Char to index map
itos = { i:ch for i,ch in enumerate(chars) }  # Index to char map

# Function to extract all the replies except for the last one
def get_customer_replies(conversation):
    turns = re.split(r'\r\n\r\n', conversation)  # Split conversation into turns
    customer_replies = [turn.replace("Customer:", "").strip() for turn in turns if turn.startswith("Customer:")]  # Remove "Customer:" and extra spaces
    last_n_replies = customer_replies[0:-1]  # Get the replies excluding the last reply
    return " ".join(last_n_replies)  # Join them into one text block

# Apply function to extract customer replies except for the last one
# Note: Last reply usually contains phrases like "Goodbye", therefore it is better to not include it
df["X"] = df["conversation"].apply(lambda x: get_customer_replies(x))
df = df[df["X"].apply(len) > 0]
test_df["X"] = test_df["conversation"].apply(lambda x: get_customer_replies(x))
test_df = test_df[test_df["X"].apply(len) > 0]

# Keep only the relevant columns (X and customer_sentiment as y)
df = df[["X", "customer_sentiment"]]
test_df = test_df[["X", "customer_sentiment"]]

enc = tiktoken.get_encoding("gpt2")
# Encode all conversations (X values)
encoded_X = [enc.encode_ordinary(text) for text in df["X"]]
encoded_X_test = [enc.encode_ordinary(text) for text in test_df["X"]]

# Encode the sentiments (y values)
sentiment_to_int = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}
df["encoded_sentiment"] = df["customer_sentiment"].map(sentiment_to_int)
test_df["encoded_sentiment"] = test_df["customer_sentiment"].map(sentiment_to_int)

# Convert to numpy arrays
X_encoded = np.array(encoded_X, dtype=object)  # X values (customer replies)
y_encoded = df["encoded_sentiment"].values

# Split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y_encoded, 
    test_size=0.2,
    random_state=2299436, 
    shuffle=True,  # shuffle the data
    stratify=y_encoded  # ensure class distribution is the same in both sets
)

# Save the data to binary files
with open("X_train.bin", "wb") as f:
    pickle.dump(X_train, f)

with open("X_val.bin", "wb") as f:
    pickle.dump(X_val, f)

with open("y_train.bin", "wb") as f:
    pickle.dump(y_train, f)

with open("y_val.bin", "wb") as f:
    pickle.dump(y_val, f)

X_encoded_test = np.array(encoded_X_test, dtype=object)  # X values (customer replies)
y_encoded_test = test_df["encoded_sentiment"].values

with open("X_test.bin", "wb") as f:
    pickle.dump(X_encoded_test, f)

with open("y_test.bin", "wb") as f:
    pickle.dump(y_encoded_test, f)

#Save the sentiment - int encodings to use later
meta = {
    'sentiment_to_int': sentiment_to_int,
    'vocab_size' : len(chars),
    'itos': itos,
    'stoi': stoi
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Train dataset size: {X_train.shape[0]}")
print(f"Validation dataset size: {X_val.shape[0]}")
print(f"Test dataset size: {X_encoded_test.shape[0]}")