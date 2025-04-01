import os
import pickle
import numpy as np
import pandas as pd
import re


# Change the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the dataset
df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
# Get all possible characters
full_text = "\n".join(df["conversation"]).join(test_df["conversation"])
# Build a vocabulary (character-level)
chars = sorted(list(set(full_text)))  # Get unique characters
stoi = { ch:i for i,ch in enumerate(chars) }  # Char to index map
itos = { i:ch for i,ch in enumerate(chars) }  # Index to char map

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Function to extract the last 3 customer replies
def get_customer_replies(conversation):
    turns = re.split(r'\r\n\r\n', conversation)  # Split conversation into turns
    customer_replies = [turn.replace("Customer:", "").strip() for turn in turns if turn.startswith("Customer:")]  # Remove "Customer:" and extra spaces
    last_n_replies = customer_replies[0:-2]  # Get the replies excluding the last 2 replies
    return " ".join(last_n_replies)  # Join them into one text block

# Apply function to extract last 3 customer replies
df["X"] = df["conversation"].apply(lambda x: get_customer_replies(x))

# Keep only the relevant columns (X and customer_sentiment as y)
df = df[["X", "customer_sentiment"]]

# Function to encode text to integers
def encode(text):
    return [stoi[c] for c in text]  # Convert characters to their corresponding indices

# Function to decode integers back to text
def decode(int_list):
    return ''.join([itos[i] for i in int_list])  # Convert indices back to characters

# Encode all conversations (X values)
encoded_X = [encode(text) for text in df["X"]]

# Encode the sentiments (y values)
sentiment_to_int = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}
df["encoded_sentiment"] = df["customer_sentiment"].map(sentiment_to_int)

# Convert to numpy arrays
X_encoded = np.array(encoded_X, dtype=object)  # X values (customer replies)
y_encoded = df["encoded_sentiment"].values

# Split into train and validation sets (80% train, 20% validation)
n = len(X_encoded)
X_train, X_val = X_encoded[:int(n*0.8)], X_encoded[int(n*0.8):]
y_train, y_val = y_encoded[:int(n*0.8)], y_encoded[int(n*0.8):]

# Save the data to binary files
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
