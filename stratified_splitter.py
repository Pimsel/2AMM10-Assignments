import pandas as pd
from sklearn.model_selection import train_test_split

# Load in data to split
df = pd.read_csv('dataset/train.csv')

# Make stratified split (watch test_size and random_state)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=12,
    stratify=df['articleTypeId']
)

# Save split data to their own folder, !adjust file path for new splits!
train_df.to_csv('adjusted/filepath/file.csv', index=False)
val_df.to_csv('adjusted/filepath/file.csv', index=False)