import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
ds = load_dataset("maharshipandya/spotify-tracks-dataset")
df = ds['train'].to_pandas()

print(f"Original dataset: {len(df)} rows")
df = df.drop_duplicates(subset=['track_id'])
df = df.dropna(subset=['track_id'])  # Ensure track_id is not null
df = df.dropna(subset=['track_name'])  # Ensure track_name is not null

print(f"After dropping duplicates: {len(df)} rows")

# Select all textual columns for embedding
text_cols = ['artists', 'track_name', 'track_genre', 'album_name']
# Select all numerical columns
num_cols = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'time_signature']

model = SentenceTransformer('all-MiniLM-L6-v2')  # Outputs 384-dimensional embeddings

# Generate embeddings for textual columns
text_embeddings = {}
for col in text_cols:
    text_embeddings[col] = model.encode(df[col].astype(str).tolist(), show_progress_bar=True)

# Combine textual embeddings by averaging
combined_text_embeddings = np.mean(
    [text_embeddings[col] for col in text_cols],
    axis=0
)
print("Combined text embedding size:", combined_text_embeddings.shape)  # (n_samples, 384)

scaler = MinMaxScaler()
num_features = df[num_cols].copy()

# Convert boolean 'explicit' to numeric (0/1)
num_features['explicit'] = num_features['explicit'].astype(int)
# Handle any missing or invalid values (if any)
num_features = num_features.fillna(0)  # Simple imputation; adjust as needed
normalized_num_features = scaler.fit_transform(num_features)
print("Normalized numerical features size:", normalized_num_features.shape)  # (n_samples, 15)

# Concatenate text embeddings with numerical features
# Text embeddings: 384 dims, Numerical features: 15 dims -> Total: 399 dims
combined_embeddings = np.concatenate([combined_text_embeddings, normalized_num_features], axis=1)
print("Final combined embedding size:", combined_embeddings.shape)  # (n_samples, 399)

client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "spotify_tracks_collection_all_features"
try:
    collection = client.create_collection(name=collection_name)
except:
    collection = client.get_collection(name=collection_name)

ids = df['track_id'].tolist()

for col in text_cols + num_cols:
    if df[col].isnull().any():
        # Replace NaN/None with appropriate defaults based on column type
        if col in text_cols:
            df[col] = df[col].fillna("Unknown")
        elif col == 'explicit':
            df[col] = df[col].fillna(False)
        elif col in ['key', 'mode', 'time_signature']:
            df[col] = df[col].fillna(0).astype(int)
        else:  # Numeric columns
            df[col] = df[col].fillna(0.0)
# Now create metadata with no None values
metadatas = df[text_cols + num_cols].to_dict(orient="records")

# Add to ChromaDB
batch_size = 5000  # Safely below the 5461 limit
total_items = len(ids)
total_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division

print(f"\nAdding {total_items} records to ChromaDB in {total_batches} batches...")

for i in range(0, total_items, batch_size):
    end_idx = min(i + batch_size, total_items)
    batch_num = (i // batch_size) + 1
    print(f"Adding batch {batch_num}/{total_batches} ({end_idx - i} items)...")
    
    # Add the current batch
    collection.add(
        ids=ids[i:end_idx],
        embeddings=combined_embeddings.tolist()[i:end_idx],
        metadatas=metadatas[i:end_idx]
    )

print(f"Successfully added all {total_items} tracks to the vector database!")

# Sample query
query_text = "happy upbeat dance song"
query_text_embedding = model.encode([query_text])[0]  # 384 dims
query_num_features = np.array([[75, 180000, 0, 0.8, 0.9, 5, -5, 
                                1, 0.1, 0.2, 0.05, 0.3, 0.7, 120, 4]])

query_num_features = scaler.transform(query_num_features)  # Normalize with the
query_embedding = np.concatenate([query_text_embedding, query_num_features[0]])

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

for i, (id, distance, metadata) in enumerate(zip(results["ids"][0], results["distances"][0], results["metadatas"][0])):
    print(f"Match {i+1}:")
    print(f"ID: {id}")
    print(f"Distance: {distance}")
    print(f"Metadata: {metadata}")