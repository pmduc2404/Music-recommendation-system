import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

np.random.seed(42)

ds = load_dataset("maharshipandya/spotify-tracks-dataset")

df = ds['train'].to_pandas()

print(f"Dataset shape: {df.shape}")
print(f"Number of unique tracks: {df['track_id'].nunique()}")

# Fixed artists calculation
artist_count = 0
if 'artists' in df.columns:
    # Safer approach to count unique artists
    unique_artists = set()
    for artists_str in df['artists'].dropna():
        if isinstance(artists_str, str):
            for artist in artists_str.split(';'):
                unique_artists.add(artist.strip())
    artist_count = len(unique_artists)
print(f"Number of unique artists: {artist_count}")

print(f"Number of genres: {df['track_genre'].nunique()}")
print("\nMissing values per column:")
print(df.isnull().sum())

df['key'] = df['key'].fillna(-1)  # -1 is already used for "No key detected"
df['mode'] = df['mode'].fillna(0)  # Default to minor
df['time_signature'] = df['time_signature'].fillna(4)  # Default to 4/4
df['acousticness'] = df['acousticness'].fillna(df['acousticness'].mean())
df['instrumentalness'] = df['instrumentalness'].fillna(df['instrumentalness'].mean())
df['liveness'] = df['liveness'].fillna(df['liveness'].mean())
df['speechiness'] = df['speechiness'].fillna(df['speechiness'].mean())
df['tempo'] = df['tempo'].fillna(df['tempo'].mean())
df['duration_ms'] = df['duration_ms'].fillna(df['duration_ms'].mean())
df['valence'] = df['valence'].fillna(df['valence'].mean())
df['danceability'] = df['danceability'].fillna(df['danceability'].mean())
df['energy'] = df['energy'].fillna(df['energy'].mean())
df['loudness'] = df['loudness'].fillna(df['loudness'].mean())
df['popularity'] = df['popularity'].fillna(df['popularity'].mean())
df['explicit'] = df['explicit'].fillna(False)

numerical_features = [
    'popularity', 'duration_ms', 'danceability', 'energy', 
    'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]

scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df_embed = df[['track_name', 'artists', 'album_name', 'track_genre']]

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

# Generate embeddings for text fields
print("\nGenerating embeddings...")
track_embeddings = model.encode(df_embed["track_name"].tolist(), show_progress_bar=True)
artist_embeddings = model.encode(df_embed["artists"].tolist(), show_progress_bar=True)
album_embeddings = model.encode(df_embed["album_name"].tolist(), show_progress_bar=True)
genre_embeddings = model.encode(df_embed["track_genre"].tolist(), show_progress_bar=True)

# Create feature vectors combining text embeddings and audio features
print("\nCreating combined feature vectors...")

# Combine text embeddings with weighted average
# Give more weight to track name and genre as they're more relevant for mood matching
text_weights = [0.35, 0.25, 0.15, 0.25]  # track, artist, album, genre
text_embeddings = np.average(
    [track_embeddings, artist_embeddings, album_embeddings, genre_embeddings],
    axis=0,
    weights=text_weights
)

audio_features = df[numerical_features].values

from sklearn.decomposition import PCA
pca = PCA(n_components=min(11, audio_features.shape[1]))
audio_embeddings = pca.fit_transform(audio_features)

# Continue with the rest of your code as before
audio_embeddings = (audio_embeddings - audio_embeddings.min()) / (audio_embeddings.max() - audio_embeddings.min())

# Scale to match text embeddings influence (text is 384 dimensions vs audio's 64)
audio_weight = 0.4  # Adjust this to control influence of audio vs text features
text_weight = 1 - audio_weight

# Create final combined embeddings
# We need to pad audio_embeddings to match the dimensions of text_embeddings
padded_audio = np.zeros((audio_embeddings.shape[0], text_embeddings.shape[1]))
padded_audio[:, :audio_embeddings.shape[1]] = audio_embeddings * (text_embeddings.shape[1] / audio_embeddings.shape[1])

# Final weighted combination
combined_embeddings = text_weight * text_embeddings + audio_weight * padded_audio

print(f"Combined embeddings shape: {combined_embeddings.shape}")

print("\nInitializing ChromaDB...")
client = chromadb.PersistentClient(path="./spotify_chroma_db")

collection_name = "spotify_music_collection"
try:
    collection = client.create_collection(name=collection_name)
    print(f"Created new collection: {collection_name}")
except:
    collection = client.get_collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")

# Prepare metadata
print("\nPreparing metadata for vector database...")
metadata = []
for idx, row in df.iterrows():
    metadata.append({
        "track_id": str(row["track_id"]),
        "track_name": str(row["track_name"]),
        "artists": str(row["artists"]),
        "album_name": str(row["album_name"]),
        "track_genre": str(row["track_genre"]),
        "popularity": float(row["popularity"]),
        "danceability": float(row["danceability"]),
        "energy": float(row["energy"]),
        "valence": float(row["valence"]),
        "tempo": float(row["tempo"]),
        "acousticness": float(row["acousticness"]),
        "instrumentalness": float(row["instrumentalness"])
    })

# Generate IDs
ids = [f"song_{i}" for i in range(len(df))]

print("\nAdding data to ChromaDB...")
batch_size = 5000  # Safely below the 5461 limit
total_items = len(ids)
total_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division

for i in range(0, total_items, batch_size):
    end_idx = min(i + batch_size, total_items)
    print(f"Adding batch {(i // batch_size) + 1}/{total_batches} ({end_idx - i} items)...")
    
    # Add the current batch
    collection.add(
        ids=ids[i:end_idx],
        embeddings=combined_embeddings.tolist()[i:end_idx],
        metadatas=metadata[i:end_idx]
    )

print(f"\nSuccessfully added {total_items} tracks to the vector database!")

# Create a mood classifier based on valence and energy
def classify_mood(valence, energy):
    if valence >= 0.5 and energy >= 0.5:
        return "happy/energetic"
    elif valence >= 0.5 and energy < 0.5:
        return "relaxed/positive"
    elif valence < 0.5 and energy >= 0.5:
        return "angry/intense"
    else:
        return "sad/melancholic"

# Add mood classification to the dataframe
df["mood"] = df.apply(lambda x: classify_mood(x["valence"], x["energy"]), axis=1)

# Plot the mood distribution
plt.figure(figsize=(10, 6))
sns.countplot(x="mood", data=df)
plt.title("Distribution of Mood Classifications")
plt.tight_layout()
plt.savefig("mood_distribution.png")

# Test query
print("\nTesting a sample query...")
test_query = "energetic happy pop music"
test_embedding = model.encode([test_query])[0]

results = collection.query(
    query_embeddings=[test_embedding.tolist()],
    n_results=5,
    include=["metadatas"]
)

print("\nTop 5 results for query 'energetic happy pop music':")
for i, metadata in enumerate(results["metadatas"][0]):
    print(f"{i+1}. {metadata['track_name']} by {metadata['artists']} - Genre: {metadata['track_genre']}")
    print(f"   Valence: {metadata['valence']:.2f}, Energy: {metadata['energy']:.2f}, Danceability: {metadata['danceability']:.2f}")
    print()

print("Preprocessing and vector database creation complete!")