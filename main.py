import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from spotipy.cache_handler import MemoryCacheHandler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
import os
from tqdm import tqdm

# Replace these with your actual Spotify API credentials
CLIENT_ID = os.environ['SP_CLIENT_ID']
CLIENT_SECRET = os.environ['SP_CLIENT_SECRET']
REDIRECT_URI = 'https://localhost:8080/'  # Make sure this URI is set in your Spotify app settings
# Define the scope
scope = "user-library-read"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               cache_handler=MemoryCacheHandler(),
                                               scope=scope))

# Initialize list to keep track of all liked songs
liked_songs = []

# Get liked songs; loop through pages if there are more than 50
results = sp.current_user_saved_tracks(limit=50)
while results:
    liked_songs.extend(results['items'])
    if results['next']:
        results = sp.next(results)
    else:
        results = None

# Extract track IDs
track_ids = [item['track']['id'] for item in liked_songs]

# Function to retrieve audio features in batches
def get_audio_features_in_batches(track_ids, batch_size=100):
    audio_features = []
    valid_track_ids = []  # To store track_ids where audio features are available
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        features = sp.audio_features(batch)
        for track_id, feature in zip(batch, features):
            if feature is not None:  # Only add if features are present
                audio_features.append(feature)
                valid_track_ids.append(track_id)
    return audio_features, valid_track_ids

# Retrieve audio features for each track in batches of 100
audio_features, valid_track_ids = get_audio_features_in_batches(track_ids)

# Create a DataFrame from the filtered audio features
audio_features_df = pd.DataFrame(audio_features)
features_list = ['danceability', 'energy', 'loudness',
                 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo']

# Scale the audio features
audio_features_scaled = StandardScaler().fit_transform(audio_features_df[features_list])

track_artist_map = {}

# Populate the track_artist_map with track ID and artist name(s)
for item in tqdm(liked_songs, total=len(liked_songs),desc='Fetching liked Songs'):
    track_id = item['track']['id']
    artist_names = [artist['id'] for artist in item['track']['artists']]
    track_artist_map[track_id] = artist_names


# Initialize SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a dictionary to hold artist genres
artist_genre_map = {}
artist_set = set()

# Iterate over each liked track to collect genres
for item in tqdm(liked_songs, total=len(liked_songs),desc='Fetching Artist Data'):
    for artist in item['track']['artists']:
        artist_id = artist['id']
        artist_set.add(artist_id)

for artist_id in tqdm(artist_set, total=len(artist_set), desc='Fetching Genre Data'):
        artist_info = sp.artist(artist_id)
        genres = artist_info['genres']
        artist_genre_map[artist_id] = genres

# Generate embeddings for concatenated genres
artist_embeddings_dict = {}
for artist_id, genres in tqdm(artist_genre_map.items(),total= len(artist_genre_map), desc= 'Making Artist Embeddings'):
    # Concatenate genres into a single string
    genre_str = ', '.join(genres)
    # Compute the sentence embedding
    embedding = model.encode(genre_str)
    # Store it in the dictionary
    artist_embeddings_dict[artist_id] = embedding



# Compute and align track embeddings
track_embeddings = []
for track, artists in track_artist_map.items():
    embeddings = [artist_embeddings_dict[artist_id] for artist_id in artists]
    avg_embedding = np.mean(embeddings, axis=0)
    track_embeddings.append(avg_embedding)

# Convert to NumPy array for t-SNE processing:
track_embeddings = np.array(track_embeddings)

print('Reducing Embedding Dimensions')
# Dimensionality reduction using t-SNE with exact method
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
track_embeddings_reduced = tsne.fit_transform(track_embeddings)

# Standardize the reduced embeddings
track_embeddings_scaled = StandardScaler().fit_transform(track_embeddings_reduced)

# Since valid_track_ids was used to filter both data sources, lengths should match
combined_features = np.hstack((audio_features_scaled, track_embeddings_scaled))

print('Clustering')
# Clustering combined features
kmeans_combined = KMeans(n_clusters=10, random_state=42).fit(combined_features)
combined_cluster_labels = kmeans_combined.labels_

# Print or further process clusters
print("Combined feature-based clusters:")
print(combined_cluster_labels)

# Assuming you have a list of track names and a list of artist names
track_names = [item['track']['name'] for item in liked_songs]
artist_names = [", ".join([artist['name'] for artist in item['track']['artists']]) for item in liked_songs]

# Ensure valid_track_ids is used to align any filtering necessary from earlier
filtered_track_names = [track_names[i] for i in range(len(track_ids)) if track_ids[i] in valid_track_ids]
filtered_artist_names = [artist_names[i] for i in range(len(track_ids)) if track_ids[i] in valid_track_ids]

# Combine this metadata with cluster labels
clusters_data = pd.DataFrame({
    'Track Name': filtered_track_names,
    'Artist(s)': filtered_artist_names,
    'Cluster Label': combined_cluster_labels
})

# Display the songs in each cluster
for cluster in range(kmeans_combined.n_clusters):
    print(f"\nSongs in Cluster {cluster}:")
    cluster_songs = clusters_data[clusters_data['Cluster Label'] == cluster]
    for index, song in cluster_songs.iterrows():
        print(f"- {song['Track Name']} by {song['Artist(s)']}")
