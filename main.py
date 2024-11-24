import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from spotipy.cache_handler import  MemoryCacheHandler
# Display the DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import numpy as np

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
                                               cache_handler = MemoryCacheHandler(),
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

# Retrieve audio features for each track
audio_features = sp.audio_features(track_ids)

# Extract and scale audio features such as danceability, energy, etc.
features_list = ['danceability', 'energy', 'key', 'loudness', 'mode',
                 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                 'valence', 'tempo']
audio_features_df = pd.DataFrame(audio_features)
audio_features_scaled = StandardScaler().fit_transform(audio_features_df[features_list])

# Extract artist IDs from tracks
artist_ids = set()  # Use a set to avoid duplicates
track_artist_map = []  # To map tracks with their artists
for item in liked_songs:
    track = item['track']
    track_artists = [artist['id'] for artist in track['artists']]
    track_artist_map.append(track_artists)
    for artist in track_artists:
        artist_ids.add(artist)

# Retrieve genres for each artist
artist_genres = {}
for artist_id in artist_ids:
    artist_info = sp.artist(artist_id)
    artist_genres[artist_info['id']] = artist_info['genres']

# Compute artist embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
artist_embeddings_dict = {artist_id: model.encode(', '.join(genres) + ' music') for artist_id, genres in artist_genres.items()}

# Map track to their average artist embeddings
track_embeddings = []
for track_artists in track_artist_map:
    embeddings = [artist_embeddings_dict[artist_id] for artist_id in track_artists]
    avg_embedding = np.mean(embeddings, axis=0)
    track_embeddings.append(avg_embedding)

# Standardizing the artist embeddings
track_embeddings_scaled = StandardScaler().fit_transform(track_embeddings)

# Concatenate audio features and artist embeddings
combined_features = np.hstack((audio_features_scaled, track_embeddings_scaled))

# Clustering combined features
kmeans_combined = KMeans(n_clusters=5, random_state=42).fit(combined_features)
combined_cluster_labels = kmeans_combined.labels_

# Print or further process clusters
print("Combined feature-based clusters:")
print(combined_cluster_labels)
