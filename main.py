import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from spotipy.cache_handler import  MemoryCacheHandler
# Display the DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

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

# Fetch audio features in batches (Spotify API may limit batch sizes)
audio_features_list = []
for i in range(0, len(track_ids), 100):  # The maximum batch size for Spotify's API is 100
    audio_features_list.extend(sp.audio_features(track_ids[i:i+100]))

# Create a DataFrame with detailed metadata about each song
songs_data = []
for item, audio_features in zip(liked_songs, audio_features_list):
    if audio_features:
        song_info = {
            'Track Name': item['track']['name'],
            'Artist': ', '.join(artist['name'] for artist in item['track']['artists']),
            'Album': item['track']['album']['name'],
            'Release Date': item['track']['album']['release_date'],
            'Duration (ms)': item['track']['duration_ms'],
            'Popularity': item['track']['popularity'],
            'Track ID': item['track']['id'],
            'Spotify URL': item['track']['external_urls']['spotify'],
            'Album Type': item['track']['album']['album_type'],
            'Total Tracks in Album': item['track']['album']['total_tracks'],
            'Danceability': audio_features['danceability'],
            'Energy': audio_features['energy'],
            'Key': audio_features['key'],
            'Loudness': audio_features['loudness'],
            'Mode': audio_features['mode'],
            'Speechiness': audio_features['speechiness'],
            'Acousticness': audio_features['acousticness'],
            'Instrumentalness': audio_features['instrumentalness'],
            'Liveness': audio_features['liveness'],
            'Valence': audio_features['valence'],
            'Tempo': audio_features['tempo']
        }
        songs_data.append(song_info)

df = pd.DataFrame(songs_data)


# Select relevant audio features for clustering
audio_features = ['Danceability', 'Energy', 'Loudness', 
                  'Speechiness', 'Acousticness', 'Instrumentalness', 
                  'Liveness', 'Valence', 'Tempo']

n_cluster = 15
# Standardize the audio features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[audio_features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_cluster, random_state=42)  # Adjust n_clusters as needed
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Optional: Analyze the distribution of songs per cluster
print(df['Cluster'].value_counts())

# Determine number of top songs to display from each cluster
top_n = 5

# Print the top songs from each cluster
for cluster in range(n_cluster):  # Change 10 to your number of clusters
    print(f"\nTop {top_n} Songs from Cluster {cluster}:")
    cluster_songs = df[df['Cluster'] == cluster]
    top_songs = cluster_songs.sort_values(by='Popularity', ascending=False).head(top_n)
    print(top_songs[['Track Name', 'Artist', 'Popularity']])
