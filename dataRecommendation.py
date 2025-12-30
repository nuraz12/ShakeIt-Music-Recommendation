import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
import webbrowser
import urllib.parse
HISTORY_FILE = 'listening_history.csv'

def log_play(track_id):
    """Add the played song to the listening history file."""
    new_data = pd.DataFrame([[track_id, pd.Timestamp.now()]], columns=['track_id', 'timestamp'])
    
    # If file doesn't exist, create with header; else, append without header
    if not os.path.isfile(HISTORY_FILE):
        new_data.to_csv(HISTORY_FILE, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, mode='a', index=False, header=False)
    print(f"--- Log: {track_id} added to history. ---")
def get_user_profile(data, scaled_data):
    """Calculate the user's average taste vector based on listening history."""
    if not os.path.isfile(HISTORY_FILE):
        return None
    
    history = pd.read_csv(HISTORY_FILE)
    if history.empty:
        return None
        
    # Find indices of songs in history
    history_indices = data[data['track_id'].isin(history['track_id'])].index
    
    # Take the mean of scaled features for these songs
    user_profile = scaled_data.iloc[history_indices].mean(axis=0)
    return user_profile.values.reshape(1, -1)

def analyze_user_taste(data, scaled_data):
    """Analyze and report the user's music taste based on listening history."""
    if not os.path.isfile(HISTORY_FILE):
        print("No listening history yet.")
        return None
    
    history = pd.read_csv(HISTORY_FILE)
    if history.empty:
        print("Listening history is empty.")
        return None
        
    # Find indices of songs in history
    history_indices = data[data['track_id'].isin(history['track_id'])].index
    
    if len(history_indices) == 0:
        print("Songs in history not found in dataset.")
        return None

    print(f"\n📊 Taste Analysis ({len(history_indices)} songs):")
    
    # Calculate mean features
    mean_features = scaled_data.iloc[history_indices].mean(axis=0)
    
    # Visualize features
    for feature, value in mean_features.items():
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{feature.ljust(15)}: {bar} {value:.2f}")
        
    return mean_features.values.reshape(1, -1)

def recommend_from_profile(user_profile, data, scaled_data, top_n=5):
    """Recommend songs using the user's average taste profile."""
    # Calculate similarity between profile and all songs
    similarity_scores = cosine_similarity(user_profile, scaled_data)[0]
    
    # Add similarity scores and sort
    data_copy = data.copy()
    data_copy['similarity'] = similarity_scores
    
    # Filter out songs already listened to
    history = pd.read_csv(HISTORY_FILE)
    data_copy = data_copy[~data_copy['track_id'].isin(history['track_id'])]
    
    top_recommendations = data_copy.sort_values(by='similarity', ascending=False).head(top_n)
    return top_recommendations
def recommend_songs(song_name, data, scaled_data):
    # Check if the song name exists in the dataset
    if song_name not in data['track_name'].values:
        return f"Error: '{song_name}' not found in the dataset."

    # Find the index of the song (if there are multiple, take the first)
    song_indices = data[data['track_name'] == song_name].index
    if len(song_indices) == 0:
         return f"Error: '{song_name}' not found in the dataset."
    song_index = song_indices[0]

    # Get the feature vector of the target song
    target_song_features = scaled_data.iloc[[song_index]]

    # Calculate cosine similarity between the target song and all other songs
    # This returns a (1, n_samples) array
    similarity_scores_vector = cosine_similarity(target_song_features, scaled_data)

    # Flatten to a 1D array
    similarity_scores = similarity_scores_vector[0]

    # Get indices of sorted scores (descending)
    # We use argsort which sorts ascending, so we take the last ones
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Get the top 5 highest scoring songs (The first one is the song itself, so take 1 to 6)
    top_5_indices = sorted_indices[1:6]

    # Return song names and indices
    return data['track_name'].iloc[top_5_indices], top_5_indices

def plot_radar_chart(song_name, data, scaled_data, top_5_indices):
    # Get the index of the original song
    song_index = data[data['track_name'] == song_name].index[0]
    
    # Get features for the original song and the first recommended song
    categories = list(scaled_data.columns)
    N = len(categories)
    
    # Original song values
    values_original = scaled_data.iloc[song_index].values.flatten().tolist()
    values_original += values_original[:1] # Close the loop
    
    # First recommended song values
    values_recommended = scaled_data.iloc[top_5_indices[0]].values.flatten().tolist()
    values_recommended += values_recommended[:1] # Close the loop
    
    # Calculate angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw original song
    ax.plot(angles, values_original, linewidth=2, linestyle='solid', label=f'Original: {song_name}')
    ax.fill(angles, values_original, 'b', alpha=0.1)
    
    # Draw recommended song
    recommended_song_name = data['track_name'].iloc[top_5_indices[0]]
    ax.plot(angles, values_recommended, linewidth=2, linestyle='solid', label=f'Recommended: {recommended_song_name}', color='orange')
    ax.fill(angles, values_recommended, 'orange', alpha=0.1)
    
    # Labels
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    plt.title(f"Feature Comparison: {song_name} vs {recommended_song_name}", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

def plot_pca_clusters(scaled_data, song_index, top_5_indices):
    # Reduce dimensions to 2D using PCA
    # For visualization, we don't need to fit on the WHOLE dataset if it's huge.
    # But to be accurate, we should. PCA on 1.2M rows x 8 cols is fast enough.
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    
    plt.figure(figsize=(10, 8))
    
    # Plot a sample of songs as small grey dots to avoid performance issues
    # If dataset is large (> 10k), sample it.
    n_samples = len(scaled_data)
    if n_samples > 10000:
        sample_indices = np.random.choice(n_samples, 10000, replace=False)
        plt.scatter(components[sample_indices, 0], components[sample_indices, 1], alpha=0.1, color='grey', label='Other Songs (Sampled)')
    else:
        plt.scatter(components[:, 0], components[:, 1], alpha=0.1, color='grey', label='All Songs')
    
    # Plot original song (Blue)
    plt.scatter(components[song_index, 0], components[song_index, 1], color='blue', s=100, label='Original Song')
    
    # Plot recommended songs (Red)
    plt.scatter(components[top_5_indices, 0], components[top_5_indices, 1], color='red', s=100, label='Recommended Songs')
    
    plt.title('Song Similarity Clusters (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

def search_song(query, data):
    """Search for songs by name (case-insensitive)."""
    # Search in track_name column
    mask = data['track_name'].str.contains(query, case=False, na=False)
    results = data[mask]['track_name'].unique()
    return results


def play_song(artist, track, platform="youtube"):
    """Search for the selected song online and open it in the browser."""
    query = f"{artist} {track}"
    encoded_query = urllib.parse.quote_plus(query) # Replace spaces with +
    
    if platform == "youtube":
        url = f"https://www.youtube.com/results?search_query={encoded_query}"
    elif platform == "spotify":
        url = f"https://open.spotify.com/search/{encoded_query}"
    else:
        url = f"https://www.google.com/search?q={encoded_query}+listen"
    
    print(f"--- {track} is opening... ---")
    webbrowser.open(url) # Automatically opens browser

