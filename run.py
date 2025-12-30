import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataRecommendation import log_play, get_user_profile, recommend_from_profile, search_song, plot_radar_chart, analyze_user_taste, recommend_songs, play_song
if __name__ == "__main__":
    # 1. Load the dataset (from Kaggle or local file)
    df = pd.read_csv('tracks_features.csv') 
    
    # Fix column names for tracks_features.csv
    df.rename(columns={
        'id': 'track_id',
        'name': 'track_name',
        'artists': 'artist_name'
    }, inplace=True)
    
    # 2. Select features and scale them
    # Using columns from the correlation matrix
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Variables outside the loop
    selected_name = None
    last_recs_indices = None

    while True:
        print("\n=== ShakeIt: MUSIC RECOMMENDATION TERMINAL ===")
        print("1. Search and Play Song")
        print("2. Recommend Based on My History (Taste Analysis)")
        print("3. Visualize: Analyze Last Recommendation (Radar Chart)")
        print("4. Exit")
        
        choice = input("Your choice: ")

        if choice == '1':
            query = input("Song to search: ")
            results = search_song(query, df)
            
            if len(results) > 0:
                for i, res in enumerate(results[:5]): # First 5 results
                    print(f"{i}: {res}")
                
                try:
                    pick = int(input("Select the number of the song you want to play: "))
                    if 0 <= pick < len(results):
                        selected_name = results[pick]
                        
                        # Find track_id and artist
                        selected_row = df[df['track_name'] == selected_name].iloc[0]
                        track_id = selected_row['track_id']
                        artist_name = selected_row['artist_name']
                        
                        # Clean artist name from list characters (['Artist'] -> Artist)
                        artist_name = artist_name.replace("['", "").replace("]'", "").replace("'", "")

                        log_play(track_id)
                        play_song(artist_name, selected_name)

                        # Prepare recommendations for option 3 in the background
                        print("(Preparing recommendations for this song in the background...)")
                        _, last_recs_indices = recommend_songs(selected_name, df, df_scaled)
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a number.")
            else:
                print("Sorry, song not found.")

        elif choice == '2':
            print("Analyzing your profile...")
            # Using analyze_user_taste instead of get_user_profile
            profile = analyze_user_taste(df, df_scaled)
            if profile is not None:
                recs = recommend_from_profile(profile, df, df_scaled)
                print("\nSongs with a similar 'vibe' selected for you:")
                print(recs[['track_name', 'artist_name', 'similarity']])
                last_recs_indices = recs.index.tolist() # Save for visualization
                # In profile-based recommendation, 'selected_name' may not exist, so radar chart may not work.
                # Inform the user or use the last played song as reference.
                if selected_name is None:
                    print("\nNote: For radar chart (Option 3), it's recommended to select a reference song first (Option 1).")
            else:
                print("No listening history yet. Please play some songs first!")

        elif choice == '3':
            # Get a profile first to know what to visualize
            if selected_name is not None and last_recs_indices is not None:
                plot_radar_chart(selected_name, df, df_scaled, last_recs_indices)
            else:
                print("\n⚠️ Error: To visualize, you must first select a song (Option 1).")
                print("If you only used Option 2, there is no reference song for comparison.")

        elif choice == '4':
            print("See you soon!")
            break