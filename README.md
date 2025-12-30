# ShakeIt Music Recommendation Terminal

ShakeIt is a terminal-based music recommendation system that suggests songs based on audio features, analyzes your listening history, and visualizes similarities between tracks. Built with Python and designed for easy exploration of large music datasets.

## Features
- Search and play songs (opens YouTube/Spotify in browser)
- Recommend songs based on your listening history (taste analysis)
- Visualize feature similarities with radar charts
- PCA-based cluster visualization (optional)
- Keeps a listening history log

## Setup
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd veriBilimi
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset file (e.g. `tracks_features.csv`) in the project folder. The dataset should have columns like:
   - `id`, `name`, `artists`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

## Usage
Run the main program:
```bash
python run.py
```
Follow the interactive menu to search, play, get recommendations, and visualize.

## Data
- **tracks_features.csv**: Main dataset (audio features, track info)
- **listening_history.csv**: Automatically created to log played tracks

## Example
```
=== ShakeIt: MUSIC RECOMMENDATION TERMINAL ===
1. Search and Play Song
2. Recommend Based on My History (Taste Analysis)
3. Visualize: Analyze Last Recommendation (Radar Chart)
4. Exit
```

## License
MIT

## Author
Mehmet Emin Ateş
