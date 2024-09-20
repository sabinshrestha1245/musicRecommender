import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MusicRecommenderFromScratch:
    def __init__(self, data_file):
        # Load the CSV dataset and preprocess it during startup
        self.data = pd.read_csv(data_file)

        # Combine relevant textual features: song title, artist name(s), genres, album name
        self.data['combined_textual_features'] = self.data.apply(
            lambda row: f"{row['Track Name']} {row['Artist Name(s)']} {row['Artist Genres']} {row['Album Name']}".lower(),
            axis=1
        )

        # Precompute the TF-IDF vectors and normalize numerical features for all songs
        self.precompute_vectors()

    def precompute_vectors(self):
        """Precompute TF-IDF vectors for textual features and scale numerical features."""
        # Initialize the TF-IDF vectorizer for textual features
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Compute TF-IDF vectors for all the songs' combined textual features
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_textual_features'])

        # Normalize the numerical audio features (Danceability, Energy, Tempo, Valence, etc.)
        numerical_features = ['Danceability', 'Energy', 'Tempo', 'Valence', 'Loudness', 'Popularity']
        scaler = MinMaxScaler()  # Scale between 0 and 1
        self.numerical_matrix = scaler.fit_transform(self.data[numerical_features].fillna(0))

    def get_recommendations(self, query, top_n=5):
        """Find song recommendations based on cosine similarity of both textual and numerical features."""
        # Preprocess and vectorize the query for textual features
        query = query.lower()
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between the query and the songs' textual features
        text_similarity = cosine_similarity(query_vector, self.tfidf_matrix)

        # If numerical features are important, apply cosine similarity on numerical data as well
        query_num = np.zeros((1, self.numerical_matrix.shape[1]))  # No numeric vector for the query, assume neutral
        num_similarity = cosine_similarity(query_num, self.numerical_matrix)

        # Combine both textual and numerical similarities (tweak weights based on importance)
        combined_similarity = 0.7 * text_similarity + 0.3 * num_similarity

        # Get the similarity scores for each song and sort them
        similarity_scores = list(enumerate(combined_similarity[0]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the top N recommendations, converting numpy types to native Python types
        top_recommendations = [
            {
                'Track Name': str(self.data.iloc[idx]['Track Name']),
                'Track Web URL': f"https://open.spotify.com/track/{self.data.iloc[idx]['Track URI'].split(':')[-1]}",
                'Track URI': f"spotify:track:{self.data.iloc[idx]['Track URI'].split(':')[-1]}",
                'Artist Name(s)': str(self.data.iloc[idx]['Artist Name(s)']),
                'Album Name': str(self.data.iloc[idx]['Album Name']),
                'Album Release Date': str(self.data.iloc[idx]['Album Release Date']),
                'Album Image URL': str(self.data.iloc[idx]['Album Image URL']),
                'Popularity': int(self.data.iloc[idx]['Popularity']),
                'Danceability': float(self.data.iloc[idx]['Danceability']),
                'Energy': float(self.data.iloc[idx]['Energy']),
                'Valence': float(self.data.iloc[idx]['Valence']),
                'Tempo': float(self.data.iloc[idx]['Tempo']),
            }
            for idx, score in similarity_scores[:top_n]
        ]

        return top_recommendations

    def get_popular_music(self, top_n=10):
        """Get the top N most popular music tracks."""
        popular_music = self.data.nlargest(top_n, 'Popularity')

        # Format the result
        return [
            {
                'Track Name': str(row['Track Name']),
                'Track Web URL': f"https://open.spotify.com/track/{row['Track URI'].split(':')[-1]}",
                'Track URI': f"spotify:track:{row['Track URI'].split(':')[-1]}",
                'Artist Name(s)': str(row['Artist Name(s)']),
                'Album Name': str(row['Album Name']),
                'Album Release Date': str(row['Album Release Date']),
                'Album Image URL': str(row['Album Image URL']),
                'Popularity': int(row['Popularity']),
            }
            for _, row in popular_music.iterrows()
        ]

    def get_latest_music(self, top_n=10):
        """Get the top N latest music tracks based on album release date."""
        latest_music = self.data.sort_values(by='Album Release Date', ascending=False).head(top_n)

        # Format the result
        return [
            {
                'Track Name': str(row['Track Name']),
                'Track Web URL': f"https://open.spotify.com/track/{row['Track URI'].split(':')[-1]}",
                'Track URI': f"spotify:track:{row['Track URI'].split(':')[-1]}",
                'Artist Name(s)': str(row['Artist Name(s)']),
                'Album Name': str(row['Album Name']),
                'Album Release Date': str(row['Album Release Date']),
                'Album Image URL': str(row['Album Image URL']),
                'Popularity': int(row['Popularity']),
            }
            for _, row in latest_music.iterrows()
        ]
