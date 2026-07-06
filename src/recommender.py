"""
Movie Recommendation System
Implements collaborative filtering and content-based filtering algorithms
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class MovieRecommender:
    """
    A class to build movie recommendations using multiple algorithms
    """
    
    def __init__(self, ratings_df, movies_df):
        """
        Initialize the recommender system
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            DataFrame with columns: userId, movieId, rating
        movies_df : pd.DataFrame
            DataFrame with columns: movieId, title, genres
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.user_item_matrix = None
        self.similarity_matrix = None
        
    def create_user_item_matrix(self):
        """Create user-item matrix from ratings"""
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        return self.user_item_matrix
    
    def collaborative_filtering(self, user_id, n_recommendations=10):
        """
        Collaborative Filtering - User-based approach
        
        Parameters:
        -----------
        user_id : int
            The user ID for which to generate recommendations
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        pd.DataFrame : Top N recommended movies
        """
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Calculate user similarity using cosine similarity
        user_similarity = cosine_similarity(self.user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Get similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Get movies rated by similar users but not by target user
        user_rated = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values)
        recommendations = {}
        
        for similar_user, similarity in similar_users.items():
            similar_user_rated = self.ratings_df[self.ratings_df['userId'] == similar_user]
            
            for _, row in similar_user_rated.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                
                if movie_id not in user_rated:
                    if movie_id not in recommendations:
                        recommendations[movie_id] = 0
                    recommendations[movie_id] += similarity * rating
        
        # Sort and get top N
        top_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        # Get movie details
        movie_ids = [rec[0] for rec in top_recommendations]
        scores = [rec[1] for rec in top_recommendations]
        
        result_df = self.movies_df[self.movies_df['movieId'].isin(movie_ids)].copy()
        result_df['recommendation_score'] = result_df['movieId'].map(
            dict(zip(movie_ids, scores))
        )
        result_df = result_df.sort_values('recommendation_score', ascending=False)
        
        return result_df[['movieId', 'title', 'genres', 'recommendation_score']]
    
    def content_based_filtering(self, movie_id, n_recommendations=10):
        """
        Content-Based Filtering using genre similarity
        
        Parameters:
        -----------
        movie_id : int
            The movie ID to find similar movies for
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        pd.DataFrame : Top N similar movies
        """
        # Create genre matrix
        genres = self.movies_df['genres'].str.get_dummies(sep='|')
        similarity = cosine_similarity(genres)
        similarity_df = pd.DataFrame(
            similarity,
            index=self.movies_df['movieId'],
            columns=self.movies_df['movieId']
        )
        
        # Get similar movies
        similar_scores = similarity_df[movie_id].sort_values(ascending=False)[1:n_recommendations+1]
        similar_movie_ids = similar_scores.index.tolist()
        
        result_df = self.movies_df[self.movies_df['movieId'].isin(similar_movie_ids)].copy()
        result_df['similarity_score'] = result_df['movieId'].map(dict(similar_scores))
        result_df = result_df.sort_values('similarity_score', ascending=False)
        
        return result_df[['movieId', 'title', 'genres', 'similarity_score']]
    
    def get_movie_info(self, movie_id):
        """Get information about a specific movie"""
        return self.movies_df[self.movies_df['movieId'] == movie_id]
    
    def get_top_rated_movies(self, n=10):
        """Get top-rated movies by average rating"""
        avg_ratings = self.ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])
        avg_ratings = avg_ratings[avg_ratings['count'] > 5].sort_values('mean', ascending=False)
        top_movies = avg_ratings.head(n).index.tolist()
        
        return self.movies_df[self.movies_df['movieId'].isin(top_movies)]
