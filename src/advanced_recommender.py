"""
Advanced Movie Recommendation System
Matrix Factorization using SVD and NMF
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class MatrixFactorization:
    """
    Matrix Factorization Recommender using SVD and NMF
    
    Advanced algorithms for better recommendation accuracy
    """
    
    def __init__(self, method='svd', n_factors=50, random_state=42):
        """
        Initialize Matrix Factorization
        
        Parameters:
        -----------
        method : str, 'svd' or 'nmf'
            Decomposition method
        n_factors : int
            Number of latent factors
        random_state : int
            Random seed
        """
        self.method = method
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_item_matrix = None
        self.decomposition = None
        self.user_factors = None
        self.item_factors = None
        self.scaler = MinMaxScaler()
        
    def fit(self, user_item_matrix):
        """
        Fit the matrix factorization model
        
        Parameters:
        -----------
        user_item_matrix : pd.DataFrame or np.ndarray
            User-item rating matrix
            
        Returns:
        --------
        self
        """
        self.user_item_matrix = user_item_matrix.values if isinstance(user_item_matrix, pd.DataFrame) else user_item_matrix
        
        # Normalize the matrix
        self.user_item_matrix = self.scaler.fit_transform(self.user_item_matrix)
        
        # Handle zeros (missing values)
        self.user_item_matrix[self.user_item_matrix == 0] = np.nan
        
        if self.method == 'svd':
            self._fit_svd()
        elif self.method == 'nmf':
            self._fit_nmf()
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def _fit_svd(self):
        """Fit SVD decomposition"""
        # Fill NaN with mean
        matrix_filled = np.nan_to_num(self.user_item_matrix, nan=np.nanmean(self.user_item_matrix))
        
        svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.user_factors = svd.fit_transform(matrix_filled)
        
        # Item factors (components)
        self.item_factors = svd.components_.T
        
    def _fit_nmf(self):
        """Fit NMF decomposition"""
        # Fill NaN with 0 (NMF requires non-negative values)
        matrix_filled = np.nan_to_num(self.user_item_matrix, nan=0)
        matrix_filled = np.maximum(matrix_filled, 0)  # Ensure non-negative
        
        nmf = NMF(n_components=self.n_factors, random_state=self.random_state, init='random')
        self.user_factors = nmf.fit_transform(matrix_filled)
        self.item_factors = nmf.components_.T
        
    def predict(self, user_id, n_recommendations=10):
        """
        Predict and recommend movies for a user
        
        Parameters:
        -----------
        user_id : int
            User index (0-based)
        n_recommendations : int
            Number of recommendations
            
        Returns:
        --------
        np.ndarray : Top item indices
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted yet")
            
        # Get user factor vector
        user_vector = self.user_factors[user_id]
        
        # Predict ratings for all items
        predictions = np.dot(user_vector, self.item_factors.T)
        
        # Get top N recommendations (excluding already rated items)
        already_rated = np.where(~np.isnan(self.user_item_matrix[user_id]))[0]
        predictions[already_rated] = -np.inf
        
        top_indices = np.argsort(predictions)[-n_recommendations:][::-1]
        
        return top_indices, predictions[top_indices]
    
    def get_user_similarity(self):
        """Get user similarity matrix using latent factors"""
        return cosine_similarity(self.user_factors)
    
    def get_item_similarity(self):
        """Get item similarity matrix using latent factors"""
        return cosine_similarity(self.item_factors)
    
    def save_model(self, filepath):
        """Save model to disk"""
        model_data = {
            'method': self.method,
            'n_factors': self.n_factors,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'scaler': self.scaler
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.method = model_data['method']
        self.n_factors = model_data['n_factors']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.scaler = model_data['scaler']
        return self


class HybridRecommender:
    """
    Hybrid Recommendation System
    Combines multiple recommendation strategies
    """
    
    def __init__(self, ratings_df, movies_df, weights=None):
        """
        Initialize Hybrid Recommender
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings data
        movies_df : pd.DataFrame
            Movies data
        weights : dict
            Weights for different algorithms
            {'collaborative': 0.5, 'content': 0.3, 'matrix_fact': 0.2}
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        self.weights = weights or {
            'collaborative': 0.4,
            'content': 0.3,
            'matrix_fact': 0.3
        }
        
        self.user_item_matrix = None
        self.mf_model = None
        
    def fit(self):
        """Fit all models"""
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Fit matrix factorization model
        self.mf_model = MatrixFactorization(method='svd', n_factors=50)
        self.mf_model.fit(self.user_item_matrix)
        
        return self
    
    def get_hybrid_recommendations(self, user_id, movie_id_map, n_recommendations=10):
        """
        Get hybrid recommendations combining multiple approaches
        
        Parameters:
        -----------
        user_id : int
            User ID
        movie_id_map : dict
            Mapping of movieId to index
        n_recommendations : int
            Number of recommendations
            
        Returns:
        --------
        pd.DataFrame : Recommended movies with scores
        """
        scores = {}
        
        # 1. Collaborative Filtering Score
        cf_scores = self._collaborative_filtering_score(user_id, movie_id_map)
        
        # 2. Content-Based Score
        content_scores = self._content_based_score(user_id, movie_id_map)
        
        # 3. Matrix Factorization Score
        mf_scores = self._matrix_factorization_score(user_id, movie_id_map)
        
        # Combine scores
        for movie_id in movie_id_map.values():
            score = (
                self.weights['collaborative'] * cf_scores.get(movie_id, 0) +
                self.weights['content'] * content_scores.get(movie_id, 0) +
                self.weights['matrix_fact'] * mf_scores.get(movie_id, 0)
            )
            scores[movie_id] = score
        
        # Get top recommendations
        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        result_movies = []
        for movie_id, score in top_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                result_movies.append({
                    'movieId': movie_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'hybrid_score': score
                })
        
        return pd.DataFrame(result_movies)
    
    def _collaborative_filtering_score(self, user_id, movie_id_map):
        """Calculate collaborative filtering scores"""
        user_similarity = cosine_similarity(self.user_item_matrix.values)
        user_idx = user_id - 1  # Convert to 0-based index
        
        if user_idx < 0 or user_idx >= len(user_similarity):
            return {}
        
        similar_users = user_similarity[user_idx]
        scores = {}
        
        for movie_id, col_idx in movie_id_map.items():
            if col_idx < self.user_item_matrix.shape[1]:
                score = np.dot(similar_users, self.user_item_matrix.iloc[:, col_idx].values)
                scores[movie_id] = max(0, score / (np.sum(similar_users) + 1e-6))
        
        return scores
    
    def _content_based_score(self, user_id, movie_id_map):
        """Calculate content-based scores"""
        # Get user's liked movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].values
        
        if len(liked_movies) == 0:
            return {}
        
        # Get genres of liked movies
        liked_genres = set()
        for movie_id in liked_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                genres = movie_info.iloc[0]['genres'].split('|')
                liked_genres.update(genres)
        
        scores = {}
        for movie_id in movie_id_map.keys():
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                movie_genres = set(movie_info.iloc[0]['genres'].split('|'))
                if len(liked_genres) > 0:
                    similarity = len(movie_genres & liked_genres) / len(liked_genres | movie_genres)
                    scores[movie_id] = similarity
        
        return scores
    
    def _matrix_factorization_score(self, user_id, movie_id_map):
        """Calculate matrix factorization scores"""
        user_idx = user_id - 1
        if user_idx < 0 or user_idx >= len(self.mf_model.user_factors):
            return {}
        
        scores = {}
        user_vector = self.mf_model.user_factors[user_idx]
        
        for movie_id, col_idx in movie_id_map.items():
            if col_idx < self.mf_model.item_factors.shape[0]:
                score = np.dot(user_vector, self.mf_model.item_factors[col_idx])
                scores[movie_id] = max(0, (score + 1) / 2)  # Normalize to [0, 1]
        
        return scores
