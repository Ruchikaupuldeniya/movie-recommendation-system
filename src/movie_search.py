"""
Movie Search and Information Module
Search for movies by name and show detailed information
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class MovieSearchEngine:
    """
    Search for movies and display detailed information
    """
    
    def __init__(self, ratings_df, movies_df):
        """
        Initialize search engine
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings data with columns: userId, movieId, rating
        movies_df : pd.DataFrame
            Movies data with columns: movieId, title, genres
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.movie_cache = {}
        
    def search_movie(self, query, exact_match=False, top_results=5):
        """
        Search for movies by name
        
        Parameters:
        -----------
        query : str
            Movie name to search
        exact_match : bool
            If True, search for exact match only
        top_results : int
            Number of results to return
            
        Returns:
        --------
        list : List of matching movies with similarity score
        """
        query = query.lower().strip()
        
        if not query:
            return []
        
        results = []
        
        for _, movie in self.movies_df.iterrows():
            title = movie['title'].lower()
            
            if exact_match:
                if query == title:
                    similarity = 1.0
                else:
                    continue
            else:
                # Fuzzy matching
                similarity = SequenceMatcher(None, query, title).ratio()
            
            if similarity > 0.6:  # At least 60% match
                results.append({
                    'movie_id': int(movie['movieId']),
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'similarity': similarity
                })
        
        # Sort by similarity and return top results
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return results[:top_results]
    
    def get_movie_stats(self, movie_id):
        """
        Get detailed statistics about a movie
        
        Parameters:
        -----------
        movie_id : int
            Movie ID
            
        Returns:
        --------
        dict : Movie statistics
        """
        # Get movie info
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        
        if len(movie) == 0:
            return None
        
        movie_info = movie.iloc[0]
        
        # Get ratings for this movie
        movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']
        
        if len(movie_ratings) == 0:
            return {
                'movie_id': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'total_ratings': 0,
                'average_rating': 0,
                'min_rating': 0,
                'max_rating': 0,
                'std_rating': 0,
                'rating_distribution': {},
                'status': 'No ratings found'
            }
        
        # Calculate statistics
        stats = {
            'movie_id': movie_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'total_ratings': len(movie_ratings),
            'average_rating': float(movie_ratings.mean()),
            'min_rating': float(movie_ratings.min()),
            'max_rating': float(movie_ratings.max()),
            'std_rating': float(movie_ratings.std()),
            'status': 'Found'
        }
        
        # Rating distribution
        rating_dist = {}
        for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
            count = len(movie_ratings[movie_ratings == rating])
            rating_dist[f'{rating}⭐'] = count
        
        stats['rating_distribution'] = rating_dist
        
        return stats
    
    def display_movie_info(self, movie_id):
        """
        Display formatted movie information
        
        Parameters:
        -----------
        movie_id : int
            Movie ID
        """
        stats = self.get_movie_stats(movie_id)
        
        if stats is None:
            print(f"\n❌ Sorry, I don't have data about this film!")
            print(f"   Movie ID {movie_id} not found in database")
            return
        
        if stats['status'] != 'Found' or stats['total_ratings'] == 0:
            print(f"\n❌ Sorry, I don't have data about this film!")
            print(f"   \"{stats['title']}\" has no ratings in our database")
            return
        
        # Display formatted information
        print("\n" + "="*70)
        print("🎬 MOVIE INFORMATION")
        print("="*70)
        
        print(f"\n📽️  Title: {stats['title']}")
        print(f"📚 Genres: {stats['genres']}")
        
        print(f"\n⭐ RATING INFORMATION:")
        print(f"   Average Rating: {stats['average_rating']:.2f} / 5.0")
        print(f"   Rating Range: {stats['min_rating']:.1f} - {stats['max_rating']:.1f}")
        print(f"   Standard Deviation: {stats['std_rating']:.2f}")
        print(f"   Total Ratings: {stats['total_ratings']}")
        
        print(f"\n📊 RATING DISTRIBUTION:")
        total = stats['total_ratings']
        for rating, count in sorted(stats['rating_distribution'].items(), reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"   {rating}: {bar} {percentage:.1f}% ({count} ratings)")
        
        print("\n" + "="*70)
    
    def search_and_display(self, query):
        """
        Search for movie and display results
        
        Parameters:
        -----------
        query : str
            Movie name to search
        """
        print(f"\n🔍 Searching for movies like '{query}'...\n")
        
        results = self.search_movie(query, exact_match=False, top_results=5)
        
        if len(results) == 0:
            print(f"❌ Sorry, I don't have data about '{query}'!")
            print(f"   No movies found matching your search.")
            print(f"\n💡 Try searching for:")
            print(f"   - Forrest Gump")
            print(f"   - Pulp Fiction")
            print(f"   - The Shawshank Redemption")
            print(f"   - Inception")
            return None
        
        print(f"✅ Found {len(results)} matching movie(s):\n")
        
        for idx, result in enumerate(results, 1):
            print(f"{idx}. {result['title']}")
            print(f"   Genres: {result['genres']}")
            print(f"   Match Score: {result['similarity']*100:.1f}%")
            print()
        
        return results
    
    def compare_movies(self, movie_id_1, movie_id_2):
        """
        Compare two movies
        
        Parameters:
        -----------
        movie_id_1 : int
            First movie ID
        movie_id_2 : int
            Second movie ID
        """
        stats1 = self.get_movie_stats(movie_id_1)
        stats2 = self.get_movie_stats(movie_id_2)
        
        if stats1 is None or stats2 is None:
            print("❌ One or both movies not found!")
            return
        
        if stats1['status'] != 'Found' or stats2['status'] != 'Found':
            print("❌ One or both movies have no ratings!")
            return
        
        print("\n" + "="*70)
        print("🎬 MOVIE COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<25} {'Movie 1':<20} {'Movie 2':<20}")
        print("-" * 70)
        
        print(f"{'Title':<25} {stats1['title'][:19]:<20} {stats2['title'][:19]:<20}")
        print(f"{'Average Rating':<25} {stats1['average_rating']:.2f}⭐ {'':<14} {stats2['average_rating']:.2f}⭐")
        print(f"{'Total Ratings':<25} {stats1['total_ratings']:<20} {stats2['total_ratings']:<20}")
        print(f"{'Rating Range':<25} {stats1['min_rating']:.1f}-{stats1['max_rating']:.1f} {'':<12} {stats2['min_rating']:.1f}-{stats2['max_rating']:.1f}")
        
        print("\n" + "="*70)
    
    def get_popular_movies(self, min_ratings=50, top_n=10):
        """
        Get popular movies (most rated)
        
        Parameters:
        -----------
        min_ratings : int
            Minimum number of ratings
        top_n : int
            Number of top movies to return
            
        Returns:
        --------
        pd.DataFrame : Popular movies
        """
        movie_counts = self.ratings_df.groupby('movieId').size()
        popular_movies = movie_counts[movie_counts >= min_ratings].sort_values(ascending=False).head(top_n).index
        
        result = []
        for movie_id in popular_movies:
            stats = self.get_movie_stats(movie_id)
            if stats and stats['status'] == 'Found':
                result.append(stats)
        
        return pd.DataFrame(result)
    
    def get_top_rated_movies(self, min_ratings=20, top_n=10):
        """
        Get highest rated movies
        
        Parameters:
        -----------
        min_ratings : int
            Minimum number of ratings
        top_n : int
            Number of top movies to return
            
        Returns:
        --------
        pd.DataFrame : Top rated movies
        """
        # Get movies with minimum ratings
        movie_ratings = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        movie_ratings.columns = ['movieId', 'avg_rating', 'count']
        movie_ratings = movie_ratings[movie_ratings['count'] >= min_ratings]
        movie_ratings = movie_ratings.sort_values('avg_rating', ascending=False).head(top_n)
        
        result = []
        for _, row in movie_ratings.iterrows():
            stats = self.get_movie_stats(int(row['movieId']))
            if stats and stats['status'] == 'Found':
                result.append(stats)
        
        return pd.DataFrame(result)
    
    def search_by_genre(self, genre, top_n=10):
        """
        Search movies by genre
        
        Parameters:
        -----------
        genre : str
            Genre name
        top_n : int
            Number of results
            
        Returns:
        --------
        list : Movies in that genre
        """
        genre = genre.lower()
        result = []
        
        for _, movie in self.movies_df.iterrows():
            if genre in movie['genres'].lower():
                movie_id = int(movie['movieId'])
                stats = self.get_movie_stats(movie_id)
                
                if stats and stats['status'] == 'Found' and stats['total_ratings'] > 0:
                    result.append(stats)
        
        # Sort by rating
        result = sorted(result, key=lambda x: x['average_rating'], reverse=True)
        return result[:top_n]
