#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Movie Recommendation System - Interactive Script
Complete standalone application with movie search and recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import zipfile
import os
import shutil
from sklearn.metrics import mean_squared_error

# Import custom modules
from src.recommender import MovieRecommender
from src.advanced_recommender import HybridRecommender, MatrixFactorization
from src.movie_search import MovieSearchEngine
from src.evaluation import RecommendationEvaluator

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def download_dataset():
    """Download MovieLens dataset"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists('data/ratings.csv'):
        print("📥 Downloading movie dataset (1-2 minutes)...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        urllib.request.urlretrieve(url, 'data/movielens.zip')
        
        print("📦 Extracting files...")
        with zipfile.ZipFile('data/movielens.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
        
        shutil.move('data/ml-latest-small/ratings.csv', 'data/ratings.csv')
        shutil.move('data/ml-latest-small/movies.csv', 'data/movies.csv')
        shutil.rmtree('data/ml-latest-small')
        os.remove('data/movielens.zip')
        print("✅ Dataset downloaded!")
    else:
        print("✅ Dataset already exists!")

def load_data():
    """Load datasets"""
    print("\n📂 Loading data...")
    ratings_df = pd.read_csv('data/ratings.csv')
    movies_df = pd.read_csv('data/movies.csv')
    print(f"✅ Loaded {len(ratings_df)} ratings")
    return ratings_df, movies_df

def explore_data(ratings_df, movies_df):
    """Explore and visualize data"""
    print("\n" + "="*70)
    print("📊 DATA EXPLORATION")
    print("="*70)
    
    print(f"\n👥 Total Users: {ratings_df['userId'].nunique():,}")
    print(f"🎬 Total Movies: {ratings_df['movieId'].nunique():,}")
    print(f"⭐ Total Ratings: {len(ratings_df):,}")
    print(f"\n⭐ RATING STATISTICS:")
    print(f"   Average: {ratings_df['rating'].mean():.2f}")
    print(f"   Min: {ratings_df['rating'].min():.1f}, Max: {ratings_df['rating'].max():.1f}")
    
    print(f"\n🏆 TOP 5 MOST RATED MOVIES:")
    top_rated = ratings_df['movieId'].value_counts().head(5)
    for idx, (movie_id, count) in enumerate(top_rated.items(), 1):
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
        print(f"   {idx}. {movie_title} - {count} ratings, avg: {avg_rating:.2f}⭐")

def build_models(ratings_df, movies_df):
    """Build recommendation models"""
    print("\n" + "="*70)
    print("🤖 BUILDING RECOMMENDATION MODELS")
    print("="*70)
    
    # Basic recommender
    print("\n✅ Building Collaborative Filtering...")
    recommender = MovieRecommender(ratings_df, movies_df)
    recommender.create_user_item_matrix()
    
    # Matrix Factorization
    print("✅ Building Matrix Factorization (SVD)...")
    user_item_matrix = recommender.user_item_matrix
    mf_svd = MatrixFactorization(method='svd', n_factors=50)
    mf_svd.fit(user_item_matrix)
    
    # Hybrid
    print("✅ Building Hybrid System...")
    hybrid = HybridRecommender(ratings_df, movies_df)
    hybrid.fit()
    
    print("\n✅ All models ready!")
    return recommender, mf_svd, hybrid

def interactive_menu(ratings_df, movies_df, recommender, mf_svd, hybrid):
    """Interactive menu for user"""
    search_engine = MovieSearchEngine(ratings_df, movies_df)
    
    while True:
        print("\n" + "="*70)
        print("🎬 MOVIE RECOMMENDATION SYSTEM - INTERACTIVE MENU")
        print("="*70)
        print("\n1. 🔍 Search for Movie")
        print("2. 👤 Get User Recommendations")
        print("3. 🏆 View Top Rated Movies")
        print("4. 📚 Search by Genre")
        print("5. 🎯 Get Hybrid Recommendations")
        print("6. 📊 View System Statistics")
        print("7. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            query = input("\n🎬 Enter movie name: ").strip()
            if query:
                results = search_engine.search_and_display(query)
                if results:
                    view = input("\nDo you want to see details? (y/n): ").lower()
                    if view == 'y':
                        movie_id = results[0]['movie_id']
                        search_engine.display_movie_info(movie_id)
        
        elif choice == '2':
            try:
                user_id = int(input("\nEnter user ID: ").strip())
                algo = input("Select algorithm (collaborative/hybrid) [hybrid]: ").strip() or "hybrid"
                
                if algo == "collaborative":
                    recs = recommender.collaborative_filtering(user_id, n_recommendations=5)
                else:
                    movie_id_map = {mid: i for i, mid in enumerate(movies_df['movieId'].values)}
                    recs = hybrid.get_hybrid_recommendations(user_id, movie_id_map, 5)
                
                print(f"\n👤 Recommendations for User {user_id}:")
                for idx, (_, rec) in enumerate(recs.iterrows(), 1):
                    print(f"   {idx}. {rec['title']}")
                    print(f"      Genres: {rec['genres']}")
                    print(f"      Score: {rec[recs.columns[-1]]:.2f}")
            except ValueError:
                print("❌ Error: Invalid user ID")
        
        elif choice == '3':
            top_movies = search_engine.get_top_rated_movies(min_ratings=20, top_n=10)
            print("\n🏆 TOP RATED MOVIES:")
            for idx, (_, movie) in enumerate(top_movies.iterrows(), 1):
                print(f"   {idx}. {movie['title']} - {movie['average_rating']:.2f}⭐ ({movie['total_ratings']} ratings)")
        
        elif choice == '4':
            genre = input("\n📚 Enter genre (Action/Drama/Sci-Fi/etc): ").strip()
            if genre:
                movies = search_engine.search_by_genre(genre, top_n=10)
                if movies:
                    print(f"\n📚 TOP MOVIES IN '{genre.upper()}' GENRE:")
                    for idx, movie in enumerate(movies, 1):
                        print(f"   {idx}. {movie['title']} - {movie['average_rating']:.2f}⭐")
                else:
                    print(f"❌ No movies found in '{genre}' genre!")
        
        elif choice == '5':
            try:
                user_id = int(input("\nEnter user ID: ").strip())
                movie_id_map = {mid: i for i, mid in enumerate(movies_df['movieId'].values)}
                recs = hybrid.get_hybrid_recommendations(user_id, movie_id_map, 10)
                
                print(f"\n🎯 HYBRID RECOMMENDATIONS (User {user_id}):")
                for idx, (_, rec) in enumerate(recs.iterrows(), 1):
                    print(f"   {idx}. {rec['title']} - Score: {rec['hybrid_score']:.2f}")
            except ValueError:
                print("❌ Error: Invalid user ID")
        
        elif choice == '6':
            print("\n📊 SYSTEM STATISTICS:")
            print(f"   • Total Users: {ratings_df['userId'].nunique():,}")
            print(f"   • Total Movies: {ratings_df['movieId'].nunique():,}")
            print(f"   • Total Ratings: {len(ratings_df):,}")
            print(f"   • Average Rating: {ratings_df['rating'].mean():.2f}⭐")
            print(f"   • Min Rating: {ratings_df['rating'].min():.1f}⭐")
            print(f"   • Max Rating: {ratings_df['rating'].max():.1f}⭐")
        
        elif choice == '7':
            print("\n🚪 Exiting...")
            break
        
        else:
            print("❌ Error: Please select 1-7")

def generate_visualizations(ratings_df, movies_df):
    """Generate visualization plots"""
    print("\n📈 Generating visualizations...")
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Rating distribution
    plt.subplot(2, 2, 1)
    plt.hist(ratings_df['rating'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of Movie Ratings')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: User activity
    plt.subplot(2, 2, 2)
    user_ratings = ratings_df.groupby('userId')['rating'].count()
    plt.hist(user_ratings, bins=50, color='coral', edgecolor='black')
    plt.xlabel('Ratings per User')
    plt.ylabel('Number of Users')
    plt.title('User Activity Distribution')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Top rated
    plt.subplot(2, 2, 3)
    top_movies = ratings_df['movieId'].value_counts().head(5)
    movie_titles = [movies_df[movies_df['movieId'] == mid]['title'].values[0][:20] for mid in top_movies.index]
    plt.barh(movie_titles, top_movies.values, color='lightgreen')
    plt.xlabel('Number of Ratings')
    plt.title('Top 5 Most Rated Movies')
    plt.grid(axis='x', alpha=0.3)
    
    # Plot 4: Statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
STATISTICS:
• Users: {ratings_df['userId'].nunique():,}
• Movies: {ratings_df['movieId'].nunique():,}
• Ratings: {len(ratings_df):,}

RATING METRICS:
• Average: {ratings_df['rating'].mean():.2f}⭐
• Std Dev: {ratings_df['rating'].std():.2f}
• Range: {ratings_df['rating'].min():.1f}-{ratings_df['rating'].max():.1f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/analysis.png")
    plt.close()

def main():
    """Main execution"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  🎬 ADVANCED MOVIE RECOMMENDATION SYSTEM 🎬              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Download and load data
    download_dataset()
    ratings_df, movies_df = load_data()
    
    # Explore data
    explore_data(ratings_df, movies_df)
    
    # Build models
    recommender, mf_svd, hybrid = build_models(ratings_df, movies_df)
    
    # Generate visualizations
    generate_visualizations(ratings_df, movies_df)
    
    # Interactive menu
    interactive_menu(ratings_df, movies_df, recommender, mf_svd, hybrid)
    
    print("\n✅ Script completed!")
    print("📊 Check results/ folder for visualizations\n")

if __name__ == "__main__":
    main()
