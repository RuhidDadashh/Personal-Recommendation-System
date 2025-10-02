"""
MovieLens Dataset Integration
This script loads MovieLens data into the recommendation system database
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import zipfile
import requests
from io import BytesIO
from models import Movie, User, UserMovieRating, UserInteraction, db
from app import app
import re
from tqdm import tqdm

class MovieLensLoader:
    """Class to handle MovieLens dataset loading and processing"""
    
    def __init__(self, dataset_size='small'):
        """
        Initialize MovieLens loader
        
        Args:
            dataset_size (str): 'small' (100k), 'full' (25M), or 'latest-small' (100k latest)
        """
        self.dataset_size = dataset_size
        self.base_url = "https://files.grouplens.org/datasets/movielens/"
        self.dataset_urls = {
            'small': 'ml-latest-small.zip',
            'full': 'ml-25m.zip',
            'latest-small': 'ml-latest-small.zip'
        }
        self.data_dir = 'movielens_data'
        
    def download_dataset(self):
        """Download MovieLens dataset if not already present"""
        print(f"📥 Downloading MovieLens {self.dataset_size} dataset...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        dataset_file = self.dataset_urls[self.dataset_size]
        url = self.base_url + dataset_file
        local_path = os.path.join(self.data_dir, dataset_file)
        
        # Check if already downloaded
        if os.path.exists(local_path):
            print(f"✅ Dataset already exists at {local_path}")
            return local_path
        
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            raise
    
    def extract_dataset(self, zip_path):
        """Extract the downloaded dataset"""
        print("📂 Extracting dataset...")
        
        extract_path = os.path.join(self.data_dir, 'extracted')
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Find the extracted folder
            extracted_folders = [d for d in os.listdir(extract_path) 
                               if os.path.isdir(os.path.join(extract_path, d))]
            
            if extracted_folders:
                dataset_folder = os.path.join(extract_path, extracted_folders[0])
                print(f"✅ Extracted to {dataset_folder}")
                return dataset_folder
            else:
                raise Exception("No extracted folder found")
                
        except Exception as e:
            print(f"❌ Error extracting dataset: {e}")
            raise
    
    def load_movies_data(self, dataset_folder):
        """Load movies data from MovieLens CSV"""
        print("🎬 Loading movies data...")
        
        movies_file = os.path.join(dataset_folder, 'movies.csv')
        
        try:
            movies_df = pd.read_csv(movies_file)
            print(f"📊 Loaded {len(movies_df)} movies")
            
            # Clean and process movie data
            movies_df = self.process_movies_data(movies_df)
            
            return movies_df
            
        except Exception as e:
            print(f"❌ Error loading movies data: {e}")
            raise
    
    def process_movies_data(self, movies_df):
        """Process and clean movies data"""
        print("🧹 Processing movies data...")
        
        # Extract year from title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
        movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
        
        # Clean title (remove year)
        movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Process genres
        movies_df['genres'] = movies_df['genres'].str.replace('|', ', ')
        movies_df['genres'] = movies_df['genres'].replace('(no genres listed)', 'Unknown')
        
        # Add default values for missing columns
        movies_df['rating'] = 7.0  # Default rating
        movies_df['director'] = 'Unknown'
        movies_df['description'] = ''
        
        # Filter out movies without years (likely data quality issues)
        movies_df = movies_df.dropna(subset=['year'])
        movies_df = movies_df[movies_df['year'] >= 1900]
        movies_df = movies_df[movies_df['year'] <= datetime.now().year]
        
        print(f"✅ Processed {len(movies_df)} movies")
        return movies_df
    
    def load_ratings_data(self, dataset_folder):
        """Load ratings data from MovieLens CSV"""
        print("⭐ Loading ratings data...")
        
        ratings_file = os.path.join(dataset_folder, 'ratings.csv')
        
        try:
            ratings_df = pd.read_csv(ratings_file)
            print(f"📊 Loaded {len(ratings_df)} ratings")
            
            # Convert timestamp to datetime
            if 'timestamp' in ratings_df.columns:
                ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
            
            return ratings_df
            
        except Exception as e:
            print(f"❌ Error loading ratings data: {e}")
            raise
    
    def load_tags_data(self, dataset_folder):
        """Load tags data from MovieLens CSV (if available)"""
        tags_file = os.path.join(dataset_folder, 'tags.csv')
        
        if not os.path.exists(tags_file):
            print("ℹ️ No tags data available")
            return None
        
        try:
            print("🏷️ Loading tags data...")
            tags_df = pd.read_csv(tags_file)
            
            if 'timestamp' in tags_df.columns:
                tags_df['datetime'] = pd.to_datetime(tags_df['timestamp'], unit='s')
            
            print(f"📊 Loaded {len(tags_df)} tags")
            return tags_df
            
        except Exception as e:
            print(f"❌ Error loading tags data: {e}")
            return None
    
    def create_users_from_ratings(self, ratings_df, sample_size=1000):
        """Create user accounts based on MovieLens user IDs"""
        print(f"👥 Creating users from ratings data (sample: {sample_size})...")
        
        # Get unique user IDs
        unique_users = ratings_df['userId'].unique()
        
        # Sample users if dataset is too large
        if len(unique_users) > sample_size:
            unique_users = np.random.choice(unique_users, sample_size, replace=False)
            print(f"📊 Sampling {sample_size} users from {len(ratings_df['userId'].unique())} total users")
        
        users_created = 0
        
        for user_id in tqdm(unique_users, desc="Creating users"):
            # Check if user already exists
            existing_user = User.query.filter_by(email=f'movielens_user_{user_id}@example.com').first()
            
            if not existing_user:
                user = User(
                    name=f'MovieLens User {user_id}',
                    email=f'movielens_user_{user_id}@example.com',
                    password='dummy_password_hash',  # These are training users
                    created_at=datetime.utcnow()
                )
                user.movielens_id = user_id  # Store original MovieLens ID
                db.session.add(user)
                users_created += 1
        
        try:
            db.session.commit()
            print(f"✅ Created {users_created} users")
            return users_created
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error creating users: {e}")
            raise
    
    def import_movies_to_db(self, movies_df, batch_size=1000):
        """Import movies data to database"""
        print("📥 Importing movies to database...")
        
        movies_created = 0
        movies_updated = 0
        
        for i in tqdm(range(0, len(movies_df), batch_size), desc="Importing movies"):
            batch = movies_df.iloc[i:i+batch_size]
            
            for _, movie_row in batch.iterrows():
                # Check if movie already exists
                existing_movie = Movie.query.filter_by(title=movie_row['clean_title']).first()
                
                if existing_movie:
                    # Update existing movie
                    existing_movie.genre = movie_row['genres']
                    existing_movie.year = int(movie_row['year'])
                    existing_movie.movielens_id = movie_row['movieId']
                    movies_updated += 1
                else:
                    # Create new movie
                    movie = Movie(
                        title=movie_row['clean_title'],
                        genre=movie_row['genres'],
                        rating=movie_row['rating'],
                        year=int(movie_row['year']),
                        director=movie_row.get('director', 'Unknown'),
                        description=movie_row.get('description', ''),
                        created_at=datetime.utcnow()
                    )
                    movie.movielens_id = movie_row['movieId']  # Store original MovieLens ID
                    db.session.add(movie)
                    movies_created += 1
            
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"❌ Error in batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"✅ Movies imported: {movies_created} created, {movies_updated} updated")
        return movies_created, movies_updated
    
    def import_ratings_to_db(self, ratings_df, movies_df, sample_ratio=0.1, batch_size=5000):
        """Import ratings data to database"""
        print(f"⭐ Importing ratings to database (sample ratio: {sample_ratio})...")
        
        # Create mapping from MovieLens IDs to our database IDs
        movielens_to_db_movies = {}
        for movie in Movie.query.all():
            if hasattr(movie, 'movielens_id') and movie.movielens_id:
                movielens_to_db_movies[movie.movielens_id] = movie.id
        
        movielens_to_db_users = {}
        for user in User.query.filter(User.email.like('movielens_user_%')).all():
            if hasattr(user, 'movielens_id') and user.movielens_id:
                movielens_to_db_users[user.movielens_id] = user.id
        
        # Sample ratings if dataset is too large
        if sample_ratio < 1.0:
            ratings_df = ratings_df.sample(frac=sample_ratio).reset_index(drop=True)
            print(f"📊 Sampling {len(ratings_df)} ratings")
        
        # Filter ratings for movies and users we have in our database
        ratings_df = ratings_df[
            (ratings_df['movieId'].isin(movielens_to_db_movies.keys())) &
            (ratings_df['userId'].isin(movielens_to_db_users.keys()))
        ]
        
        print(f"📊 Filtered to {len(ratings_df)} ratings with matching movies and users")
        
        ratings_created = 0
        
        for i in tqdm(range(0, len(ratings_df), batch_size), desc="Importing ratings"):
            batch = ratings_df.iloc[i:i+batch_size]
            
            for _, rating_row in batch.iterrows():
                # Map MovieLens IDs to our database IDs
                db_movie_id = movielens_to_db_movies.get(rating_row['movieId'])
                db_user_id = movielens_to_db_users.get(rating_row['userId'])
                
                if db_movie_id and db_user_id:
                    # Check if rating already exists
                    existing_rating = UserMovieRating.query.filter_by(
                        user_id=db_user_id,
                        movie_id=db_movie_id
                    ).first()
                    
                    if not existing_rating:
                        rating = UserMovieRating(
                            user_id=db_user_id,
                            movie_id=db_movie_id,
                            rating=rating_row['rating'],
                            created_at=rating_row.get('datetime', datetime.utcnow())
                        )
                        db.session.add(rating)
                        ratings_created += 1
            
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"❌ Error in batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"✅ Ratings imported: {ratings_created}")
        return ratings_created
    
    def create_interaction_data(self, ratings_df):
        """Create interaction data based on ratings"""
        print("📊 Creating interaction data...")
        
        interactions_created = 0
        
        # Create interactions based on ratings (high ratings = positive interactions)
        high_ratings = ratings_df[ratings_df['rating'] >= 4.0]
        
        for _, rating_row in tqdm(high_ratings.iterrows(), desc="Creating interactions", total=len(high_ratings)):
            # Get our database IDs
            user = User.query.filter(
                User.email == f'movielens_user_{rating_row["userId"]}@example.com'
            ).first()
            
            movie = Movie.query.filter_by(movielens_id=rating_row['movieId']).first()
            
            if user and movie:
                # Create view interaction
                interaction = UserInteraction(
                    user_id=user.id,
                    movie_id=movie.id,
                    interaction_type='view',
                    timestamp=rating_row.get('datetime', datetime.utcnow()),
                    metadata={'rating': rating_row['rating']}
                )
                db.session.add(interaction)
                interactions_created += 1
        
        try:
            db.session.commit()
            print(f"✅ Interactions created: {interactions_created}")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error creating interactions: {e}")
        
        return interactions_created
    
    def load_and_import(self, sample_users=1000, sample_ratings=0.1):
        """Complete pipeline to load and import MovieLens data"""
        print("🚀 Starting MovieLens data import pipeline...")
        
        try:
            # Download and extract dataset
            zip_path = self.download_dataset()
            dataset_folder = self.extract_dataset(zip_path)
            
            # Load data files
            movies_df = self.load_movies_data(dataset_folder)
            ratings_df = self.load_ratings_data(dataset_folder)
            tags_df = self.load_tags_data(dataset_folder)
            
            # Import to database
            with app.app_context():
                print("📊 Database import starting...")
                
                # Import movies
                movies_created, movies_updated = self.import_movies_to_db(movies_df)
                
                # Create users
                users_created = self.create_users_from_ratings(ratings_df, sample_users)
                
                # Import ratings
                ratings_created = self.import_ratings_to_db(ratings_df, movies_df, sample_ratings)
                
                # Create interactions
                interactions_created = self.create_interaction_data(ratings_df)
                
                print("\n🎉 MovieLens import completed!")
                print(f"📊 Summary:")
                print(f"   Movies: {movies_created} created, {movies_updated} updated")
                print(f"   Users: {users_created} created")
                print(f"   Ratings: {ratings_created} imported")
                print(f"   Interactions: {interactions_created} created")
                
                return {
                    'movies_created': movies_created,
                    'movies_updated': movies_updated,
                    'users_created': users_created,
                    'ratings_created': ratings_created,
                    'interactions_created': interactions_created
                }
                
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main function to run MovieLens import"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import MovieLens dataset')
    parser.add_argument('--dataset', choices=['small', 'full', 'latest-small'], 
                       default='small', help='Dataset size to download')
    parser.add_argument('--sample-users', type=int, default=1000,
                       help='Number of users to sample (default: 1000)')
    parser.add_argument('--sample-ratings', type=float, default=0.1,
                       help='Fraction of ratings to import (default: 0.1)')
    
    args = parser.parse_args()
    
    loader = MovieLensLoader(args.dataset)
    result = loader.load_and_import(args.sample_users, args.sample_ratings)
    
    print(f"\n✅ Import completed successfully!")

if __name__ == '__main__':
    main()