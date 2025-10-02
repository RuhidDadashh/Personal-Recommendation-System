import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict, Counter
import math
import random
from datetime import datetime, timedelta
from models import Movie, User, UserMovieRating, UserInteraction, Recommendation

class RecommendationEngine:
    """Advanced recommendation engine with multiple algorithms"""
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.content_similarity_matrix = None
        self.user_item_matrix = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.genre_weights = {}
        self.popularity_scores = {}
        
    def initialize(self):
        """Initialize the recommendation engine with data from database"""
        self._load_data()
        self._build_content_similarity()
        self._build_collaborative_data()
        self._calculate_popularity_scores()
        
    def _load_data(self):
        """Load movie and rating data from database"""
        # Load movies
        movies = Movie.query.all()
        movie_data = []
        for movie in movies:
            movie_data.append({
                'movie_id': movie.id,
                'title': movie.title,
                'genre': movie.genre,
                'rating': movie.rating,
                'year': movie.year,
                'director': movie.director,
                'description': movie.description or ''
            })
        self.movies_df = pd.DataFrame(movie_data)
        
        # Load user ratings
        ratings = UserMovieRating.query.all()
        rating_data = []
        for rating in ratings:
            rating_data.append({
                'user_id': rating.user_id,
                'movie_id': rating.movie_id,
                'rating': rating.rating
            })
        self.ratings_df = pd.DataFrame(rating_data)
        
    def _build_content_similarity(self):
        """Build content-based similarity matrix using TF-IDF"""
        if self.movies_df is None or len(self.movies_df) == 0:
            return
            
        # Combine genre, director, and description for content features
        self.movies_df['content_features'] = (
            self.movies_df['genre'].fillna('') + ' ' +
            self.movies_df['director'].fillna('') + ' ' +
            self.movies_df['description'].fillna('')
        )
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['content_features'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
    def _build_collaborative_data(self):
        """Build user-item matrix for collaborative filtering"""
        if self.ratings_df is None or len(self.ratings_df) == 0:
            return
            
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Apply SVD for dimensionality reduction
        if len(self.user_item_matrix) > 0:
            # Use appropriate number of components for MovieLens data
            n_components = min(100, len(self.user_item_matrix.columns) - 1, len(self.user_item_matrix.index) - 1)
            if n_components > 0:
                self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
                self.item_factors = self.svd_model.components_
                
                # Calculate item-item similarity matrix for faster recommendations
                self.item_similarity_matrix = cosine_similarity(self.item_factors.T)
        
        # Build user similarity matrix for collaborative filtering
        if len(self.user_item_matrix) > 0:
            # Normalize ratings (subtract user mean)
            user_means = self.user_item_matrix.mean(axis=1)
            self.normalized_ratings = self.user_item_matrix.sub(user_means, axis=0)
            
            # Calculate user-user similarity
            self.user_similarity_matrix = cosine_similarity(self.normalized_ratings.fillna(0))
            
    def _calculate_popularity_scores(self):
        """Calculate popularity scores based on ratings and interactions"""
        if self.ratings_df is None or len(self.ratings_df) == 0:
            return
            
        # Calculate popularity based on number of ratings and average rating
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'avg_rating']
        
        # Calculate popularity score using Bayesian average
        overall_mean = self.ratings_df['rating'].mean()
        min_ratings = 10  # Minimum ratings required for reliability
        
        for movie_id in movie_stats.index:
            count = movie_stats.loc[movie_id, 'rating_count']
            avg = movie_stats.loc[movie_id, 'avg_rating']
            
            # Bayesian average formula
            bayesian_avg = (min_ratings * overall_mean + count * avg) / (min_ratings + count)
            self.popularity_scores[movie_id] = bayesian_avg
            
    def get_recommendations(self, user_id, selected_movies, algorithm='hybrid', num_recommendations=6):
        """Get recommendations using specified algorithm"""
        self.initialize()  # Refresh data
        
        if algorithm == 'content_based':
            return self._content_based_recommendations(selected_movies, num_recommendations)
        elif algorithm == 'collaborative':
            return self._collaborative_recommendations(user_id, selected_movies, num_recommendations)
        elif algorithm == 'hybrid':
            return self._hybrid_recommendations(user_id, selected_movies, num_recommendations)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
    def _content_based_recommendations(self, selected_movies, num_recommendations):
        """Generate content-based recommendations"""
        if self.content_similarity_matrix is None:
            return self._fallback_recommendations(num_recommendations)
            
        recommendations = defaultdict(float)
        
        for movie_id in selected_movies:
            # Find movie index in dataframe
            movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index
            if len(movie_idx) == 0:
                continue
                
            movie_idx = movie_idx[0]
            
            # Get similarity scores for this movie
            sim_scores = self.content_similarity_matrix[movie_idx]
            
            # Add scores to recommendations (weighted by similarity)
            for idx, score in enumerate(sim_scores):
                other_movie_id = self.movies_df.iloc[idx]['movie_id']
                if other_movie_id not in selected_movies:
                    recommendations[other_movie_id] += score
                    
        # Sort and format recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return self._format_recommendations(sorted_recs[:num_recommendations])
        
    def _collaborative_recommendations(self, user_id, selected_movies, num_recommendations):
        """Generate collaborative filtering recommendations"""
        if self.user_item_matrix is None or len(self.user_item_matrix) == 0:
            return self._fallback_recommendations(num_recommendations)
            
        recommendations = defaultdict(float)
        
        # Find similar users based on selected movies
        similar_users = self._find_similar_users(user_id, selected_movies)
        
        # Get recommendations from similar users
        for similar_user_id, similarity in similar_users:
            user_ratings = UserMovieRating.query.filter_by(user_id=similar_user_id).all()
            
            for rating in user_ratings:
                if rating.movie_id not in selected_movies and rating.rating >= 4.0:
                    recommendations[rating.movie_id] += similarity * rating.rating
                    
        # If no collaborative data, use matrix factorization
        if not recommendations and self.svd_model is not None:
            recommendations = self._matrix_factorization_recommendations(user_id, selected_movies, num_recommendations)
            
        if not recommendations:
            return self._fallback_recommendations(num_recommendations)
            
        # Sort and format recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return self._format_recommendations(sorted_recs[:num_recommendations])
        
    def _hybrid_recommendations(self, user_id, selected_movies, num_recommendations):
        """Generate hybrid recommendations combining multiple approaches"""
        content_recs = self._content_based_recommendations(selected_movies, num_recommendations * 2)
        collab_recs = self._collaborative_recommendations(user_id, selected_movies, num_recommendations * 2)
        
        # Combine recommendations with weights
        content_weight = 0.6
        collab_weight = 0.4
        
        combined_scores = defaultdict(float)
        
        # Add content-based scores
        for rec in content_recs:
            combined_scores[rec['movie_id']] += content_weight * rec['score']
            
        # Add collaborative scores
        for rec in collab_recs:
            combined_scores[rec['movie_id']] += collab_weight * rec['score']
            
        # Add popularity boost
        for movie_id in combined_scores:
            if movie_id in self.popularity_scores:
                combined_scores[movie_id] += 0.1 * self.popularity_scores[movie_id]
                
        # Sort and format final recommendations
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return self._format_recommendations(sorted_recs[:num_recommendations])
        
    def _find_similar_users(self, user_id, selected_movies):
        """Find users similar to current user based on movie preferences"""
        similar_users = []
        
        # Get users who rated the selected movies
        user_ratings = defaultdict(dict)
        for movie_id in selected_movies:
            ratings = UserMovieRating.query.filter_by(movie_id=movie_id).all()
            for rating in ratings:
                if rating.user_id != user_id:
                    user_ratings[rating.user_id][movie_id] = rating.rating
                    
        # Calculate similarity using Jaccard similarity
        for other_user_id, ratings in user_ratings.items():
            if len(ratings) >= 2:  # User must have rated at least 2 of the selected movies
                # Simple similarity: percentage of movies in common with good ratings
                good_ratings = sum(1 for r in ratings.values() if r >= 4.0)
                similarity = good_ratings / len(selected_movies)
                if similarity > 0.3:  # Minimum similarity threshold
                    similar_users.append((other_user_id, similarity))
                    
        return sorted(similar_users, key=lambda x: x[1], reverse=True)[:10]
        
    def _matrix_factorization_recommendations(self, user_id, selected_movies, num_recommendations):
        """Generate recommendations using matrix factorization"""
        if self.svd_model is None or user_id not in self.user_item_matrix.index:
            return {}
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Calculate predicted ratings for all movies
        predicted_ratings = np.dot(user_vector, self.item_factors)
        
        recommendations = {}
        movie_ids = self.user_item_matrix.columns
        
        for i, movie_id in enumerate(movie_ids):
            if movie_id not in selected_movies:
                recommendations[movie_id] = predicted_ratings[i]
                
        return recommendations
        
    def _fallback_recommendations(self, num_recommendations):
        """Fallback recommendations based on popularity and rating"""
        if self.movies_df is None:
            return []
            
        # Get top rated movies
        top_movies = self.movies_df.nlargest(num_recommendations * 2, 'rating')
        
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append((movie['movie_id'], movie['rating']))
            
        return self._format_recommendations(recommendations[:num_recommendations])
        
    def _format_recommendations(self, recommendations):
        """Format recommendations for API response"""
        formatted_recs = []
        
        for movie_id, score in recommendations:
            movie = Movie.query.get(movie_id)
            if movie:
                formatted_recs.append({
                    'movie_id': movie.id,
                    'title': movie.title,
                    'genre': movie.genre,
                    'rating': f"{movie.rating}/10",
                    'year': movie.year,
                    'director': movie.director,
                    'description': movie.description,
                    'score': float(score),
                    'reason': self._generate_explanation(movie, score)
                })
                
        return formatted_recs
        
    def _generate_explanation(self, movie, score):
        """Generate explanation for why movie was recommended"""
        explanations = [
            f"Highly rated {movie.genre.split(',')[0].strip()} film",
            f"Great {movie.year}s movie you might enjoy",
            f"Similar style to your preferences",
            f"Popular choice among users with similar taste",
            f"Acclaimed film by {movie.director}"
        ]
        
        # Choose explanation based on score
        if score > 8:
            return explanations[0]
        elif score > 7:
            return explanations[1]
        else:
            return random.choice(explanations[2:])
            
    def calculate_diversity_score(self, recommendations):
        """Calculate diversity score of recommendations"""
        if not recommendations:
            return 0.0
            
        genres = []
        years = []
        
        for rec in recommendations:
            movie_genres = rec['genre'].split(', ')
            genres.extend(movie_genres)
            years.append(rec['year'])
            
        # Genre diversity
        unique_genres = len(set(genres))
        total_genre_mentions = len(genres)
        genre_diversity = unique_genres / total_genre_mentions if total_genre_mentions > 0 else 0
        
        # Year diversity
        year_range = max(years) - min(years) if len(years) > 1 else 0
        year_diversity = min(year_range / 50, 1.0)  # Normalize to 0-1
        
        # Combined diversity score
        diversity_score = (genre_diversity * 0.7) + (year_diversity * 0.3)
        
        return min(diversity_score, 1.0)
        
    def get_trending_movies(self, days=30, limit=10):
        """Get trending movies based on recent interactions"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent interactions
        popular_movies = UserInteraction.get_popular_movies(days=days, limit=limit)
        
        trending = []
        for movie_id, interaction_count in popular_movies:
            movie = Movie.query.get(movie_id)
            if movie:
                trending.append({
                    'movie_id': movie.id,
                    'title': movie.title,
                    'genre': movie.genre,
                    'rating': f"{movie.rating}/10",
                    'interaction_count': interaction_count
                })
                
        return trending
        
    def get_algorithm_performance(self):
        """Get performance metrics for different algorithms"""
        algorithms = ['content_based', 'collaborative', 'hybrid']
        performance = {}
        
        for algorithm in algorithms:
            performance[algorithm] = Recommendation.get_algorithm_performance(algorithm)
            
        return performance
        
    def update_user_preferences(self, user_id, movie_interactions):
        """Update user preferences based on interactions"""
        # Analyze user's genre preferences
        genre_preferences = defaultdict(float)
        
        for movie_id, interaction_type, rating in movie_interactions:
            movie = Movie.query.get(movie_id)
            if movie:
                genres = movie.get_genres_list()
                weight = 1.0
                
                # Weight based on interaction type
                if interaction_type == 'rating' and rating:
                    weight = rating / 5.0
                elif interaction_type == 'click':
                    weight = 0.5
                elif interaction_type == 'view':
                    weight = 0.3
                    
                for genre in genres:
                    genre_preferences[genre] += weight
                    
        # Update user profile
        from models import UserProfile
        profile = UserProfile.get_or_create(user_id)
        
        # Get top preferred genres
        top_genres = sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True)[:5]
        preferred_genres = [genre for genre, _ in top_genres]
        
        profile.update_preferences(preferred_genres=preferred_genres)
        
        return preferred_genres