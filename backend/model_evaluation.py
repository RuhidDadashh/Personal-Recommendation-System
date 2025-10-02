"""
Model Evaluation for Movie Recommendation System
Evaluates different recommendation algorithms using various metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from datetime import datetime, timedelta

from models import User, Movie, UserMovieRating, Recommendation, db
from recommendation_engine import RecommendationEngine

class ModelEvaluator:
    """Class to evaluate recommendation system performance"""
    
    def __init__(self):
        self.engine = RecommendationEngine()
        self.test_results = {}
        
    def prepare_evaluation_data(self, test_size=0.2, min_ratings_per_user=5):
        """Prepare train/test split for evaluation"""
        print("📊 Preparing evaluation data...")
        
        # Get all ratings
        ratings_query = db.session.query(
            UserMovieRating.user_id,
            UserMovieRating.movie_id,
            UserMovieRating.rating
        ).all()
        
        ratings_df = pd.DataFrame(ratings_query, columns=['user_id', 'movie_id', 'rating'])
        
        # Filter users with minimum ratings
        user_counts = ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
        
        print(f"📈 Evaluation data: {len(ratings_df)} ratings from {len(valid_users)} users")
        
        # Create train/test split per user to ensure each user has data in both sets
        train_data = []
        test_data = []
        
        for user_id in valid_users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            if len(user_ratings) >= min_ratings_per_user:
                # Ensure at least one rating in test set
                n_test = max(1, int(len(user_ratings) * test_size))
                test_indices = np.random.choice(len(user_ratings), n_test, replace=False)
                
                user_test = user_ratings.iloc[test_indices]
                user_train = user_ratings.drop(user_ratings.index[test_indices])
                
                train_data.append(user_train)
                test_data.append(user_test)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"📊 Train set: {len(train_df)} ratings, Test set: {len(test_df)} ratings")
        
        return train_df, test_df
    
    def evaluate_rating_prediction(self, algorithm='hybrid', n_recommendations=10):
        """Evaluate rating prediction accuracy"""
        print(f"🎯 Evaluating {algorithm} algorithm for rating prediction...")
        
        train_df, test_df = self.prepare_evaluation_data()
        
        # Temporarily replace training data
        original_ratings = self.engine.ratings_df
        self.engine.ratings_df = train_df
        self.engine.initialize()
        
        predictions = []
        actuals = []
        
        # Get predictions for test set
        test_users = test_df['user_id'].unique()
        
        for user_id in test_users[:50]:  # Limit for computational efficiency
            user_test = test_df[test_df['user_id'] == user_id]
            
            # Get user's training ratings for context
            user_train = train_df[train_df['user_id'] == user_id]
            if len(user_train) < 3:
                continue
            
            # Sample some movies the user liked for recommendation context
            liked_movies = user_train[user_train['rating'] >= 4.0]['movie_id'].tolist()
            if len(liked_movies) < 2:
                liked_movies = user_train['movie_id'].tolist()
            
            context_movies = random.sample(liked_movies, min(4, len(liked_movies)))
            
            try:
                # Get recommendations
                recs = self.engine.get_recommendations(user_id, context_movies, algorithm)
                
                # Create prediction lookup
                rec_scores = {rec['movie_id']: rec['score'] for rec in recs}
                
                # Compare with actual ratings
                for _, row in user_test.iterrows():
                    if row['movie_id'] in rec_scores:
                        # Normalize recommendation score to 1-5 rating scale
                        predicted_rating = min(5, max(1, rec_scores[row['movie_id']] * 5))
                        predictions.append(predicted_rating)
                        actuals.append(row['rating'])
                        
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
        
        # Restore original data
        self.engine.ratings_df = original_ratings
        
        if len(predictions) == 0:
            print("❌ No predictions generated")
            return {}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        results = {
            'algorithm': algorithm,
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions),
            'coverage': len(predictions) / len(test_df) if len(test_df) > 0 else 0
        }
        
        print(f"📊 {algorithm} Results:")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   Coverage: {results['coverage']:.1%}")
        
        return results
    
    def evaluate_ranking_metrics(self, algorithm='hybrid', k=10):
        """Evaluate ranking metrics (Precision@K, Recall@K, NDCG@K)"""
        print(f"📈 Evaluating {algorithm} ranking metrics...")
        
        train_df, test_df = self.prepare_evaluation_data()
        
        # Temporarily replace training data
        original_ratings = self.engine.ratings_df
        self.engine.ratings_df = train_df
        self.engine.initialize()
        
        precisions = []
        recalls = []
        ndcgs = []
        
        test_users = test_df['user_id'].unique()
        
        for user_id in test_users[:30]:  # Limit for efficiency
            user_test = test_df[test_df['user_id'] == user_id]
            user_train = train_df[train_df['user_id'] == user_id]
            
            if len(user_train) < 3:
                continue
            
            # Get relevant items (rated >= 4 in test set)
            relevant_items = set(user_test[user_test['rating'] >= 4.0]['movie_id'].tolist())
            
            if len(relevant_items) == 0:
                continue
            
            # Get user's liked movies for context
            liked_movies = user_train[user_train['rating'] >= 4.0]['movie_id'].tolist()
            if len(liked_movies) < 2:
                liked_movies = user_train['movie_id'].tolist()
            
            context_movies = random.sample(liked_movies, min(4, len(liked_movies)))
            
            try:
                # Get recommendations
                recs = self.engine.get_recommendations(user_id, context_movies, algorithm)
                recommended_items = [rec['movie_id'] for rec in recs[:k]]
                
                # Calculate metrics
                recommended_relevant = set(recommended_items) & relevant_items
                
                # Precision@K
                precision = len(recommended_relevant) / k if k > 0 else 0
                precisions.append(precision)
                
                # Recall@K
                recall = len(recommended_relevant) / len(relevant_items) if len(relevant_items) > 0 else 0
                recalls.append(recall)
                
                # NDCG@K (simplified)
                dcg = 0
                idcg = sum([1/np.log2(i+2) for i in range(min(k, len(relevant_items)))])
                
                for i, item in enumerate(recommended_items):
                    if item in relevant_items:
                        dcg += 1 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
                
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
        
        # Restore original data
        self.engine.ratings_df = original_ratings
        
        results = {
            'algorithm': algorithm,
            'precision_at_k': np.mean(precisions) if precisions else 0,
            'recall_at_k': np.mean(recalls) if recalls else 0,
            'ndcg_at_k': np.mean(ndcgs) if ndcgs else 0,
            'k': k,
            'n_users_evaluated': len(precisions)
        }
        
        print(f"📊 {algorithm} Ranking Results @{k}:")
        print(f"   Precision: {results['precision_at_k']:.3f}")
        print(f"   Recall: {results['recall_at_k']:.3f}")
        print(f"   NDCG: {results['ndcg_at_k']:.3f}")
        
        return results
    
    def evaluate_diversity_and_novelty(self, algorithm='hybrid'):
        """Evaluate recommendation diversity and novelty"""
        print(f"🌈 Evaluating {algorithm} diversity and novelty...")
        
        self.engine.initialize()
        
        # Get sample of users for evaluation
        users = User.query.filter_by(is_training_user=False).limit(20).all()
        
        all_recommendations = []
        genre_diversity_scores = []
        novelty_scores = []
        
        for user in users:
            # Get some of user's ratings for context
            user_ratings = UserMovieRating.query.filter_by(user_id=user.id).limit(10).all()
            if len(user_ratings) < 2:
                continue
                
            context_movies = [r.movie_id for r in user_ratings[:4]]
            
            try:
                recs = self.engine.get_recommendations(user.id, context_movies, algorithm)
                all_recommendations.extend(recs)
                
                # Calculate diversity for this user's recommendations
                if recs:
                    diversity_score = self.engine.calculate_diversity_score(recs)
                    genre_diversity_scores.append(diversity_score)
                    
                    # Calculate novelty (average inverse popularity)
                    movie_ids = [rec['movie_id'] for rec in recs]
                    popularities = []
                    
                    for movie_id in movie_ids:
                        rating_count = UserMovieRating.query.filter_by(movie_id=movie_id).count()
                        popularity = rating_count / UserMovieRating.query.count()
                        popularities.append(popularity)
                    
                    novelty = np.mean([1 - p for p in popularities]) if popularities else 0
                    novelty_scores.append(novelty)
                    
            except Exception as e:
                print(f"Error processing user {user.id}: {e}")
                continue
        
        # Calculate catalog coverage
        total_movies = Movie.query.count()
        unique_recommended = len(set(rec['movie_id'] for rec in all_recommendations))
        catalog_coverage = unique_recommended / total_movies if total_movies > 0 else 0
        
        results = {
            'algorithm': algorithm,
            'avg_diversity': np.mean(genre_diversity_scores) if genre_diversity_scores else 0,
            'avg_novelty': np.mean(novelty_scores) if novelty_scores else 0,
            'catalog_coverage': catalog_coverage,
            'unique_items_recommended': unique_recommended,
            'total_recommendations': len(all_recommendations)
        }
        
        print(f"📊 {algorithm} Diversity Results:")
        print(f"   Average Diversity: {results['avg_diversity']:.3f}")
        print(f"   Average Novelty: {results['avg_novelty']:.3f}")
        print(f"   Catalog Coverage: {results['catalog_coverage']:.1%}")
        
        return results
    
    def evaluate_all_algorithms(self):
        """Evaluate all recommendation algorithms"""
        print("🔍 Comprehensive Algorithm Evaluation")
        print("=" * 50)
        
        algorithms = ['content_based', 'collaborative', 'hybrid']
        all_results = {}
        
        for algorithm in algorithms:
            print(f"\n🚀 Evaluating {algorithm} algorithm...")
            
            try:
                # Rating prediction metrics
                rating_results = self.evaluate_rating_prediction(algorithm)
                
                # Ranking metrics
                ranking_results = self.evaluate_ranking_metrics(algorithm)
                
                # Diversity metrics
                diversity_results = self.evaluate_diversity_and_novelty(algorithm)
                
                # Combine results
                all_results[algorithm] = {
                    'rating_prediction': rating_results,
                    'ranking_metrics': ranking_results,
                    'diversity_metrics': diversity_results
                }
                
            except Exception as e:
                print(f"❌ Error evaluating {algorithm}: {e}")
                all_results[algorithm] = {'error': str(e)}
        
        # Print summary comparison
        self.print_algorithm_comparison(all_results)
        
        return all_results
    
    def print_algorithm_comparison(self, results):
        """Print comparison table of all algorithms"""
        print("\n📊 ALGORITHM COMPARISON SUMMARY")
        print("=" * 70)
        
        # Create comparison table
        comparison_data = []
        
        for algorithm, metrics in results.items():
            if 'error' in metrics:
                continue
                
            row = {
                'Algorithm': algorithm.replace('_', ' ').title(),
                'RMSE': metrics.get('rating_prediction', {}).get('rmse', 'N/A'),
                'MAE': metrics.get('rating_prediction', {}).get('mae', 'N/A'),
                'Precision@10': metrics.get('ranking_metrics', {}).get('precision_at_k', 'N/A'),
                'Recall@10': metrics.get('ranking_metrics', {}).get('recall_at_k', 'N/A'),
                'Diversity': metrics.get('diversity_metrics', {}).get('avg_diversity', 'N/A'),
                'Coverage': metrics.get('diversity_metrics', {}).get('catalog_coverage', 'N/A')
            }
            comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False, float_format='%.3f'))
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("- Content-Based: Good for new users, explainable recommendations")
        print("- Collaborative: Better accuracy with sufficient user data")
        print("- Hybrid: Balanced approach, combines strengths of both methods")
    
    def generate_evaluation_report(self, output_file='evaluation_report.txt'):
        """Generate detailed evaluation report"""
        results = self.evaluate_all_algorithms()
        
        with open(output_file, 'w') as f:
            f.write("MOVIE RECOMMENDATION SYSTEM - EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for algorithm, metrics in results.items():
                f.write(f"\n{algorithm.upper()} ALGORITHM\n")
                f.write("-" * 30 + "\n")
                
                if 'error' in metrics:
                    f.write(f"Error: {metrics['error']}\n")
                    continue
                
                # Rating prediction
                if 'rating_prediction' in metrics:
                    rp = metrics['rating_prediction']
                    f.write(f"Rating Prediction:\n")
                    f.write(f"  RMSE: {rp.get('rmse', 'N/A'):.3f}\n")
                    f.write(f"  MAE: {rp.get('mae', 'N/A'):.3f}\n")
                    f.write(f"  Coverage: {rp.get('coverage', 'N/A'):.1%}\n\n")
                
                # Ranking metrics
                if 'ranking_metrics' in metrics:
                    rm = metrics['ranking_metrics']
                    f.write(f"Ranking Metrics @10:\n")
                    f.write(f"  Precision: {rm.get('precision_at_k', 'N/A'):.3f}\n")
                    f.write(f"  Recall: {rm.get('recall_at_k', 'N/A'):.3f}\n")
                    f.write(f"  NDCG: {rm.get('ndcg_at_k', 'N/A'):.3f}\n\n")
                
                # Diversity metrics
                if 'diversity_metrics' in metrics:
                    dm = metrics['diversity_metrics']
                    f.write(f"Diversity Metrics:\n")
                    f.write(f"  Average Diversity: {dm.get('avg_diversity', 'N/A'):.3f}\n")
                    f.write(f"  Average Novelty: {dm.get('avg_novelty', 'N/A'):.3f}\n")
                    f.write(f"  Catalog Coverage: {dm.get('catalog_coverage', 'N/A'):.1%}\n\n")
        
        print(f"📄 Detailed report saved to {output_file}")

def main():
    """Main function for running evaluations"""
    from app import app
    
    with app.app_context():
        evaluator = ModelEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_all_algorithms()
        
        # Generate report
        evaluator.generate_evaluation_report()
        
        print("\n✅ Evaluation completed!")

if __name__ == '__main__':
    main()