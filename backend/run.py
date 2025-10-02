#!/usr/bin/env python3
"""
Application runner script
Use this script to start the Flask application with different configurations
"""

import os
import sys
from app import app, init_db

def main():
    """Main function to run the application"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'init-db':
            print("Initializing database...")
            init_db()
            print("Database initialized successfully!")
            return
            
        elif command == 'reset-db':
            print("Resetting database...")
            with app.app_context():
                from init_db import reset_database
                reset_database()
            return
            
        elif command == 'shell':
            print("Starting Flask shell...")
            with app.app_context():
                from models import Movie, User, UserInteraction, UserMovieRating, UserProfile, MovieTag, Recommendation
                import code
                code.interact(local=locals())
            return
            
        elif command == 'import-movielens':
            print("📥 Importing MovieLens dataset...")
            from movielens_loader import MovieLensLoader
            
            # Parse additional arguments
            dataset_size = 'small'
            sample_users = 1000
            sample_ratings = 0.1
            
            if len(sys.argv) > 2:
                dataset_size = sys.argv[2]
            if len(sys.argv) > 3:
                sample_users = int(sys.argv[3])
            if len(sys.argv) > 4:
                sample_ratings = float(sys.argv[4])
            
            loader = MovieLensLoader(dataset_size)
            with app.app_context():
                result = loader.load_and_import(sample_users, sample_ratings)
                print(f"✅ MovieLens import completed: {result}")
            return
            
        elif command == 'train-model':
            print("🤖 Training recommendation model...")
            with app.app_context():
                from recommendation_engine import RecommendationEngine
                engine = RecommendationEngine()
                engine.initialize()
                print("✅ Model training completed!")
            return
            
        elif command == 'evaluate-model':
            print("📊 Evaluating recommendation model...")
            with app.app_context():
                from model_evaluation import ModelEvaluator
                evaluator = ModelEvaluator()
                results = evaluator.evaluate_all_algorithms()
                print(f"📈 Evaluation results: {results}")
            return
            
        elif command in ['help', '-h', '--help']:
            print_help()
            return
    
    # Set environment variables if not set
    if not os.environ.get('FLASK_CONFIG'):
        os.environ['FLASK_CONFIG'] = 'development'
    
    # Initialize database if needed
    init_db()
    
    # Run the application
    print(f"Starting Flask application in {os.environ.get('FLASK_CONFIG', 'development')} mode...")
    print("Access the application at: http://localhost:5000")
    print("API documentation available at: http://localhost:5000/api/docs")
    print("Press CTRL+C to stop the server")
    
    app.run(
        debug=os.environ.get('FLASK_CONFIG') == 'development',
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )

def print_help():
    """Print help information"""
    help_text = """
Flask Movie Recommendation System - Usage Guide

Commands:
  python run.py                 - Start the application server
  python run.py init-db         - Initialize the database with sample data
  python run.py reset-db        - Reset the database (WARNING: This will delete all data)
  python run.py shell           - Start an interactive Flask shell
  python run.py test            - Run the test suite
  python run.py help            - Show this help message

Environment Variables:
  FLASK_CONFIG                  - Configuration mode (development, testing, production)
  DATABASE_URL                  - Database connection string
  SECRET_KEY                    - Secret key for sessions and security
  PORT                          - Port to run the application on (default: 5000)

Examples:
  # Start in development mode
  FLASK_CONFIG=development python run.py
  
  # Start in production mode
  FLASK_CONFIG=production python run.py
  
  # Initialize database with sample data
  python run.py init-db
  
  # Reset database (be careful!)
  python run.py reset-db

For more information, visit: https://github.com/yourusername/movie-recommendation-system
    """
    print(help_text)

if __name__ == '__main__':
    main()