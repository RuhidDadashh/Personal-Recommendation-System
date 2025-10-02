from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import datetime, timedelta
import os
from functools import wraps
import secrets

# Initialize Flask app
app = Flask(__name__)

# Load configuration
from config import config
config_name = os.environ.get('FLASK_CONFIG', 'development')
app.config.from_object(config[config_name])
config[config_name].init_app(app)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
CORS(app)

# Import models after db initialization
from models import db as models_db
models_db.init_app(app)

from models import User, Movie, UserMovieRating, Recommendation, UserInteraction, UserProfile

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

# API Routes for Authentication
@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.capitalize()} is required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Validate password strength
        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Create new user
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        user = User(
            name=data['name'],
            email=data['email'],
            password=hashed_password
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Create user profile
        profile = UserProfile(user_id=user.id)
        db.session.add(profile)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(email=data['email']).first()
        
        # Check credentials (accept demo credentials or any valid user)
        if (data['email'] == 'demo@example.com' and data['password'] == 'demo123') or \
           (user and bcrypt.check_password_hash(user.password, data['password'])):
            
            if not user:
                # Create demo user if it doesn't exist
                user = User(
                    name='Demo User',
                    email='demo@example.com',
                    password=bcrypt.generate_password_hash('demo123').decode('utf-8')
                )
                db.session.add(user)
                db.session.commit()
            
            # Create session
            session['user_id'] = user.id
            session['user_email'] = user.email
            session.permanent = True
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': user.to_dict()
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

# API Routes for Movies
@app.route('/api/movies', methods=['GET'])
def api_get_movies():
    try:
        movies = Movie.query.all()
        movies_data = [movie.to_dict() for movie in movies]
        return jsonify({'movies': movies_data}), 200
        
    except Exception as e:
        app.logger.error(f"Error fetching movies: {str(e)}")
        return jsonify({'error': 'Failed to fetch movies'}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def api_get_movie(movie_id):
    try:
        movie = Movie.query.get_or_404(movie_id)
        return jsonify(movie.to_dict()), 200
        
    except Exception as e:
        return jsonify({'error': 'Movie not found'}), 404

# API Routes for Recommendations
@app.route('/api/recommendations', methods=['POST'])
@login_required
def api_get_recommendations():
    try:
        data = request.get_json()
        selected_movies = data.get('selected_movies', [])
        algorithm = data.get('algorithm', 'hybrid')
        
        if not selected_movies:
            return jsonify({'error': 'Please select at least one movie'}), 400
        
        # Import and use recommendation engine
        from recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine()
        recommendations = engine.get_recommendations(
            user_id=session['user_id'],
            selected_movies=selected_movies,
            algorithm=algorithm
        )
        
        # Store recommendation in database for analytics
        for rec in recommendations:
            recommendation = Recommendation(
                user_id=session['user_id'],
                movie_id=rec['movie_id'],
                algorithm=algorithm,
                score=rec['score']
            )
            db.session.add(recommendation)
        
        # Track user interaction
        for movie_id in selected_movies:
            UserInteraction.track_interaction(
                user_id=session['user_id'],
                movie_id=movie_id,
                interaction_type='selected',
                metadata={'algorithm': algorithm}
            )
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'algorithm': algorithm,
            'diversity_score': engine.calculate_diversity_score(recommendations)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Recommendation error: {str(e)}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

# API Routes for User Interactions
@app.route('/api/rate_movie', methods=['POST'])
@login_required
def api_rate_movie():
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        
        if not movie_id or rating is None:
            return jsonify({'error': 'Movie ID and rating are required'}), 400
        
        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        # Check if rating already exists
        existing_rating = UserMovieRating.query.filter_by(
            user_id=session['user_id'],
            movie_id=movie_id
        ).first()
        
        if existing_rating:
            existing_rating.rating = rating
            existing_rating.updated_at = datetime.utcnow()
        else:
            new_rating = UserMovieRating(
                user_id=session['user_id'],
                movie_id=movie_id,
                rating=rating
            )
            db.session.add(new_rating)
        
        # Track interaction
        UserInteraction.track_interaction(
            user_id=session['user_id'],
            movie_id=movie_id,
            interaction_type='rating',
            metadata={'rating': rating}
        )
        
        db.session.commit()
        
        return jsonify({'message': 'Rating saved successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Rating error: {str(e)}")
        return jsonify({'error': 'Failed to save rating'}), 500

@app.route('/api/user_stats', methods=['GET'])
@login_required
def api_user_stats():
    try:
        user_id = session['user_id']
        
        # Get user statistics
        total_ratings = UserMovieRating.query.filter_by(user_id=user_id).count()
        total_recommendations = Recommendation.query.filter_by(user_id=user_id).count()
        total_interactions = UserInteraction.query.filter_by(user_id=user_id).count()
        
        # Get favorite genres
        user_ratings = db.session.query(
            Movie.genre, 
            db.func.avg(UserMovieRating.rating).label('avg_rating'),
            db.func.count(UserMovieRating.rating).label('count')
        ).join(
            UserMovieRating, Movie.id == UserMovieRating.movie_id
        ).filter(
            UserMovieRating.user_id == user_id
        ).group_by(Movie.genre).order_by(
            db.func.avg(UserMovieRating.rating).desc()
        ).limit(5).all()
        
        favorite_genres = []
        for genre, avg_rating, count in user_ratings:
            favorite_genres.append({
                'genre': genre,
                'avg_rating': float(avg_rating),
                'count': count
            })
        
        return jsonify({
            'total_ratings': total_ratings,
            'total_recommendations': total_recommendations,
            'total_interactions': total_interactions,
            'favorite_genres': favorite_genres
        }), 200
        
    except Exception as e:
        app.logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': 'Failed to get user stats'}), 500

@app.route('/api/trending', methods=['GET'])
def api_trending():
    """Get trending movies"""
    try:
        from recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        trending = engine.get_trending_movies()
        return jsonify({'trending': trending}), 200
    except Exception as e:
        app.logger.error(f"Trending error: {str(e)}")
        return jsonify({'error': 'Failed to get trending movies'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

# Database initialization function
def init_db():
    """Initialize database with sample data if needed"""
    with app.app_context():
        db.create_all()
        
        # Check if we need to add sample data
        if Movie.query.count() == 0:
            from init_db import init_database
            init_database()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

# API Routes for Authentication
@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.capitalize()} is required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Validate password strength
        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Create new user
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        user = User(
            name=data['name'],
            email=data['email'],
            password=hashed_password
        )
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User registered successfully',
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(email=data['email']).first()
        
        # Check credentials
        if user and bcrypt.check_password_hash(user.password, data['password']):
            # Create session
            session['user_id'] = user.id
            session['user_email'] = user.email
            session.permanent = True
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'message': 'Login successful',
                'user': {
                    'id': user.id,
                    'name': user.name,
                    'email': user.email
                }
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

# API Routes for Movies
@app.route('/api/movies', methods=['GET'])
def api_get_movies():
    try:
        movies = Movie.query.all()
        movies_data = []
        
        for movie in movies:
            movies_data.append({
                'id': movie.id,
                'title': movie.title,
                'genre': movie.genre,
                'rating': movie.rating,
                'year': movie.year,
                'director': movie.director,
                'description': movie.description
            })
        
        return jsonify({'movies': movies_data}), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to fetch movies'}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def api_get_movie(movie_id):
    try:
        movie = Movie.query.get_or_404(movie_id)
        return jsonify({
            'id': movie.id,
            'title': movie.title,
            'genre': movie.genre,
            'rating': movie.rating,
            'year': movie.year,
            'director': movie.director,
            'description': movie.description
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Movie not found'}), 404

# API Routes for Recommendations
@app.route('/api/recommendations', methods=['POST'])
@login_required
def api_get_recommendations():
    try:
        data = request.get_json()
        selected_movies = data.get('selected_movies', [])
        algorithm = data.get('algorithm', 'hybrid')
        
        if not selected_movies:
            return jsonify({'error': 'Please select at least one movie'}), 400
        
        # Import recommendation engine (we'll create this next)
        from recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine()
        recommendations = engine.get_recommendations(
            user_id=session['user_id'],
            selected_movies=selected_movies,
            algorithm=algorithm
        )
        
        # Store recommendation in database for analytics
        for rec in recommendations:
            recommendation = Recommendation(
                user_id=session['user_id'],
                movie_id=rec['movie_id'],
                algorithm=algorithm,
                score=rec['score']
            )
            db.session.add(recommendation)
        
        db.session.commit()
        
        return jsonify({
            'recommendations': recommendations,
            'algorithm': algorithm,
            'diversity_score': engine.calculate_diversity_score(recommendations)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to get recommendations'}), 500

# API Routes for User Interactions
@app.route('/api/rate_movie', methods=['POST'])
@login_required
def api_rate_movie():
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        
        if not movie_id or rating is None:
            return jsonify({'error': 'Movie ID and rating are required'}), 400
        
        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        # Check if rating already exists
        existing_rating = UserMovieRating.query.filter_by(
            user_id=session['user_id'],
            movie_id=movie_id
        ).first()
        
        if existing_rating:
            existing_rating.rating = rating
            existing_rating.updated_at = datetime.utcnow()
        else:
            new_rating = UserMovieRating(
                user_id=session['user_id'],
                movie_id=movie_id,
                rating=rating
            )
            db.session.add(new_rating)
        
        db.session.commit()
        
        return jsonify({'message': 'Rating saved successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to save rating'}), 500

@app.route('/api/user_stats', methods=['GET'])
@login_required
def api_user_stats():
    try:
        user_id = session['user_id']
        
        # Get user statistics
        total_ratings = UserMovieRating.query.filter_by(user_id=user_id).count()
        total_recommendations = Recommendation.query.filter_by(user_id=user_id).count()
        
        # Get favorite genres
        user_ratings = db.session.query(
            Movie.genre, 
            db.func.avg(UserMovieRating.rating).label('avg_rating')
        ).join(
            UserMovieRating, Movie.id == UserMovieRating.movie_id
        ).filter(
            UserMovieRating.user_id == user_id
        ).group_by(Movie.genre).order_by(
            db.func.avg(UserMovieRating.rating).desc()
        ).limit(5).all()
        
        favorite_genres = [{'genre': genre, 'avg_rating': float(avg_rating)} 
                          for genre, avg_rating in user_ratings]
        
        return jsonify({
            'total_ratings': total_ratings,
            'total_recommendations': total_recommendations,
            'favorite_genres': favorite_genres
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get user stats'}), 500

# Database initialization
def init_db():
    with app.app_context():
        db.create_all()
        
        # Add sample movies if database is empty
        if Movie.query.count() == 0:
            sample_movies = [
                {
                    'title': 'The Shawshank Redemption',
                    'genre': 'Drama',
                    'rating': 9.3,
                    'year': 1994,
                    'director': 'Frank Darabont',
                    'description': 'Two imprisoned men bond over years, finding solace and eventual redemption.'
                },
                {
                    'title': 'The Godfather',
                    'genre': 'Crime, Drama',
                    'rating': 9.2,
                    'year': 1972,
                    'director': 'Francis Ford Coppola',
                    'description': 'The aging patriarch of an organized crime dynasty transfers control to his son.'
                },
                {
                    'title': 'The Dark Knight',
                    'genre': 'Action, Crime',
                    'rating': 9.0,
                    'year': 2008,
                    'director': 'Christopher Nolan',
                    'description': 'Batman must accept one of the greatest psychological and physical tests.'
                },
                {
                    'title': 'Pulp Fiction',
                    'genre': 'Crime, Drama',
                    'rating': 8.9,
                    'year': 1994,
                    'director': 'Quentin Tarantino',
                    'description': 'The lives of two mob hitmen, a boxer, and others intertwine.'
                },
                {
                    'title': 'Forrest Gump',
                    'genre': 'Drama, Romance',
                    'rating': 8.8,
                    'year': 1994,
                    'director': 'Robert Zemeckis',
                    'description': 'The presidencies of Kennedy and Johnson through the eyes of Alabama man.'
                },
                {
                    'title': 'Inception',
                    'genre': 'Sci-Fi, Thriller',
                    'rating': 8.8,
                    'year': 2010,
                    'director': 'Christopher Nolan',
                    'description': 'A thief who steals corporate secrets through dream-sharing technology.'
                },
                {
                    'title': 'The Matrix',
                    'genre': 'Sci-Fi, Action',
                    'rating': 8.7,
                    'year': 1999,
                    'director': 'The Wachowskis',
                    'description': 'A computer programmer discovers reality is a simulation.'
                },
                {
                    'title': 'Goodfellas',
                    'genre': 'Crime, Drama',
                    'rating': 8.7,
                    'year': 1990,
                    'director': 'Martin Scorsese',
                    'description': 'The story of Henry Hill and his life in the mob.'
                },
                {
                    'title': 'Interstellar',
                    'genre': 'Sci-Fi, Drama',
                    'rating': 8.6,
                    'year': 2014,
                    'director': 'Christopher Nolan',
                    'description': 'A team of explorers travel through a wormhole in space.'
                },
                {
                    'title': 'The Lion King',
                    'genre': 'Animation, Family',
                    'rating': 8.5,
                    'year': 1994,
                    'director': 'Roger Allers',
                    'description': 'Lion prince Simba flees his kingdom after his father\'s murder.'
                },
                {
                    'title': 'Parasite',
                    'genre': 'Thriller, Drama',
                    'rating': 8.6,
                    'year': 2019,
                    'director': 'Bong Joon-ho',
                    'description': 'Greed and class discrimination threaten a newly formed symbiotic relationship.'
                },
                {
                    'title': 'Avengers: Endgame',
                    'genre': 'Action, Adventure',
                    'rating': 8.4,
                    'year': 2019,
                    'director': 'Anthony Russo',
                    'description': 'The Avengers assemble once more to reverse Thanos\' actions.'
                }
            ]
            
            for movie_data in sample_movies:
                movie = Movie(**movie_data)
                db.session.add(movie)
            
            db.session.commit()
            print("Sample movies added to database!")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)