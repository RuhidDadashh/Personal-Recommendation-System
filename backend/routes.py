"""
Application routes
"""

from flask import render_template, request, jsonify, session, redirect, url_for
from functools import wraps
from datetime import datetime
from models import db, User, Movie, UserMovieRating, Recommendation, UserInteraction, UserProfile
from create_app import bcrypt

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def register_routes(app):
    """Register all application routes"""
    
    # Page routes
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
            
            # Check credentials properly
            if user and bcrypt.check_password_hash(user.password, data['password']):
                # Valid existing user
                pass
            elif data['email'] == 'demo@example.com' and data['password'] == 'demo123':
                # Demo user - create if doesn't exist
                if not user:
                    user = User(
                        name='Demo User',
                        email='demo@example.com',
                        password=bcrypt.generate_password_hash('demo123').decode('utf-8')
                    )
                    db.session.add(user)
                    db.session.commit()
            else:
                # Invalid credentials
                return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
            
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
            # Get query parameters
            search = request.args.get('search', '').strip()
            genre = request.args.get('genre', '').strip()
            limit = min(int(request.args.get('limit', 50)), 100)  # Max 100 movies
            page = max(int(request.args.get('page', 1)), 1)
            offset = (page - 1) * limit
            
            # Build query
            query = Movie.query
            
            # Apply search filter
            if search:
                query = query.filter(
                    db.or_(
                        Movie.title.ilike(f'%{search}%'),
                        Movie.director.ilike(f'%{search}%')
                    )
                )
            
            # Apply genre filter
            if genre:
                query = query.filter(Movie.genre.ilike(f'%{genre}%'))
            
            # Order by rating and year (most popular first)
            query = query.order_by(Movie.rating.desc(), Movie.year.desc())
            
            # Get total count for pagination
            total = query.count()
            
            # Apply pagination
            movies = query.offset(offset).limit(limit).all()
            
            # Convert to dict
            movies_data = []
            for movie in movies:
                movie_dict = movie.to_dict()
                # Add additional info
                movie_dict['avg_user_rating'] = movie.get_average_rating()
                movie_dict['rating_count'] = movie.get_rating_count()
                movies_data.append(movie_dict)
            
            return jsonify({
                'movies': movies_data,
                'pagination': {
                    'page': page,
                    'per_page': limit,
                    'total': total,
                    'pages': (total + limit - 1) // limit
                }
            }), 200
            
        except Exception as e:
            app.logger.error(f"Error fetching movies: {str(e)}")
            return jsonify({'error': 'Failed to fetch movies'}), 500

    @app.route('/api/movies/search', methods=['GET'])
    def api_search_movies():
        try:
            query = request.args.get('q', '').strip()
            if not query:
                return jsonify({'movies': []}), 200
            
            # Search movies by title, director, or genre
            movies = Movie.query.filter(
                db.or_(
                    Movie.title.ilike(f'%{query}%'),
                    Movie.director.ilike(f'%{query}%'),
                    Movie.genre.ilike(f'%{query}%')
                )
            ).order_by(Movie.rating.desc()).limit(20).all()
            
            movies_data = [movie.to_dict() for movie in movies]
            
            return jsonify({'movies': movies_data}), 200
            
        except Exception as e:
            app.logger.error(f"Error searching movies: {str(e)}")
            return jsonify({'error': 'Search failed'}), 500

    @app.route('/api/genres', methods=['GET'])
    def api_get_genres():
        try:
            # Get all unique genres from database
            movies = Movie.query.all()
            genres = set()
            
            for movie in movies:
                movie_genres = movie.get_genres_list()
                genres.update(movie_genres)
            
            # Sort genres alphabetically
            sorted_genres = sorted(list(genres))
            
            return jsonify({'genres': sorted_genres}), 200
            
        except Exception as e:
            app.logger.error(f"Error fetching genres: {str(e)}")
            return jsonify({'error': 'Failed to fetch genres'}), 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not found'}), 404
        return render_template('index.html'), 404  # Redirect to home for frontend routes

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('index.html'), 500