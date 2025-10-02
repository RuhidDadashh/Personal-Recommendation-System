from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy (imported in app.py)
db = SQLAlchemy()

class MovieTag(db.Model):
    """Model for storing movie tags from MovieLens dataset"""
    __tablename__ = 'movie_tags'
    
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    tag = db.Column(db.String(100), nullable=False, index=True)
    relevance = db.Column(db.Float, default=1.0)  # Tag relevance score
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MovieTag {self.tag} for Movie:{self.movie_id}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'movie_id': self.movie_id,
            'user_id': self.user_id,
            'tag': self.tag,
            'relevance': self.relevance,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def get_movie_tags(cls, movie_id, limit=10):
        """Get tags for a specific movie"""
        return cls.query.filter_by(movie_id=movie_id).order_by(
            cls.relevance.desc()
        ).limit(limit).all()
    
    @classmethod
    def get_popular_tags(cls, limit=50):
        """Get most popular tags across all movies"""
        return db.session.query(
            cls.tag,
            db.func.count(cls.id).label('count'),
            db.func.avg(cls.relevance).label('avg_relevance')
        ).group_by(cls.tag).order_by(
            db.func.count(cls.id).desc()
        ).limit(limit).all()

class User(db.Model):
    """User model for storing user account information"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    movielens_id = db.Column(db.Integer, unique=True, index=True)  # Original MovieLens user ID
    is_training_user = db.Column(db.Boolean, default=False)  # Flag for MovieLens training users
    
    # Relationships
    ratings = db.relationship('UserMovieRating', backref='user', lazy=True, cascade='all, delete-orphan')
    recommendations = db.relationship('Recommendation', backref='user', lazy=True, cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', backref='user', lazy=True, cascade='all, delete-orphan')
    tags = db.relationship('MovieTag', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<User {self.email}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_training_user': self.is_training_user,
            'movielens_id': self.movielens_id
        }
    
    def get_rating_stats(self):
        """Get user's rating statistics"""
        ratings = [r.rating for r in self.ratings]
        if not ratings:
            return {'count': 0, 'average': 0, 'std': 0}
        
        return {
            'count': len(ratings),
            'average': sum(ratings) / len(ratings),
            'min': min(ratings),
            'max': max(ratings),
            'std': np.std(ratings) if len(ratings) > 1 else 0
        }
    
    def get_favorite_genres(self, limit=5):
        """Get user's favorite genres based on ratings"""
        genre_ratings = defaultdict(list)
        
        for rating in self.ratings:
            if rating.rating >= 4.0:  # Consider ratings >= 4 as liked
                genres = rating.movie.get_genres_list()
                for genre in genres:
                    genre_ratings[genre].append(rating.rating)
        
        # Calculate average rating for each genre
        genre_averages = []
        for genre, ratings in genre_ratings.items():
            avg_rating = sum(ratings) / len(ratings)
            genre_averages.append({
                'genre': genre,
                'avg_rating': avg_rating,
                'count': len(ratings)
            })
        
        return sorted(genre_averages, key=lambda x: x['avg_rating'], reverse=True)[:limit]
    
    @classmethod
    def get_training_users(cls, limit=None):
        """Get MovieLens training users"""
        query = cls.query.filter_by(is_training_user=True)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    @classmethod
    def get_active_users(cls, days=30):
        """Get users active in the last N days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return cls.query.filter(
            db.or_(
                cls.last_login >= cutoff_date,
                cls.created_at >= cutoff_date
            )
        ).all()

class Movie(db.Model):
    """Movie model for storing movie information"""
    __tablename__ = 'movies'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False, index=True)
    genre = db.Column(db.String(100), nullable=False, index=True)
    rating = db.Column(db.Float, nullable=False)
    year = db.Column(db.Integer, nullable=False, index=True)
    director = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    poster_url = db.Column(db.String(500))
    imdb_id = db.Column(db.String(20), unique=True)
    movielens_id = db.Column(db.Integer, unique=True, index=True)  # MovieLens ID
    tmdb_id = db.Column(db.Integer, unique=True)  # TMDB ID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user_ratings = db.relationship('UserMovieRating', backref='movie', lazy=True, cascade='all, delete-orphan')
    recommendations = db.relationship('Recommendation', backref='movie', lazy=True, cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', backref='movie', lazy=True, cascade='all, delete-orphan')
    tags = db.relationship('MovieTag', backref='movie', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Movie {self.title} ({self.year})>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'genre': self.genre,
            'rating': self.rating,
            'year': self.year,
            'director': self.director,
            'description': self.description,
            'poster_url': self.poster_url,
            'movielens_id': self.movielens_id
        }
    
    def get_genres_list(self):
        """Return list of genres for this movie"""
        return [genre.strip() for genre in self.genre.split(',')]
    
    def get_average_rating(self):
        """Get average user rating for this movie"""
        ratings = [r.rating for r in self.user_ratings]
        return sum(ratings) / len(ratings) if ratings else 0.0
    
    def get_rating_count(self):
        """Get number of ratings for this movie"""
        return len(self.user_ratings)
    
    @classmethod
    def get_by_genre(cls, genre):
        """Get movies by genre"""
        return cls.query.filter(cls.genre.like(f'%{genre}%')).all()
    
    @classmethod
    def get_top_rated(cls, limit=10, min_ratings=10):
        """Get top rated movies with minimum number of ratings"""
        # Subquery to get movies with minimum ratings count
        subquery = db.session.query(
            UserMovieRating.movie_id,
            db.func.count(UserMovieRating.rating).label('rating_count'),
            db.func.avg(UserMovieRating.rating).label('avg_rating')
        ).group_by(UserMovieRating.movie_id).having(
            db.func.count(UserMovieRating.rating) >= min_ratings
        ).subquery()
        
        return db.session.query(cls).join(
            subquery, cls.id == subquery.c.movie_id
        ).order_by(subquery.c.avg_rating.desc()).limit(limit).all()
    
    @classmethod
    def search_movies(cls, query, limit=20):
        """Search movies by title or genre"""
        return cls.query.filter(
            db.or_(
                cls.title.ilike(f'%{query}%'),
                cls.genre.ilike(f'%{query}%')
            )
        ).limit(limit).all()

class UserMovieRating(db.Model):
    """Model for storing user ratings of movies"""
    __tablename__ = 'user_movie_ratings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)  # Rating from 1-5
    review = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Composite unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id'),)
    
    def __repr__(self):
        return f"<UserMovieRating User:{self.user_id} Movie:{self.movie_id} Rating:{self.rating}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'movie_id': self.movie_id,
            'rating': self.rating,
            'review': self.review,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def get_user_ratings(cls, user_id):
        """Get all ratings by a user"""
        return cls.query.filter_by(user_id=user_id).all()
    
    @classmethod
    def get_movie_ratings(cls, movie_id):
        """Get all ratings for a movie"""
        return cls.query.filter_by(movie_id=movie_id).all()
    
    @classmethod
    def get_average_rating(cls, movie_id):
        """Get average rating for a movie"""
        result = db.session.query(db.func.avg(cls.rating)).filter_by(movie_id=movie_id).scalar()
        return float(result) if result else 0.0

class Recommendation(db.Model):
    """Model for storing recommendation history"""
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.id'), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)  # 'collaborative', 'content-based', 'hybrid'
    score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    clicked = db.Column(db.Boolean, default=False)
    feedback_rating = db.Column(db.Float)  # User feedback on recommendation quality
    
    def __repr__(self):
        return f"<Recommendation User:{self.user_id} Movie:{self.movie_id} Score:{self.score}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'movie_id': self.movie_id,
            'algorithm': self.algorithm,
            'score': self.score,
            'created_at': self.created_at.isoformat(),
            'clicked': self.clicked,
            'feedback_rating': self.feedback_rating
        }
    
    @classmethod
    def get_user_recommendations(cls, user_id, limit=10):
        """Get recent recommendations for a user"""
        return cls.query.filter_by(user_id=user_id).order_by(cls.created_at.desc()).limit(limit).all()
    
    @classmethod
    def get_algorithm_performance(cls, algorithm):
        """Get performance metrics for an algorithm"""
        recommendations = cls.query.filter_by(algorithm=algorithm).all()
        if not recommendations:
            return {'accuracy': 0, 'click_rate': 0}
        
        clicked_count = sum(1 for r in recommendations if r.clicked)
        feedback_ratings = [r.feedback_rating for r in recommendations if r.feedback_rating is not None]
        
        return {
            'click_rate': clicked_count / len(recommendations) if recommendations else 0,
            'avg_feedback': sum(feedback_ratings) / len(feedback_ratings) if feedback_ratings else 0,
            'total_recommendations': len(recommendations)
        }

class UserInteraction(db.Model):
    """Model for tracking user interactions with movies and recommendations"""
    __tablename__ = 'user_interactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.id'), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)  # 'view', 'click', 'like', 'share', 'watch'
    session_id = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    interaction_data = db.Column(db.JSON)  # Store additional interaction data (renamed from metadata)
    
    def __repr__(self):
        return f"<UserInteraction User:{self.user_id} Movie:{self.movie_id} Type:{self.interaction_type}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'movie_id': self.movie_id,
            'interaction_type': self.interaction_type,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'interaction_data': self.interaction_data
        }
    
    @classmethod
    def track_interaction(cls, user_id, movie_id, interaction_type, session_id=None, interaction_data=None):
        """Track a new user interaction"""
        interaction = cls(
            user_id=user_id,
            movie_id=movie_id,
            interaction_type=interaction_type,
            session_id=session_id,
            interaction_data=interaction_data
        )
        db.session.add(interaction)
        db.session.commit()
        return interaction
    
    @classmethod
    def get_user_interactions(cls, user_id, interaction_type=None, limit=None):
        """Get user interactions, optionally filtered by type"""
        query = cls.query.filter_by(user_id=user_id)
        
        if interaction_type:
            query = query.filter_by(interaction_type=interaction_type)
        
        query = query.order_by(cls.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @classmethod
    def get_popular_movies(cls, days=30, interaction_type=None, limit=10):
        """Get popular movies based on interactions"""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.session.query(
            cls.movie_id,
            db.func.count(cls.id).label('interaction_count')
        ).filter(cls.timestamp >= cutoff_date)
        
        if interaction_type:
            query = query.filter_by(interaction_type=interaction_type)
        
        return query.group_by(cls.movie_id).order_by(
            db.func.count(cls.id).desc()
        ).limit(limit).all()

class UserProfile(db.Model):
    """Extended user profile for recommendation preferences"""
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    preferred_genres = db.Column(db.JSON)  # List of preferred genres
    preferred_decades = db.Column(db.JSON)  # List of preferred decades
    min_rating_threshold = db.Column(db.Float, default=6.0)
    language_preference = db.Column(db.String(10), default='en')
    content_maturity = db.Column(db.String(10), default='PG-13')  # G, PG, PG-13, R
    recommendation_frequency = db.Column(db.String(20), default='weekly')  # daily, weekly, monthly
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('profile', uselist=False))
    
    def __repr__(self):
        return f"<UserProfile User:{self.user_id}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'preferred_genres': self.preferred_genres,
            'preferred_decades': self.preferred_decades,
            'min_rating_threshold': self.min_rating_threshold,
            'language_preference': self.language_preference,
            'content_maturity': self.content_maturity,
            'recommendation_frequency': self.recommendation_frequency,
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def get_or_create(cls, user_id):
        """Get existing profile or create new one"""
        profile = cls.query.filter_by(user_id=user_id).first()
        if not profile:
            profile = cls(user_id=user_id)
            db.session.add(profile)
            db.session.commit()
        return profile
    
    def update_preferences(self, **kwargs):
        """Update user preferences"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        db.session.commit()

# Create indexes for better performance
def create_indexes():
    """Create additional database indexes for better query performance"""
    db.Index('idx_movie_genre_rating', Movie.genre, Movie.rating)
    db.Index('idx_user_rating_movie', UserMovieRating.user_id, UserMovieRating.movie_id)
    db.Index('idx_recommendation_user_created', Recommendation.user_id, Recommendation.created_at)
    db.Index('idx_interaction_user_timestamp', UserInteraction.user_id, UserInteraction.timestamp)
    db.Index('idx_interaction_movie_type', UserInteraction.movie_id, UserInteraction.interaction_type)