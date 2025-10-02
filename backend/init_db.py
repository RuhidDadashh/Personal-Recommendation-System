"""
Database initialization script
Run this script to set up the database with sample data
"""

from app import app, db
from models import User, Movie, UserMovieRating, UserProfile, UserInteraction
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
import random

bcrypt = Bcrypt()

def create_sample_movies():
    """Create sample movies for the database"""
    sample_movies = [
        {
            'title': 'The Shawshank Redemption',
            'genre': 'Drama',
            'rating': 9.3,
            'year': 1994,
            'director': 'Frank Darabont',
            'description': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'imdb_id': 'tt0111161'
        },
        {
            'title': 'The Godfather',
            'genre': 'Crime, Drama',
            'rating': 9.2,
            'year': 1972,
            'director': 'Francis Ford Coppola',
            'description': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'imdb_id': 'tt0068646'
        },
        {
            'title': 'The Dark Knight',
            'genre': 'Action, Crime, Drama',
            'rating': 9.0,
            'year': 2008,
            'director': 'Christopher Nolan',
            'description': 'When the menace known as the Joker wreaks havoc on Gotham, Batman must accept one of the greatest psychological tests.',
            'imdb_id': 'tt0468569'
        },
        {
            'title': 'Pulp Fiction',
            'genre': 'Crime, Drama',
            'rating': 8.9,
            'year': 1994,
            'director': 'Quentin Tarantino',
            'description': 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
            'imdb_id': 'tt0110912'
        },
        {
            'title': 'Forrest Gump',
            'genre': 'Drama, Romance',
            'rating': 8.8,
            'year': 1994,
            'director': 'Robert Zemeckis',
            'description': 'The presidencies of Kennedy and Johnson through the eyes of an Alabama man with an IQ of 75.',
            'imdb_id': 'tt0109830'
        },
        {
            'title': 'Inception',
            'genre': 'Action, Sci-Fi, Thriller',
            'rating': 8.8,
            'year': 2010,
            'director': 'Christopher Nolan',
            'description': 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task.',
            'imdb_id': 'tt1375666'
        },
        {
            'title': 'The Matrix',
            'genre': 'Action, Sci-Fi',
            'rating': 8.7,
            'year': 1999,
            'director': 'The Wachowskis',
            'description': 'A computer programmer is led to fight an underground war against powerful computers.',
            'imdb_id': 'tt0133093'
        },
        {
            'title': 'Goodfellas',
            'genre': 'Biography, Crime, Drama',
            'rating': 8.7,
            'year': 1990,
            'director': 'Martin Scorsese',
            'description': 'The story of Henry Hill and his life in the mob, covering his relationship with his wife.',
            'imdb_id': 'tt0099685'
        },
        {
            'title': 'Interstellar',
            'genre': 'Adventure, Drama, Sci-Fi',
            'rating': 8.6,
            'year': 2014,
            'director': 'Christopher Nolan',
            'description': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
            'imdb_id': 'tt0816692'
        },
        {
            'title': 'The Lion King',
            'genre': 'Animation, Adventure, Drama, Family',
            'rating': 8.5,
            'year': 1994,
            'director': 'Roger Allers, Rob Minkoff',
            'description': 'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
            'imdb_id': 'tt0110357'
        },
        {
            'title': 'Parasite',
            'genre': 'Comedy, Drama, Thriller',
            'rating': 8.6,
            'year': 2019,
            'director': 'Bong Joon Ho',
            'description': 'Greed and class discrimination threaten the newly formed symbiotic relationship between the Park family.',
            'imdb_id': 'tt6751668'
        },
        {
            'title': 'Avengers: Endgame',
            'genre': 'Action, Adventure, Drama',
            'rating': 8.4,
            'year': 2019,
            'director': 'Anthony Russo, Joe Russo',
            'description': 'After the devastating events of Infinity War, the Avengers assemble once more to reverse Thanos\' actions.',
            'imdb_id': 'tt4154796'
        },
        {
            'title': 'Spider-Man: Into the Spider-Verse',
            'genre': 'Animation, Action, Adventure, Family, Sci-Fi',
            'rating': 8.4,
            'year': 2018,
            'director': 'Bob Persichetti, Peter Ramsey, Rodney Rothman',
            'description': 'Teen Miles Morales becomes Spider-Man and must save the multiverse.',
            'imdb_id': 'tt4633694'
        },
        {
            'title': 'The Green Mile',
            'genre': 'Crime, Drama, Fantasy, Mystery',
            'rating': 8.6,
            'year': 1999,
            'director': 'Frank Darabont',
            'description': 'The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder.',
            'imdb_id': 'tt0120689'
        },
        {
            'title': 'Saving Private Ryan',
            'genre': 'Drama, War',
            'rating': 8.6,
            'year': 1998,
            'director': 'Steven Spielberg',
            'description': 'Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper.',
            'imdb_id': 'tt0120815'
        },
        {
            'title': 'Schindler\'s List',
            'genre': 'Biography, Drama, History',
            'rating': 8.9,
            'year': 1993,
            'director': 'Steven Spielberg',
            'description': 'In German-occupied Poland during World War II, Oskar Schindler gradually becomes concerned for his Jewish workforce.',
            'imdb_id': 'tt0108052'
        },
        {
            'title': 'Fight Club',
            'genre': 'Drama',
            'rating': 8.8,
            'year': 1999,
            'director': 'David Fincher',
            'description': 'An insomniac office worker and a devil-may-care soapmaker form an underground fight club.',
            'imdb_id': 'tt0137523'
        },
        {
            'title': 'The Lord of the Rings: The Fellowship of the Ring',
            'genre': 'Action, Adventure, Drama, Fantasy',
            'rating': 8.8,
            'year': 2001,
            'director': 'Peter Jackson',
            'description': 'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring.',
            'imdb_id': 'tt0120737'
        },
        {
            'title': 'Star Wars: Episode V - The Empire Strikes Back',
            'genre': 'Action, Adventure, Fantasy, Sci-Fi',
            'rating': 8.7,
            'year': 1980,
            'director': 'Irvin Kershner',
            'description': 'After the Rebels are brutally overpowered by the Empire, Luke Skywalker begins Jedi training.',
            'imdb_id': 'tt0080684'
        },
        {
            'title': 'One Flew Over the Cuckoo\'s Nest',
            'genre': 'Drama',
            'rating': 8.7,
            'year': 1975,
            'director': 'Milos Forman',
            'description': 'A criminal pleads insanity and is admitted to a mental institution, where he rebels against the oppressive nurse.',
            'imdb_id': 'tt0073486'
        }
    ]
    
    movies = []
    for movie_data in sample_movies:
        movie = Movie(**movie_data)
        movies.append(movie)
        db.session.add(movie)
    
    db.session.commit()
    print(f"Added {len(movies)} sample movies to database")
    return movies

def create_sample_users():
    """Create sample users for testing"""
    sample_users = [
        {
            'name': 'Demo User',
            'email': 'demo@example.com',
            'password': bcrypt.generate_password_hash('demo123').decode('utf-8')
        },
        {
            'name': 'John Doe',
            'email': 'john@example.com',
            'password': bcrypt.generate_password_hash('password123').decode('utf-8')
        },
        {
            'name': 'Jane Smith',
            'email': 'jane@example.com',
            'password': bcrypt.generate_password_hash('password123').decode('utf-8')
        },
        {
            'name': 'Movie Lover',
            'email': 'movielover@example.com',
            'password': bcrypt.generate_password_hash('movies123').decode('utf-8')
        },
        {
            'name': 'Cinema Fan',
            'email': 'cinemafan@example.com',
            'password': bcrypt.generate_password_hash('cinema123').decode('utf-8')
        }
    ]
    
    users = []
    for user_data in sample_users:
        user = User(**user_data)
        users.append(user)
        db.session.add(user)
    
    db.session.commit()
    print(f"Added {len(users)} sample users to database")
    return users

def create_sample_ratings(users, movies):
    """Create sample ratings to train the recommendation system"""
    ratings = []
    
    # Define some user preferences patterns
    user_preferences = {
        1: {'Drama': 0.8, 'Crime': 0.7, 'Biography': 0.6},  # Demo user likes dramas and crime
        2: {'Action': 0.9, 'Adventure': 0.8, 'Sci-Fi': 0.7},  # John likes action and sci-fi
        3: {'Romance': 0.8, 'Drama': 0.7, 'Family': 0.9},  # Jane likes romance and family films
        4: {'Crime': 0.9, 'Thriller': 0.8, 'Drama': 0.7},  # Movie lover likes intense films
        5: {'Sci-Fi': 0.9, 'Fantasy': 0.8, 'Adventure': 0.7}  # Cinema fan likes sci-fi and fantasy
    }
    
    for user in users:
        user_prefs = user_preferences.get(user.id, {})
        
        # Rate 60-80% of movies based on preferences
        num_ratings = random.randint(len(movies) * 6 // 10, len(movies) * 8 // 10)
        rated_movies = random.sample(movies, num_ratings)
        
        for movie in rated_movies:
            # Calculate base rating based on movie's actual rating
            base_rating = movie.rating / 2  # Convert from 10-scale to 5-scale
            
            # Adjust based on user preferences
            movie_genres = [g.strip() for g in movie.genre.split(',')]
            genre_bonus = 0
            
            for genre in movie_genres:
                if genre in user_prefs:
                    genre_bonus += user_prefs[genre] * 0.5
            
            # Add some randomness
            randomness = random.uniform(-0.3, 0.3)
            
            # Calculate final rating (1-5 scale)
            final_rating = max(1, min(5, base_rating + genre_bonus + randomness))
            final_rating = round(final_rating * 2) / 2  # Round to nearest 0.5
            
            rating = UserMovieRating(
                user_id=user.id,
                movie_id=movie.id,
                rating=final_rating,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365))
            )
            ratings.append(rating)
            db.session.add(rating)
    
    db.session.commit()
    print(f"Added {len(ratings)} sample ratings to database")
    return ratings

def create_sample_interactions(users, movies):
    """Create sample user interactions"""
    interactions = []
    interaction_types = ['view', 'click', 'like', 'share']
    
    for user in users:
        # Create 50-100 interactions per user
        num_interactions = random.randint(50, 100)
        
        for _ in range(num_interactions):
            movie = random.choice(movies)
            interaction_type = random.choice(interaction_types)
            
            # Weight interactions towards higher-rated movies
            if movie.rating > 8.5 and random.random() < 0.7:
                interaction = UserInteraction(
                    user_id=user.id,
                    movie_id=movie.id,
                    interaction_type=interaction_type,
                    timestamp=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                    metadata={'source': 'web', 'device': random.choice(['desktop', 'mobile'])}
                )
                interactions.append(interaction)
                db.session.add(interaction)
    
    db.session.commit()
    print(f"Added {len(interactions)} sample interactions to database")
    return interactions

def create_sample_profiles(users):
    """Create sample user profiles"""
    profiles = []
    
    genre_options = ['Action', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi', 'Romance', 'Horror', 'Adventure']
    decade_options = [1970, 1980, 1990, 2000, 2010, 2020]
    
    for user in users:
        # Create varied preferences for each user
        preferred_genres = random.sample(genre_options, random.randint(2, 4))
        preferred_decades = random.sample(decade_options, random.randint(1, 3))
        
        profile = UserProfile(
            user_id=user.id,
            preferred_genres=preferred_genres,
            preferred_decades=preferred_decades,
            min_rating_threshold=random.choice([6.0, 6.5, 7.0, 7.5]),
            language_preference='en',
            content_maturity=random.choice(['PG', 'PG-13', 'R']),
            recommendation_frequency=random.choice(['daily', 'weekly', 'monthly'])
        )
        profiles.append(profile)
        db.session.add(profile)
    
    db.session.commit()
    print(f"Added {len(profiles)} sample user profiles to database")
    return profiles

def init_database():
    """Initialize the database with sample data"""
    print("Initializing database...")
    
    # Create all tables
    db.create_all()
    print("Database tables created")
    
    # Check if data already exists
    if Movie.query.count() > 0:
        print("Database already contains data. Skipping initialization.")
        return
    
    # Create sample data
    movies = create_sample_movies()
    users = create_sample_users()
    ratings = create_sample_ratings(users, movies)
    interactions = create_sample_interactions(users, movies)
    profiles = create_sample_profiles(users)
    
    print("Database initialization completed successfully!")
    print(f"Summary:")
    print(f"- Movies: {len(movies)}")
    print(f"- Users: {len(users)}")
    print(f"- Ratings: {len(ratings)}")
    print(f"- Interactions: {len(interactions)}")
    print(f"- Profiles: {len(profiles)}")

def reset_database():
    """Reset the database (drop all tables and recreate)"""
    print("Resetting database...")
    db.drop_all()
    print("All tables dropped")
    init_database()

if __name__ == '__main__':
    with app.app_context():
        import sys
        
        if len(sys.argv) > 1 and sys.argv[1] == 'reset':
            reset_database()
        else:
            init_database()