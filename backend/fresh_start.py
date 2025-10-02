#!/usr/bin/env python3
"""
Fresh start script - resets database and starts clean
"""

import os
import sys

def clean_database():
    """Remove existing database files"""
    db_files = [
        'movie_recommendations.db',
        'test.db',
        'instance/movie_recommendations.db'
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"✅ Removed {db_file}")
            except Exception as e:
                print(f"⚠️  Could not remove {db_file}: {e}")
    
    # Remove instance directory if exists
    if os.path.exists('instance'):
        try:
            import shutil
            shutil.rmtree('instance')
            print("✅ Removed instance directory")
        except Exception as e:
            print(f"⚠️  Could not remove instance directory: {e}")

def test_simple_app():
    """Test the simple app"""
    print("\n🧪 Testing simple app...")
    
    try:
        from simple_app import app, db, User, Movie, bcrypt
        print("✅ App imported successfully")
        
        with app.app_context():
            # Create fresh database
            db.create_all()
            print("✅ Database created")
            
            # Add sample movies
            sample_movies = [
                {'title': 'The Shawshank Redemption', 'genre': 'Drama', 'rating': 9.3, 'year': 1994, 'director': 'Frank Darabont', 'description': 'Prison drama'},
                {'title': 'The Godfather', 'genre': 'Crime, Drama', 'rating': 9.2, 'year': 1972, 'director': 'Francis Ford Coppola', 'description': 'Crime saga'},
                {'title': 'The Dark Knight', 'genre': 'Action, Crime', 'rating': 9.0, 'year': 2008, 'director': 'Christopher Nolan', 'description': 'Batman vs Joker'},
                {'title': 'Pulp Fiction', 'genre': 'Crime, Drama', 'rating': 8.9, 'year': 1994, 'director': 'Quentin Tarantino', 'description': 'Interconnected stories'},
                {'title': 'Forrest Gump', 'genre': 'Drama, Romance', 'rating': 8.8, 'year': 1994, 'director': 'Robert Zemeckis', 'description': 'Life story'},
            ]
            
            for movie_data in sample_movies:
                movie = Movie(**movie_data)
                db.session.add(movie)
            
            # Create demo user
            demo_user = User(
                name='Demo User',
                email='demo@example.com',
                password=bcrypt.generate_password_hash('demo123').decode('utf-8')
            )
            db.session.add(demo_user)
            
            db.session.commit()
            print(f"✅ Added {len(sample_movies)} movies and demo user")
            
            # Verify data
            movie_count = Movie.query.count()
            user_count = User.query.count()
            print(f"✅ Verification: {movie_count} movies, {user_count} users")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎬 Movie Recommendation System - Fresh Start")
    print("=" * 50)
    
    # Clean old database
    print("🧹 Cleaning old database files...")
    clean_database()
    
    # Test simple app
    if test_simple_app():
        print("\n🎉 Fresh start completed successfully!")
        print("=" * 50)
        print("🚀 Next steps:")
        print("1. Start the app: python simple_app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Login with: demo@example.com / demo123")
        print("\n✅ Features available:")
        print("   - User authentication (fixed)")
        print("   - Movie database with search")
        print("   - Basic recommendations")
        print("   - Responsive design")
    else:
        print("\n❌ Fresh start failed")
        print("Please check the error messages above")

if __name__ == '__main__':
    main()