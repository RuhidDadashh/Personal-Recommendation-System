// Main JavaScript for Movie Recommendation System

// Sample movie data
const movies = [
    { id: 1, title: "The Shawshank Redemption", genre: "Drama", rating: "9.3/10" },
    { id: 2, title: "The Godfather", genre: "Crime, Drama", rating: "9.2/10" },
    { id: 3, title: "The Dark Knight", genre: "Action, Crime", rating: "9.0/10" },
    { id: 4, title: "Pulp Fiction", genre: "Crime, Drama", rating: "8.9/10" },
    { id: 5, title: "Forrest Gump", genre: "Drama, Romance", rating: "8.8/10" },
    { id: 6, title: "Inception", genre: "Sci-Fi, Thriller", rating: "8.8/10" },
    { id: 7, title: "The Matrix", genre: "Sci-Fi, Action", rating: "8.7/10" },
    { id: 8, title: "Goodfellas", genre: "Crime, Drama", rating: "8.7/10" },
    { id: 9, title: "Interstellar", genre: "Sci-Fi, Drama", rating: "8.6/10" },
    { id: 10, title: "The Lion King", genre: "Animation, Family", rating: "8.5/10" },
    { id: 11, title: "Parasite", genre: "Thriller, Drama", rating: "8.6/10" },
    { id: 12, title: "Avengers: Endgame", genre: "Action, Adventure", rating: "8.4/10" }
];

// Global variables
let selectedMovies = [];
let currentUser = null;

// Initialize the app
function init() {
    populateMoviesGrid();
    loadUserData();
}

// Load user data from session storage (if available)
function loadUserData() {
    const userData = sessionStorage.getItem('currentUser');
    if (userData) {
        currentUser = JSON.parse(userData);
        updateUserDisplay();
    }
}

// Update user display
function updateUserDisplay() {
    if (currentUser) {
        const userNameElement = document.getElementById('currentUser');
        if (userNameElement) {
            userNameElement.textContent = currentUser.name || currentUser.email.split('@')[0];
        }
    }
}

// Populate movies grid
function populateMoviesGrid() {
    const grid = document.getElementById('moviesGrid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    movies.forEach(movie => {
        const movieCard = document.createElement('div');
        movieCard.className = 'movie-card';
        movieCard.onclick = () => selectMovie(movie.id, movieCard);
        
        movieCard.innerHTML = `
            <div class="movie-poster">🎬</div>
            <div class="movie-info">
                <div class="movie-title">${movie.title}</div>
                <div class="movie-genre">${movie.genre}</div>
                <div class="movie-rating">★ ${movie.rating}</div>
            </div>
        `;
        
        grid.appendChild(movieCard);
    });
}

// Movie selection logic
function selectMovie(movieId, cardElement) {
    if (selectedMovies.includes(movieId)) {
        // Deselect movie
        selectedMovies = selectedMovies.filter(id => id !== movieId);
        cardElement.classList.remove('selected');
    } else if (selectedMovies.length < 4) {
        // Select movie
        selectedMovies.push(movieId);
        cardElement.classList.add('selected');
    } else {
        showAlert('You can select up to 4 movies only!', 'error');
    }
    
    updateRecommendButton();
}

// Update recommendation button state
function updateRecommendButton() {
    const btn = document.getElementById('getRecommendationsBtn');
    if (btn) {
        btn.textContent = selectedMovies.length > 0 
            ? `Get Recommendations (${selectedMovies.length} selected)` 
            : 'Get My Recommendations';
    }
}

// Get recommendations
function getRecommendations() {
    if (selectedMovies.length === 0) {
        showAlert('Please select at least one movie!', 'error');
        return;
    }

    // Show loading state
    const btn = document.getElementById('getRecommendationsBtn');
    const originalText = btn.textContent;
    btn.textContent = 'Getting Recommendations...';
    btn.disabled = true;
    btn.classList.add('loading');

    // Simulate API call to backend
    setTimeout(() => {
        fetchRecommendations()
            .then(recommendations => {
                displayRecommendations(recommendations);
                btn.textContent = originalText;
                btn.disabled = false;
                btn.classList.remove('loading');
            })
            .catch(error => {
                console.error('Error fetching recommendations:', error);
                showAlert('Error getting recommendations. Please try again.', 'error');
                btn.textContent = originalText;
                btn.disabled = false;
                btn.classList.remove('loading');
            });
    }, 2000);
}

// Simulate API call for recommendations
function fetchRecommendations() {
    return new Promise((resolve) => {
        // Simple recommendation logic (in real app, this would come from backend)
        const recommendations = [
            { title: "Blade Runner 2049", genre: "Sci-Fi", rating: "8.0/10" },
            { title: "The Departed", genre: "Crime, Thriller", rating: "8.5/10" },
            { title: "Mad Max: Fury Road", genre: "Action", rating: "8.1/10" },
            { title: "Her", genre: "Romance, Sci-Fi", rating: "8.0/10" },
            { title: "Whiplash", genre: "Drama", rating: "8.5/10" },
            { title: "Ex Machina", genre: "Sci-Fi", rating: "7.7/10" }
        ];
        
        resolve(recommendations);
    });
}

// Display recommendations
function displayRecommendations(recommendations) {
    const section = document.getElementById('recommendationsSection');
    const grid = document.getElementById('recommendationsGrid');
    
    if (!section || !grid) return;
    
    grid.innerHTML = '';
    
    recommendations.forEach((rec, index) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.style.animationDelay = `${index * 0.1}s`;
        card.innerHTML = `
            <div style="font-size: 2rem; margin-bottom: 10px;">🎭</div>
            <h3 style="margin-bottom: 8px; color: #333;">${rec.title}</h3>
            <p style="color: #666; margin-bottom: 5px;">${rec.genre}</p>
            <p style="color: #ff6b6b; font-weight: bold;">★ ${rec.rating}</p>
        `;
        grid.appendChild(card);
    });

    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth' });
}

// Logout function
function logout() {
    currentUser = null;
    selectedMovies = [];
    sessionStorage.removeItem('currentUser');
    
    // Hide recommendations
    const recommendationsSection = document.getElementById('recommendationsSection');
    if (recommendationsSection) {
        recommendationsSection.classList.add('hidden');
    }
    
    // Redirect to login
    window.location.href = 'login.html';
}

// Show alert messages
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    // Insert at the top of the page
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Remove alert after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
}

// Check if user is authenticated
function checkAuth() {
    const userData = sessionStorage.getItem('currentUser');
    if (!userData) {
        window.location.href = 'login.html';
        return false;
    }
    return true;
}

// Utility function to get selected movie titles
function getSelectedMovieTitles() {
    return selectedMovies.map(id => {
        const movie = movies.find(m => m.id === id);
        return movie ? movie.title : '';
    }).filter(title => title !== '');
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    init();
});

// Export functions for use in other files
window.MovieApp = {
    init,
    selectMovie,
    getRecommendations,
    logout,
    showAlert,
    checkAuth
};